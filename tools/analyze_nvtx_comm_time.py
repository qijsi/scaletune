#!/usr/bin/env python3
"""Roll up NCCL GPU time by parallel dimension using NVTX-like Chrome trace ranges.

Megatron ``enhanced_comm_profiler`` can wrap collectives with ``torch.cuda.nvtx``
labels like ``tp|AR``, ``pp|P2P`` (see ``megatron/core/distributed/enhanced_comm_profiler.py``).

PyTorch Profiler Chrome traces also emit ``nccl:*`` as ``user_annotation`` (CPU)
and ``gpu_user_annotation`` (GPU).  CPU NVTX and GPU kernels use **different**
``pid`` rows; this script matches by **global timestamp** (innermost enclosing
interval), not by ``pid``.

Only ``cat == "kernel"`` events are counted as communication GPU time (not
``cpu_op`` / ``python_function`` rows such as ``c10d::...``).

When PyTorch records NCCL kernels, ``args`` often includes **Process Group Description**
(registered name such as ``TENSOR_MODEL_PARALLEL_GROUP``).  Rollup **prefers** this
field to map GPU time to parallel dimensions, which fixes the common case where
time-based NVTX nesting attributes work to short ``nccl:*`` rows instead of
Megatron ``tp|AR`` (async launch).

Kernels bucketed as ``param_comms`` (``cpu_op record_param_comms`` + ``default_pg``)
can be **refined** onto Megatron ``dim|op`` intervals by **overlap** or by **minimum
temporal gap** (see ``--param-comms-nearest-megatron-us``) when GPU work is async
relative to ``record_function``.

**Accuracy tips (most important first):**

1. Enable Megatron enhanced comm profiling so every wrapped collective emits
   ``tp|AR``-style labels: set ``MEGATRON_ENHANCED_COMM_PROFILE=1`` (or
   ``SCALETUNE_ENHANCED_COMM_PROFILE=1``) before training; see
   ``enhanced_comm_profiler.maybe_install_enhanced_comm_profiler``.
2. Prefer **runtime** rollup from ``CommRecord`` / ``finalize_comm_record`` when you
   need causal labels (CPU-side ``parallel_dim``), even if GPU time differs from
   wall time around the call.
3. For trace-only GPU time that must line up with Python ranges, use a **profiling
   build** that synchronizes the CUDA device after each collective (or use Nsight
   Systems + NVTX on the same timeline); otherwise async NCCL launch can leave
   kernels **outside** any ``record_function`` window — no post-hoc matcher can
   reliably fix that without heuristics.

Usage::

    python scaletune/analyze_nvtx_comm_time.py --trace-dir torch_profiler_traces_qwen2-7b
    python scaletune/analyze_nvtx_comm_time.py --trace-dir /path/to/traces --output-dir rollup_out
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Must stay in sync with _NVTX_DIM_ALIASES in enhanced_comm_profiler.py
_NVXT_TO_PARALLEL_DIM: Dict[str, str] = {
    "tp": "tp",
    "dp": "dp",
    "pp": "pp",
    "cp": "cp",
    "sp": "sp",
    "ep": "ep",
    "etp": "etp",
    "edp": "edp",
    "embp": "embp",
    "dp1": "dp_single",
    "tpDp": "tp_dp",
    "tpCp": "tp_cp",
    "dpCp": "dp_cp",
    "tpSp": "tp_sp",
    "tpDpSp": "tp_dp_sp",
    "tpEpPp": "tp_ep_pp",
    "tpEp": "tp_ep",
    "tpPp": "tp_pp",
    "tpPpG": "tp_pp_grad",
    "tpDpCp": "tp_dp_cp",
}

_PYTORCH_NCCL_PREFIX = "pytorch_nccl_"

# PyTorch / Megatron register ProcessGroup with ``group_desc=...``; Kineto copies it
# into each NCCL kernel's ``args['Process Group Description']``.  This is the most
# reliable way to attribute GPU time to a parallel axis when present (independent of
# async NVTX / ``record_function`` nesting).
_PG_DESC_TO_PARALLEL_DIM: Dict[str, str] = {
    "TENSOR_MODEL_PARALLEL_GROUP": "tp",
    "PIPELINE_MODEL_PARALLEL_GROUP": "pp",
    "DATA_PARALLEL_GROUP": "dp",
    "CONTEXT_PARALLEL_GROUP": "cp",
    "MODEL_PARALLEL_GROUP": "tp_pp",
    "EXPERT_MODEL_PARALLEL_GROUP": "ep",
    "EXPERT_TENSOR_PARALLEL_GROUP": "etp",
    "EXPERT_DATA_PARALLEL_GROUP": "edp",
    "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP": "tp_ep",
    "EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP": "tp_ep_pp",
    "TENSOR_AND_DATA_PARALLEL_GROUP": "tp_dp",
    "TENSOR_AND_CONTEXT_PARALLEL_GROUP": "tp_cp",
    "DATA_PARALLEL_GROUP_WITH_CP": "dp_cp",
    "INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP": "dp_cp",
    "TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP": "tp_dp_cp",
    "EMBEDDING_GROUP": "embed",
    "POSITION_EMBEDDING_GROUP": "pos_embed",
    "default_pg": "default_pg",
}


def build_external_id_cpu_op_hints(events: List[Dict[str, Any]]) -> Dict[int, str]:
    """Map ``External id`` to a **neutral** rollup bucket from Kineto cpu_op names.

    PyTorch links NCCL kernels to initiating ops via ``args['External id']``.  When
    ``Process Group Description`` is ``default_pg``, we can recover the **cpu_op**
    name (e.g. ``record_param_comms``).

    **Important:** ``record_param_comms`` means *parameter communication* in the
    PyTorch distributed stack (DDP/FSDP/optimizer hooks). It is **not** the same as
    Megatron's **data-parallel axis** (``DATA_PARALLEL_GROUP``).  Mapping it to
    ``dp`` was wrong when e.g. ``--data-parallel-size 1`` — that time is **not** DP
    in the Megatron sense.  We use a dedicated bucket ``param_comms`` instead.
    """
    hints: Dict[int, str] = {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat") or ""
        if cat not in ("cpu_op", "python_function", "user_annotation"):
            continue
        args = ev.get("args") or {}
        eid = args.get("External id")
        if eid is None:
            continue
        try:
            eid_i = int(eid)
        except (TypeError, ValueError):
            continue
        name = str(ev.get("name", ""))
        dim: Optional[str] = None
        if "record_param_comms" in name:
            dim = "param_comms"
        elif "grad" in name.lower() and "bucket" in name.lower():
            dim = "grad_bucket_comm"
        if dim is not None:
            hints[eid_i] = dim
    return hints


def parallel_dim_from_pytorch_kernel_args(args: Any) -> Optional[str]:
    """Map Kineto NCCL kernel ``args`` to a canonical parallel dimension key, if possible.

    Returns ``None`` when ``Process Group Description`` is absent so callers can fall
    back to NVTX / time-based attribution.
    """
    if not isinstance(args, dict):
        return None
    desc = args.get("Process Group Description")
    if desc is None:
        return None
    if not isinstance(desc, str):
        desc = str(desc)
    desc = desc.strip()
    if not desc:
        return None
    mapped = _PG_DESC_TO_PARALLEL_DIM.get(desc)
    if mapped is not None:
        return mapped
    # Unknown but non-empty PG name from PyTorch — keep a stable bucket for inspection.
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", desc)[:48]
    return f"pg_desc:{safe}"


def _is_gpu_nccl_kernel(name: str) -> bool:
    n = (name or "").lower()
    if "nccl" in n:
        return True
    return any(x in n for x in ("c10d::", "allreduce", "all_reduce", "reduce_scatter", "all_gather"))


def _classify_nvtx_label(label: str) -> Tuple[str, str]:
    raw = (label or "").strip()
    if not raw:
        return ("unresolved", "")
    if raw.lower().startswith("nccl:"):
        op = raw.split(":", 1)[-1].replace(" ", "_")
        return (f"{_PYTORCH_NCCL_PREFIX}{op.lower()}", op)
    parts = raw.split("|", 1)
    dim_key = parts[0]
    op_short = parts[1] if len(parts) > 1 else "?"
    parallel_dim = _NVXT_TO_PARALLEL_DIM.get(dim_key, dim_key)
    return (parallel_dim, op_short)


def _is_nvtx_range_for_comm_matching(name: str) -> bool:
    n = (name or "").lower()
    if not n:
        return False
    if "|" in name:
        return True
    if n.startswith("nccl:"):
        return True
    return False


def _is_megatron_parallel_nvtx(name: str) -> bool:
    """True for ``enhanced_comm_profiler`` labels like ``tp|AR``, ``tpPpG|P2P`` (not ``nccl:*``)."""
    n = (name or "").strip()
    if "|" not in n or n.lower().startswith("nccl:"):
        return False
    left, right = n.split("|", 1)
    if not left or not right:
        return False
    # Known short dim token from _NVTX_DIM_ALIASES, or legacy ``tp`` / ``pp`` style.
    if left in _NVXT_TO_PARALLEL_DIM:
        return True
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9]{0,15}$", left)) and len(right) <= 8


def find_trace_files(trace_dir: str, max_files: int = 0) -> List[str]:
    trace_dir = os.path.abspath(trace_dir)
    if not os.path.isdir(trace_dir):
        print(f"Trace directory not found: {trace_dir}")
        return []

    candidates: List[str] = []
    for pat in (
        os.path.join(trace_dir, "*.json"),
        os.path.join(trace_dir, "*", "*.json"),
        os.path.join(trace_dir, "*.json.gz"),
        os.path.join(trace_dir, "*", "*.json.gz"),
    ):
        candidates.extend(glob.glob(pat))

    out: List[str] = []
    for p in sorted(set(candidates)):
        if not os.path.isfile(p):
            continue
        base = os.path.basename(p).lower()
        if ".pt.trace" in base or "trace" in base:
            out.append(p)
    if not out:
        for p in sorted(set(candidates)):
            if os.path.isfile(p) and p.endswith(".json"):
                out.append(p)

    if max_files > 0:
        out = out[:max_files]
    return out


def load_trace(filepath: str) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  [WARN] Cannot load {filepath}: {exc}")
        return [], None

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    rank: Optional[int] = None
    for ev in events[:500]:
        args = ev.get("args")
        if isinstance(args, dict):
            r = args.get("Rank") or args.get("rank")
            if r is not None:
                rank = int(r)
                break
    if rank is None:
        m = re.search(r"rank(\d+)", os.path.basename(filepath), re.IGNORECASE)
        if m:
            rank = int(m.group(1))
    return events, rank


def _collect_nvtx_intervals_flat(events: List[Dict[str, Any]]) -> List[Tuple[int, int, Dict]]:
    """Collect time intervals used to parent NCCL kernels.

    - PyTorch ``nccl:*`` on ``user_annotation`` / ``gpu_user_annotation``.
    - Megatron parallel labels: ``torch.profiler.record_function`` emits
      ``python_function`` events named ``tp|AR``, ``pp|P2P``, etc. (NVTX text alone
      is often absent from Chrome JSON.)
    """
    out: List[Tuple[int, int, Dict]] = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat") or ""
        name = str(ev.get("name", ""))
        if cat == "python_function" and _is_megatron_parallel_nvtx(name):
            include = True
        elif cat in ("user_annotations", "user_annotation", "gpu_user_annotation") and _is_nvtx_range_for_comm_matching(
            name
        ):
            include = True
        else:
            include = False
        if not include:
            continue
        ts = int(float(ev.get("ts", 0) or 0))
        dur = int(float(ev.get("dur", 0) or 0))
        if dur <= 0:
            continue
        out.append((ts, ts + dur, ev))
    out.sort(key=lambda x: x[0])
    return out


def _interval_separation_us(k0: int, k1: int, s: int, e: int) -> int:
    """Non-negative gap between [k0,k1] and [s,e] in µs; ``0`` if they overlap or touch."""
    if k1 <= k0:
        k1 = k0
    if e <= s:
        return 10**18
    if k1 < s:
        return s - k1
    if k0 > e:
        return k0 - e
    return 0


def _megatron_only_intervals(
    nvtx_intervals_flat: List[Tuple[int, int, Dict]],
) -> List[Tuple[int, int, Dict]]:
    """Intervals whose names are Megatron ``dim|op`` (exclude PyTorch ``nccl:*``)."""
    return [
        t
        for t in nvtx_intervals_flat
        if _is_megatron_parallel_nvtx(str(t[2].get("name", "")))
    ]


def _refine_param_bucket_with_megatron_time(
    dim: str,
    src: str,
    k0: int,
    k1: int,
    megatron_intervals: List[Tuple[int, int, Dict]],
    nearest_max_gap_us: int,
) -> Tuple[str, str]:
    """Try to move ``param_comms`` / ``grad_bucket_comm`` into a Megatron parallel dim.

    1. **Overlap**: kernel window intersects a ``tp|AR``-style interval (strict).
    2. **Nearest interval**: if (1) fails, pick the Megatron interval with minimum
       temporal **gap** to the kernel window; if gap ≤ ``nearest_max_gap_us``, use it.

    This addresses async launch: the GPU NCCL kernel often starts *after* the short
    ``record_function`` window, so containment fails but the gap to the matching
    ``tpPpG|AR`` (etc.) is small (milliseconds).
    """
    if dim not in ("param_comms", "grad_bucket_comm"):
        return dim, src
    if not megatron_intervals:
        return dim, src

    # (1) Same overlap logic as the main matcher, but Megatron-only parents.
    parent = _find_best_parent_for_kernel(megatron_intervals, k0, k1 - k0 if k1 > k0 else 0)
    if parent is not None and _is_megatron_parallel_nvtx(str(parent.get("name", ""))):
        d, _ = _classify_nvtx_label(str(parent.get("name", "")))
        return d, "nvtx_megatron_overlap_refine"

    if nearest_max_gap_us <= 0:
        return dim, src

    best_gap = 10**18
    best_ev: Optional[Dict[str, Any]] = None
    best_span = 10**18
    for s, e, ev in megatron_intervals:
        g = _interval_separation_us(k0, k1, s, e)
        span = e - s
        if g < best_gap or (g == best_gap and span < best_span):
            best_gap = g
            best_span = span
            best_ev = ev

    if best_ev is None or best_gap > nearest_max_gap_us:
        return dim, src

    d, _ = _classify_nvtx_label(str(best_ev.get("name", "")))
    return d, "nvtx_megatron_nearest_gap_refine"


def _interval_overlap_us(k0: int, k1: int, s: int, e: int) -> int:
    """Overlap length between kernel [k0, k1] and parent [s, e] (all µs, inclusive ends).

    Chrome trace ``dur`` is on the same clock as ``ts``; we treat the kernel window as
    **[k0, k1]** with ``k1 = ts + dur``. For a **degenerate** kernel (``k1 <= k0``), use
    **containment** of the start time so point-like events still match parents.
    """
    if k1 <= k0:
        return 1 if s <= k0 <= e else 0
    return max(0, min(k1, e) - max(k0, s))


def _pick_innermost_megatron_else_shortest(
    candidates: List[Tuple[int, int, Dict]],
) -> Tuple[int, int, Dict]:
    """Prefer Megatron ``dim|op`` parents; tie-break by shortest span (innermost)."""
    meg = [t for t in candidates if _is_megatron_parallel_nvtx(str(t[2].get("name", "")))]
    pool = meg if meg else candidates
    return min(pool, key=lambda t: t[1] - t[0])


def _find_best_parent_for_kernel(
    intervals: List[Tuple[int, int, Dict]],
    ts: int,
    dur_us: int = 0,
) -> Optional[Dict[str, Any]]:
    """Pick the best NVTX / ``python_function`` parent for an NCCL kernel.

    Matching strategy (most accurate for post-hoc Chrome JSON without CUPTI):

    1. **Time overlap**: Among parents whose time window overlaps the kernel window
       ``[ts, ts+dur]``, prefer Megatron ``dim|op`` labels, then maximize overlap
       (then shorter span). This catches kernels that **span** CPU/Python range
       boundaries; pure point-in-time containment at ``ts`` can miss those.
    2. If no overlap (typical when the GPU kernel **starts after** the CPU range
       ends — async launch), fall back to containment at **kernel start**, then at
       **midpoint** ``ts + dur/2`` (mitigates minor clock / ordering quirks).

    For **maximum** attribution accuracy when overlap stays empty, enable profiling-only
    synchronization in the trainer (see module docstring) so CPU ``record_function``
    windows cover GPU execution.

    PyTorch emits **short** ``gpu_user_annotation`` / ``nccl:*`` windows on the GPU
    row; Megatron wraps the collective with **longer** ``tp|AR`` CPU ranges. Taking
    the globally shortest interval always picked ``nccl:*`` and hid TP/PP/DP; we
    **prefer** overlapping Megatron labels first.
    """
    if not intervals:
        return None

    k0 = ts
    k1 = ts + max(0, int(dur_us))

    # 1) Overlap-based (skip empty kernel duration only for containment via degenerate rule)
    scored: List[Tuple[int, int, int, int, Dict]] = []
    for s, e, ev in intervals:
        ov = _interval_overlap_us(k0, k1, s, e)
        if ov > 0:
            scored.append((ov, s, e, e - s, ev))

    if scored:
        meg = [t for t in scored if _is_megatron_parallel_nvtx(str(t[4].get("name", "")))]
        pool = meg if meg else scored
        # Max overlap µs, then shorter parent span (innermost).
        pool.sort(key=lambda t: (-t[0], t[3]))
        return pool[0][4]

    # 2) Point-in-time containment: start, then midpoint (helps rare ordering quirks)
    for probe in (ts, (ts + k1) // 2 if k1 > ts else ts):
        enclosing = [(s, e, ev) for s, e, ev in intervals if s <= probe <= e]
        if enclosing:
            _s, _e, best_ev = _pick_innermost_megatron_else_shortest(enclosing)
            return best_ev
    return None


def _kernel_cuda_time_us(ev: Dict[str, Any]) -> float:
    args = ev.get("args")
    if isinstance(args, dict):
        v = args.get("self_cuda_time_total")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return float(ev.get("dur", 0) or 0)


def _resolve_parallel_dim_for_nccl_kernel(
    ev: Dict[str, Any],
    parent: Optional[Dict[str, Any]],
    external_id_hints: Optional[Dict[int, str]] = None,
) -> Tuple[str, str]:
    """Return (canonical_dim, attribution_source).

    Source is one of: ``kernel_pg_desc``, ``nvtx_megatron``, ``nvtx_pytorch_nccl``,
    ``nvtx_unresolved``, ``mixed_default_pg``, ``kernel_external_id_hint``.
    """
    args = ev.get("args") if isinstance(ev.get("args"), dict) else {}
    pg_dim = parallel_dim_from_pytorch_kernel_args(args)

    eid_hint: Optional[str] = None
    if external_id_hints:
        raw_eid = args.get("External id")
        if raw_eid is not None:
            try:
                eid_hint = external_id_hints.get(int(raw_eid))
            except (TypeError, ValueError):
                pass

    nv_dim: Optional[str] = None
    nv_megatron = False
    if parent is not None:
        pname = str(parent.get("name", ""))
        nv_megatron = _is_megatron_parallel_nvtx(pname)
        nv_dim, _ = _classify_nvtx_label(pname)

    # 1) Kineto NCCL kernel metadata: Megatron-registered process groups (non-default).
    if pg_dim is not None and pg_dim != "default_pg" and not pg_dim.startswith("pg_desc:"):
        return pg_dim, "kernel_pg_desc"

    # 2) Prefer Megatron ``tp|AR`` parents when PG metadata is missing or only default_pg.
    if nv_megatron and nv_dim is not None:
        return nv_dim, "nvtx_megatron"

    # 3) default_pg: try External id → cpu_op hint (e.g. param comms → DP) before PyTorch nccl:*.
    if pg_dim == "default_pg":
        if eid_hint is not None:
            return eid_hint, "kernel_external_id_hint"
        if nv_dim is not None and nv_dim not in ("unresolved",):
            if nv_dim.startswith(_PYTORCH_NCCL_PREFIX):
                return nv_dim, "nvtx_pytorch_nccl"
            return nv_dim, "mixed_default_pg"
        return "default_pg", "kernel_pg_desc"

    if pg_dim is not None and pg_dim.startswith("pg_desc:"):
        if eid_hint is not None:
            return eid_hint, "kernel_external_id_hint"
        if nv_megatron and nv_dim is not None:
            return nv_dim, "nvtx_megatron"
        return pg_dim, "kernel_pg_desc"

    # 4) No Process Group Description in args — External id hint, then NVTX.
    if eid_hint is not None:
        return eid_hint, "kernel_external_id_hint"
    if nv_dim is None:
        return "unresolved", "nvtx_unresolved"
    if nv_dim.startswith(_PYTORCH_NCCL_PREFIX):
        return nv_dim, "nvtx_pytorch_nccl"
    return nv_dim, "nvtx_megatron" if nv_megatron else "nvtx_other"


def aggregate_nvtx_comm_time(
    events: List[Dict[str, Any]],
    nvtx_intervals_flat: List[Tuple[int, int, Dict]],
    external_id_hints: Optional[Dict[int, str]] = None,
    param_comms_nearest_megatron_us: int = 25_000,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, int]]:
    global_rollup: Dict[str, float] = defaultdict(float)
    per_rank_rollup: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    stats: Dict[str, Any] = {
        "kernels_comm": 0,
        "kernels_with_parent": 0,
        "kernels_megatron_labeled": 0,
        "kernels_from_kernel_pg_desc": 0,
        "kernels_from_external_id_hint": 0,
        "kernels_refined_param_overlap": 0,
        "kernels_refined_param_nearest_gap": 0,
        "nvtx_intervals_used": len(nvtx_intervals_flat),
        "param_comms_nearest_megatron_us": param_comms_nearest_megatron_us,
    }
    by_source: Dict[str, float] = defaultdict(float)
    meg_only = _megatron_only_intervals(nvtx_intervals_flat)
    stats["megatron_intervals_only"] = len(meg_only)

    default_rank: Optional[int] = None
    pid_to_rank: Dict[str, int] = {}
    for ev in events[:5000]:
        args = ev.get("args")
        if isinstance(args, dict):
            r = args.get("Rank") or args.get("rank")
            if r is not None:
                default_rank = int(r)
                pid_to_rank[str(ev.get("pid", ""))] = int(r)
                break

    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "kernel":
            continue
        if not _is_gpu_nccl_kernel(str(ev.get("name", ""))):
            continue

        cuda_us = _kernel_cuda_time_us(ev)
        if cuda_us <= 0:
            continue
        stats["kernels_comm"] += 1

        ts = int(float(ev.get("ts", 0) or 0))
        dur_ev = int(float(ev.get("dur", 0) or 0))
        if dur_ev <= 0:
            cuda_us = _kernel_cuda_time_us(ev)
            dur_ev = int(round(cuda_us)) if cuda_us > 0 else 0

        parent = _find_best_parent_for_kernel(nvtx_intervals_flat, ts, dur_ev)
        if parent is not None:
            stats["kernels_with_parent"] += 1
            pname = str(parent.get("name", ""))
            if _is_megatron_parallel_nvtx(pname):
                stats["kernels_megatron_labeled"] += 1

        dim, src = _resolve_parallel_dim_for_nccl_kernel(ev, parent, external_id_hints)
        if src == "kernel_pg_desc":
            stats["kernels_from_kernel_pg_desc"] += 1
        if src == "kernel_external_id_hint":
            stats["kernels_from_external_id_hint"] += 1

        k0 = ts
        k1 = ts + max(0, dur_ev)
        dim0, src0 = dim, src
        dim, src = _refine_param_bucket_with_megatron_time(
            dim, src, k0, k1, meg_only, param_comms_nearest_megatron_us
        )
        if (dim0, src0) != (dim, src):
            if src == "nvtx_megatron_overlap_refine":
                stats["kernels_refined_param_overlap"] += 1
            elif src == "nvtx_megatron_nearest_gap_refine":
                stats["kernels_refined_param_nearest_gap"] += 1

        global_rollup[dim] += cuda_us
        by_source[src] += cuda_us

        rank = pid_to_rank.get(str(ev.get("pid", "")))
        if rank is None:
            rank = default_rank
        if rank is not None:
            per_rank_rollup[str(rank)][dim] += cuda_us

    stats["attribution_cuda_us_by_source"] = {k: float(v) for k, v in by_source.items()}
    return dict(global_rollup), {k: dict(v) for k, v in per_rank_rollup.items()}, stats


_DIM_LABELS: Dict[str, str] = {
    "tp": "Tensor Parallel",
    "pp": "Pipeline Parallel",
    "dp": "Data Parallel",
    "dp_single": "Data Parallel (WS=1)",
    "cp": "Context Parallel",
    "sp": "Sequence Parallel",
    "ep": "Expert Parallel",
    "etp": "Expert TP",
    "edp": "Expert DP",
    "embp": "Embedding Parallel",
    "tp_dp": "TP+DP",
    "tp_cp": "TP+CP",
    "dp_cp": "DP+CP",
    "tp_sp": "TP+SP",
    "tp_dp_sp": "TP+DP+SP",
    "tp_ep_pp": "TP+EP+PP",
    "tp_ep": "TP+EP",
    "tp_pp": "TP+PP (mesh)",
    "tp_pp_grad": "TP+PP (grad)",
    "tp_dp_cp": "TP+DP+CP",
    "unresolved": "Unresolved (no NVTX parent)",
    "default_pg": "Default process group (see kernel metadata)",
    "param_comms": "Parameter comm (cpu_op record_param_comms; not Megatron DP)",
    "grad_bucket_comm": "Grad bucket comm (cpu_op heuristic; not a mesh axis)",
}


def _dim_display_name(dim: str) -> str:
    if dim.startswith(_PYTORCH_NCCL_PREFIX):
        return f"PyTorch NCCL ({dim[len(_PYTORCH_NCCL_PREFIX):]})"
    if dim.startswith("pg_desc:"):
        return f"Process group ({dim})"
    return _DIM_LABELS.get(dim, dim)


def print_rollup(
    global_rollup: Dict[str, float],
    per_rank_rollup: Dict[str, Dict[str, float]],
    output_dir: Optional[str] = None,
    attribution_by_source: Optional[Dict[str, float]] = None,
) -> None:
    def _us_to_ms(us: float) -> float:
        return us / 1000.0

    print("\n" + "=" * 80)
    print("  Communication CUDA Time by Parallel Dimension (via NVTX / nccl ranges)")
    print("=" * 80)
    total_us = sum(global_rollup.values())
    if total_us == 0:
        print("  No NCCL GPU kernel time found.")
        print("  Check trace path and that the run recorded CUDA kernels.")
        return

    print(
        "\n  Attribution order: (1) ``Process Group Description`` on the kernel, "
        "(2) Megatron ``tp|AR`` / NVTX parents, (3) ``External id`` → ``param_comms`` "
        "(not Megatron DP), (4) PyTorch ``nccl:*``.\n"
        "  **param_comms refinement**: remaining ``param_comms`` / ``grad_bucket_comm`` "
        "time can be reassigned to the closest Megatron ``dim|op`` interval by **overlap**, "
        "else by **minimum temporal gap** (see ``--param-comms-nearest-megatron-us``). "
        "True **dp** requires ``DATA_PARALLEL_GROUP`` on the kernel."
    )

    print(f"\n  {'Dimension':<25} {'CUDA Time (ms)':>15} {'Share':>8}")
    print("  " + "-" * 50)

    for dim in sorted(global_rollup.keys(), key=lambda d: -global_rollup[d]):
        us = global_rollup[dim]
        pct = us / total_us * 100 if total_us > 0 else 0
        print(f"  {_dim_display_name(dim):<25} {_us_to_ms(us):>15.2f} {pct:>7.1f}%")

    print(f"  {'TOTAL':<25} {_us_to_ms(total_us):>15.2f} {'100.0%':>8}")

    if per_rank_rollup:
        print("\n" + "-" * 80)
        print("  Per-Rank Breakdown")
        print("-" * 80)
        ranks = sorted(per_rank_rollup.keys(), key=lambda r: int(r))
        dims_all = sorted(
            {d for rd in per_rank_rollup.values() for d in rd},
            key=lambda d: -global_rollup.get(d, 0),
        )
        header = f"  {'Rank':>6}"
        for d in dims_all[:12]:
            header += f" {_dim_display_name(d)[:12]:>12}"
        print(header)
        print("  " + "-" * min(len(header) - 2, 120))
        for rank in ranks:
            rd = per_rank_rollup[rank]
            row = f"  {rank:>6}"
            for d in dims_all[:12]:
                row += f" {_us_to_ms(rd.get(d, 0)):>12.2f}"
            print(row)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        jpath = os.path.join(output_dir, "nvtx_comm_rollup.json")
        with open(jpath, "w", encoding="utf-8") as f:
            payload: Dict[str, Any] = {
                "global_cuda_us": global_rollup,
                "per_rank_cuda_us": per_rank_rollup,
                "dim_labels": _DIM_LABELS,
            }
            if attribution_by_source:
                payload["attribution_cuda_us_by_source"] = attribution_by_source
            json.dump(payload, f, indent=2)
        print(f"\n  JSON export: {jpath}")

        cpath = os.path.join(output_dir, "nvtx_comm_rollup_global.csv")
        with open(cpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["parallel_dimension", "display_name", "cuda_time_us", "cuda_time_ms", "share_pct"])
            for dim in sorted(global_rollup, key=lambda d: -global_rollup[d]):
                us = global_rollup[dim]
                w.writerow(
                    [
                        dim,
                        _dim_display_name(dim),
                        f"{us:.1f}",
                        f"{_us_to_ms(us):.2f}",
                        f"{us / total_us * 100:.1f}" if total_us > 0 else "0.0",
                    ]
                )
        print(f"  CSV export:  {cpath}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roll up NCCL GPU kernel time using NVTX / nccl Chrome trace ranges."
    )
    parser.add_argument("--trace-dir", default=None, help="Directory with PyTorch Profiler trace JSON.")
    parser.add_argument("--output-dir", default=None, help="Write JSON + CSV here.")
    parser.add_argument("--max-files", type=int, default=0, help="Max trace files (0 = all).")
    parser.add_argument(
        "--param-comms-nearest-megatron-us",
        type=int,
        default=25_000,
        help="After labeling ``param_comms`` / ``grad_bucket_comm``, reassign to the "
        "closest Megatron ``dim|op`` interval if the temporal gap (µs) is at most this "
        "value. Default 25000 (25 ms). Use 0 to disable nearest-gap refinement.",
    )
    args = parser.parse_args()

    trace_dir = args.trace_dir
    if trace_dir is None:
        for cand in (
            "torch_profiler_traces_qwen2-7b",
            "torch_profiler_traces_llama7b",
            "torch_profiler_traces_mamba8b",
        ):
            if os.path.isdir(cand):
                trace_dir = cand
                break
    if trace_dir is None:
        print("No trace directory specified. Use --trace-dir.")
        sys.exit(1)

    trace_files = find_trace_files(trace_dir, max_files=args.max_files)
    if not trace_files:
        print(f"No trace files found under {trace_dir}")
        sys.exit(1)

    print(f"Found {len(trace_files)} trace file(s) under {trace_dir}")

    global_total: Dict[str, float] = defaultdict(float)
    per_rank_total: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    global_attribution_by_source: Dict[str, float] = defaultdict(float)

    for tf in trace_files:
        print(f"  Processing {os.path.basename(tf)} ...")
        events, rank_hint = load_trace(tf)
        if not events:
            continue
        nvtx_flat = _collect_nvtx_intervals_flat(events)
        ext_hints = build_external_id_cpu_op_hints(events)
        file_global, file_per_rank, stats = aggregate_nvtx_comm_time(
            events,
            nvtx_flat,
            external_id_hints=ext_hints,
            param_comms_nearest_megatron_us=args.param_comms_nearest_megatron_us,
        )
        by_src = stats.get("attribution_cuda_us_by_source") or {}
        print(
            f"    NVTX-like intervals: {stats['nvtx_intervals_used']}, "
            f"Megatron-only intervals: {stats.get('megatron_intervals_only', 0)}, "
            f"comm kernels: {stats['kernels_comm']}, matched: {stats['kernels_with_parent']}, "
            f"Megatron dim|op parents: {stats['kernels_megatron_labeled']}, "
            f"kernels via Process Group Description: {stats.get('kernels_from_kernel_pg_desc', 0)}, "
            f"kernels via External id→cpu_op hint: {stats.get('kernels_from_external_id_hint', 0)}"
        )
        print(
            f"    param_comms refine: overlap={stats.get('kernels_refined_param_overlap', 0)}, "
            f"nearest-gap≤{stats.get('param_comms_nearest_megatron_us', 0)}µs="
            f"{stats.get('kernels_refined_param_nearest_gap', 0)}"
        )
        if by_src:
            print(f"    CUDA µs by attribution source: {dict(sorted(by_src.items(), key=lambda x: -x[1]))}")

        for dim, us in file_global.items():
            global_total[dim] += us
        for rank_str, rd in file_per_rank.items():
            for dim, us in rd.items():
                per_rank_total[rank_str][dim] += us
        for src, us in (stats.get("attribution_cuda_us_by_source") or {}).items():
            global_attribution_by_source[src] += float(us)

        if rank_hint is not None and str(rank_hint) not in per_rank_total and file_global:
            for dim, us in file_global.items():
                per_rank_total[str(rank_hint)][dim] += us

    print_rollup(
        dict(global_total),
        dict(per_rank_total),
        output_dir=args.output_dir,
        attribution_by_source=dict(global_attribution_by_source) if global_attribution_by_source else None,
    )


if __name__ == "__main__":
    main()
