# AIPerf-LLM `scaletune` layout

## What was extracted

| Path | Contents |
|------|-----------|
| `tools/analyze_nvtx_comm_time.py` | Copied from `Megatron-LM/scaletune/` (Megatron’s Python scaletune package is otherwise minimal in this tree). |
| `vendor/megatron-lm/examples/scaletune/*.sh` | Snapshot copies of Megatron-LM example helpers (reference; **run from repo** for correct `ROOT`). |
| `vendor/megadlms/examples/scaletune/*.sh` | Snapshot copies of MegaDLMs example helpers. |

## Single entry: `run_with_salloc.sh`

The **canonical** `run_with_salloc.sh` implementations remain in:

- `Megatron-LM/examples/scaletune/run_with_salloc.sh`
- `MegaDLMs/examples/scaletune/run_with_salloc.sh`
- `Megatron-Bridge/examples/scaletune/run_with_salloc.sh`

This directory’s **`run_with_salloc.sh`** is a thin **dispatcher**: you pass **`--framework megatron-lm`** or **`--framework megadlms`** (short **`-f`**), and it `cd`s into the right repo and `exec`s that repo’s script with your remaining arguments. That way:

- Training, `PYTHONPATH`, and Megatron-Core paths stay correct.
- You only memorize **one** command path under AIPerf-LLM.
- You avoid maintaining two near-duplicate 800+ line scripts in a third place.

### Examples

```bash
export AIPERF_ROOT=/path/to/AIPerf-LLM

# MegaDLMs (DiffLM / default llama in that script)
bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" --framework megadlms -m llama -p torch
# equivalent: -f megadlms ... --profiling-mode torch

# Megatron-LM (Qwen / Llama / Mamba as in that script)
bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" -f megatron-lm -m qwen2-7b -p scaletune
```

# Megatron-Bridge (uv .venv; login node one-time setup — see Megatron-Bridge/examples/scaletune/run_with_salloc.sh)
bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" -f megatron-bridge -m nano -s tiny -p none -b 1
```

Default profiling artifact root for MegaDLMs runs:

`PROFILE_OUTPUT_ROOT` → **`${AIPERF_ROOT}/scaletune_runs`** (override as needed).

### Environment shortcuts

```bash
export AIPERF_FRAMEWORK=megadlms
bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" -m mamba8b
```

(`AIPERF_STACK` is still read as a deprecated alias for `AIPERF_FRAMEWORK`. `--stack` is accepted as an alias for `--framework`.)

## Fully merging the two bash scripts

The two upstream scripts still differ in model matrix, `ROOT`, `MEGATRON_ROOT`, wrapper `TRAIN_SCRIPT`, and some Slurm / export flags. A literal single-file merge would be ~1600+ lines and easy to drift. The dispatcher keeps **one user-facing entry** while preserving **one source of truth per repo**. If you later want a shared `lib/*.sh` included by both repos, that can be done incrementally.
