#!/usr/bin/env bash
# HLT (Hierarchical Layer Trace) post-processing for run_four_models_profile.sh.
#
# Summarizes compute / memory / communication buckets and top-K GPU operators
# across qwen3-32b, qwen3-30b-a3b, nemotron3-nano, dlm-qwen3-32b.
#
# No re-run needed — reads existing *.pt.trace.json / rank-*.json.gz only.
#
# Usage:
#   bash scaletune/analyze_four_models_hlt.sh
#   bash scaletune/analyze_four_models_hlt.sh \\
#       --batch-dir /home/qi/AIPerf-LLM/scaletune_runs/batch_four_models_20260607_083109
#   python3 scaletune/tools/analyze_four_models_hlt.py \\
#       --artifact-dir /path/to/torch_qwen3_32b_... \\
#       --artifact-dir /path/to/torch_megadlms_... \\
#       --output-dir /tmp/hlt_out
#   TOP_K=10 bash scaletune/analyze_four_models_hlt.sh --rank 0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIPERF_ROOT="${AIPERF_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ANALYZER="${AIPERF_ROOT}/scaletune/tools/analyze_four_models_hlt.py"

if [[ ! -f "${ANALYZER}" ]]; then
    echo "ERROR: missing ${ANALYZER}" >&2
    exit 2
fi

export PROFILE_OUTPUT_ROOT="${PROFILE_OUTPUT_ROOT:-${AIPERF_ROOT}/scaletune_runs}"

ARGS=()
if [[ -n "${BATCH_DIR:-}" ]]; then
    ARGS+=(--batch-dir "${BATCH_DIR}")
fi
if [[ -n "${RUNS_ROOT:-}" ]]; then
    ARGS+=(--runs-root "${RUNS_ROOT}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${RANK:-}" ]]; then
    ARGS+=(--rank "${RANK}")
fi
ARGS+=(--top-k "${TOP_K:-5}")

exec python3 "${ANALYZER}" "${ARGS[@]}" "$@"
