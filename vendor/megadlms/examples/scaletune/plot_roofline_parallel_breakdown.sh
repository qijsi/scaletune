#!/usr/bin/env bash
# Plot ScaleTune roofline JSON (parallel breakdown) using Megatron-LM's bundled tool.
#
# Requires: Megatron-LM checkout with scaletune/, matplotlib (see scaletune/requirements.txt).
#
# Usage (from anywhere):
#   bash examples/scaletune/plot_roofline_parallel_breakdown.sh
#   MEGATRON_ROOT=~/WORK/Megatron-LM bash examples/scaletune/plot_roofline_parallel_breakdown.sh
#
# Default (no args): same idea as:
#   cd ~/WORK/Megatron-LM
#   python -m scaletune.roofline.plot_parallel_breakdown_v2 \\
#     -i roofline_out_llama7b/roofline_cluster_iter1.json \\
#     -o roofline_out_llama7b/phase_breakdown_by_training_stage.png
#
# Custom defaults via env:
#   INPUT=roofline_out_llama7b/roofline_cluster_iter0.json \\
#   OUTPUT=roofline_out_llama7b/my_plot.png \\
#   bash examples/scaletune/plot_roofline_parallel_breakdown.sh
#
# Pass-through (any python -m args):
#   bash examples/scaletune/plot_roofline_parallel_breakdown.sh \\
#     -i run1.json -i run2.json --aggregate-ranks -o compare.png
#
set -euo pipefail

MEGATRON_ROOT="${MEGATRON_ROOT:-${HOME}/WORK/Megatron-LM}"
if [[ ! -d "${MEGATRON_ROOT}" ]]; then
  echo "ERROR: MEGATRON_ROOT is not a directory: ${MEGATRON_ROOT}" >&2
  echo "  export MEGATRON_ROOT=/path/to/Megatron-LM" >&2
  exit 2
fi
if [[ ! -f "${MEGATRON_ROOT}/scaletune/roofline/plot_parallel_breakdown_v2.py" ]]; then
  echo "ERROR: plot_parallel_breakdown_v2.py not found under ${MEGATRON_ROOT}/scaletune/roofline/" >&2
  exit 2
fi

cd "${MEGATRON_ROOT}"
export PYTHONPATH="${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -gt 0 ]]; then
  exec python -m scaletune.roofline.plot_parallel_breakdown_v2 "$@"
fi

INPUT="${INPUT:-roofline_out_llama7b/roofline_cluster_iter1.json}"
OUTPUT="${OUTPUT:-roofline_out_llama7b/phase_breakdown_by_training_stage.png}"

exec python -m scaletune.roofline.plot_parallel_breakdown_v2 \
  -i "${INPUT}" \
  -o "${OUTPUT}"
