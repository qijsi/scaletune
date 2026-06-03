#!/bin/bash
# ScaleTune roofline / profiler launcher: works with salloc OR bare srun.
#
# This is a thin wrapper around run_with_srun.sh. It sets default Slurm
# resource flags when you are not already inside an allocation (Mode 2).
# When SLURM_JOB_ID is set (after salloc/sbatch), arguments are forwarded
# directly to run_with_salloc.sh — same as calling run_with_salloc.sh yourself.
#
# Usage (Qwen2-7B + PyTorch profiler, matches common local workflow):
#
#   # After: salloc -p a01 -N 1 --gres=gpu:2 --cpus-per-task=32 ...
#   bash examples/scaletune/run_scaletune_slurm.sh -m qwen2-7b --profiling-mode torch
#
#   # One-shot: single srun (all ranks in one step; no nested srun)
#   bash examples/scaletune/run_scaletune_slurm.sh -m qwen2-7b --profiling-mode torch
#
# Override Slurm resources / CPU per rank (SCALETUNE_SRUN_CPUS_PER_TASK, default 16):
#   SLURM_ARGS="-p a01 -N 1 --gres=gpu:8 --time=02:00:00" SCALETUNE_SRUN_CPUS_PER_TASK=32 \
#     bash examples/scaletune/run_scaletune_slurm.sh -m qwen2-7b --profiling-mode torch
#
# Or pass any other flags through to run_with_salloc.sh:
#   bash examples/scaletune/run_scaletune_slurm.sh -m llama --profiling-mode torch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default resources when launching without a prior allocation (see run_with_srun.sh).
export SLURM_ARGS="${SLURM_ARGS:--p a01 -N 1 --gres=gpu:2 --time=01:00:00}"

exec bash "${SCRIPT_DIR}/run_with_srun.sh" "$@"
