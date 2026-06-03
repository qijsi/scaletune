#!/bin/bash
# Run Megatron pretrain with ScaleTune roofline on Slurm.
#
# Supports two launch modes:
#
#   Mode 1: Inside an existing salloc / sbatch allocation
#     - Detects SLURM_JOB_ID and forwards to run_with_salloc.sh.
#
#   Mode 2: Direct launch (NOT inside an allocation)
#     - One srun allocates resources and starts run_with_salloc.sh on every rank
#       (--ntasks = nodes * gpus/node). No nested srun, --overlap, or re-exec of this script.
#
# Usage examples:
#
#   # --- Mode 1: Inside salloc/sbatch allocation ---
#   salloc -N 2 -n 16 --gpus-per-node=8 --time=02:00:00 --partition=a01
#   bash examples/scaletune/run_with_srun.sh -m qwen2-7b --profiling-mode torch
#
#   # --- Mode 2: Single srun (allocates GPUs/CPUs and starts all ranks in one step) ---
#   # Defaults: -p a01 -N 1 --gres=gpu:2 -> ntasks=2, cpus-per-task from SCALETUNE_SRUN_CPUS_PER_TASK (default 16).
#   bash examples/scaletune/run_with_srun.sh -m qwen2-7b --profiling-mode torch
#
#   # Custom Slurm resources (uniform GPUs per node; set SCALETUNE_DEFAULT_GPUS_PER_NODE if no --gres in args):
#   bash examples/scaletune/run_with_srun.sh -m qwen2-7b --profiling-mode torch \
#       --srun-args="-p a01 -N 1 --gres=gpu:2 --time=02:00:00"
#
#   # Or via environment variable:
#   SLURM_ARGS="-p a01 -N 2 -n 16 --gres=gpu:8 --time=01:00:00" \
#       bash examples/scaletune/run_with_srun.sh -m llama
#
# Stale SLURM_JOB_ID (old salloc shell) is ignored: squeue must show the job as
# running (R), completing (CG), or configuring (CF); otherwise a new srun allocation
# is submitted (Mode 2).

set -euo pipefail

# True when SLURM_JOB_ID is non-empty AND still appears in squeue in a state where
# `srun --jobid=...` can attach a new step (RUNNING / COMPLETING / CONFIGURING).
# Stale shells often keep SLURM_JOB_ID after the job ends -> srun fails with
# "Invalid job id specified".
slurm_job_id_accepts_new_step() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || return 1
    local state
    state=$(squeue -j "${SLURM_JOB_ID}" -h -o "%t" 2>/dev/null | head -n1 | tr -d '[:space:]')
    [[ -n "${state}" ]] || return 1
    case "${state}" in
        R|CG|CF) return 0 ;;
        *) return 1 ;;
    esac
}

clear_stale_slurm_allocation_env() {
    unset SLURM_JOB_ID
    unset SLURM_JOB_NODELIST SLURM_NODELIST SLURM_JOB_NUM_NODES SLURM_NNODES 2>/dev/null || true
}

# ============================================================================
# Parse arguments (only our own; everything else is forwarded)
# ============================================================================
PROFILING_MODE=""
SLURM_ARGS_FROM_CLI=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_FAMILY="${2:-}"
            if [[ -z "${MODEL_FAMILY}" ]]; then
                echo "ERROR: --model requires a value (llama | mamba8b | falcon-mamba-7b | qwen2-7b | qwen3-32b | qwen3-30b-a3b | nemotron3-nano30b)" >&2
                exit 2
            fi
            FORWARD_ARGS+=(--model "${MODEL_FAMILY}")
            shift 2
            ;;
        --model=*)
            MODEL_FAMILY="${1#*=}"
            FORWARD_ARGS+=(--model "${MODEL_FAMILY}")
            shift
            ;;
        --profiling-mode|--profiling-mod)
            PROFILING_MODE="${2:-}"
            if [[ -n "${PROFILING_MODE}" ]]; then
                FORWARD_ARGS+=(--profiling-mode "${PROFILING_MODE}")
            fi
            shift 2
            ;;
        --profiling-mode=*|--profiling-mod=*)
            PROFILING_MODE="${1#*=}"
            FORWARD_ARGS+=(--profiling-mode "${PROFILING_MODE}")
            shift
            ;;
        --srun-args)
            SLURM_ARGS_FROM_CLI="${2:-}"
            if [[ -z "${SLURM_ARGS_FROM_CLI}" ]]; then
                echo "ERROR: --srun-args requires a value" >&2
                exit 2
            fi
            shift 2
            ;;
        --srun-args=*)
            SLURM_ARGS_FROM_CLI="${1#*=}"
            shift
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ============================================================================
# Detect launch mode
# ============================================================================
if [[ -n "${SLURM_JOB_ID:-}" ]] && slurm_job_id_accepts_new_step; then
    # -----------------------------------------------------------------------
    # Mode 1: Inside a Slurm allocation (salloc / sbatch)
    #   Forward to run_with_salloc.sh which handles the actual launch.
    # -----------------------------------------------------------------------
    echo "[run_with_srun] Detected active Slurm allocation (job=${SLURM_JOB_ID})"
    echo "[run_with_srun] Forwarding to run_with_salloc.sh"
    echo ""

    exec bash "${SCRIPT_DIR}/run_with_salloc.sh" "${FORWARD_ARGS[@]}"
else
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "[run_with_srun] WARNING: SLURM_JOB_ID=${SLURM_JOB_ID} is not a running Slurm job"
        echo "[run_with_srun] (stale env from an old salloc/sbatch shell, or job ended)."
        echo "[run_with_srun] Clearing it and submitting a fresh allocation via srun."
        echo ""
        clear_stale_slurm_allocation_env
    fi
    # -----------------------------------------------------------------------
    # Mode 2: NOT inside a Slurm allocation — single srun for the whole run.
    # run_with_salloc.sh detects SLURM_NTASKS == TOTAL_GPUS and skips nested srun.
    # -----------------------------------------------------------------------

    # Resolve Slurm resource arguments.
    # Priority: --srun-args CLI > SLURM_ARGS env > defaults
    SLURM_ARGS="${SLURM_ARGS_FROM_CLI:-${SLURM_ARGS:-${SLURM_DEFAULT_ARGS:-}}}"
    if [[ -z "${SLURM_ARGS}" ]]; then
        echo "[run_with_srun] No Slurm allocation and no --srun-args / SLURM_ARGS set"
        echo "[run_with_srun] Using defaults: -p a01 -N 1 --gres=gpu:2 --time=01:00:00"
        echo "[run_with_srun] Override: --srun-args='...' or SLURM_ARGS='...'"
        echo ""
        SLURM_ARGS="-p a01 -N 1 --gres=gpu:2 --time=01:00:00"
    else
        echo "[run_with_srun] No Slurm allocation; submitting single srun with: ${SLURM_ARGS}"
    fi
    echo ""

    # Derive task geometry (uniform GPUs per node). Trailing srun flags override duplicates in SLURM_ARGS.
    SRUN_NODES=1
    SRUN_GPUS_PER_NODE="${SCALETUNE_DEFAULT_GPUS_PER_NODE:-2}"
    if [[ "${SLURM_ARGS}" =~ -N[[:space:]]+([0-9]+) ]]; then
        SRUN_NODES="${BASH_REMATCH[1]}"
    fi
    if [[ "${SLURM_ARGS}" =~ --nodes=([0-9]+) ]]; then
        SRUN_NODES="${BASH_REMATCH[1]}"
    fi
    if [[ "${SLURM_ARGS}" =~ --gres=gpu:([0-9]+) ]]; then
        SRUN_GPUS_PER_NODE="${BASH_REMATCH[1]}"
    fi
    if [[ "${SLURM_ARGS}" =~ --gpus-per-node=([0-9]+) ]]; then
        SRUN_GPUS_PER_NODE="${BASH_REMATCH[1]}"
    fi

    SRUN_NTASKS="${SCALETUNE_SRUN_NTASKS:-$((SRUN_NODES * SRUN_GPUS_PER_NODE))}"
    SRUN_NTASKS_PER_NODE="${SCALETUNE_SRUN_NTASKS_PER_NODE:-${SRUN_GPUS_PER_NODE}}"
    SCALETUNE_SRUN_CPUS_PER_TASK="${SCALETUNE_SRUN_CPUS_PER_TASK:-16}"

    echo "[run_with_srun] srun task layout: nodes=${SRUN_NODES} gpus/node=${SRUN_GPUS_PER_NODE} ntasks=${SRUN_NTASKS} ntasks-per-node=${SRUN_NTASKS_PER_NODE} cpus-per-task=${SCALETUNE_SRUN_CPUS_PER_TASK}"
    echo "[run_with_srun] Override layout: SCALETUNE_SRUN_NTASKS, SCALETUNE_SRUN_NTASKS_PER_NODE, SCALETUNE_DEFAULT_GPUS_PER_NODE"
    echo ""

    exec srun ${SLURM_ARGS} \
        --ntasks="${SRUN_NTASKS}" \
        --ntasks-per-node="${SRUN_NTASKS_PER_NODE}" \
        --cpus-per-task="${SCALETUNE_SRUN_CPUS_PER_TASK}" \
        bash "${SCRIPT_DIR}/run_with_salloc.sh" "${FORWARD_ARGS[@]}"
fi
