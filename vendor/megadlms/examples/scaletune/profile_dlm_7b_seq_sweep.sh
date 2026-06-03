#!/bin/bash
# Sweep DLM 7B profiling over multiple sequence lengths on an existing Slurm allocation.
#
# It calls:
#   bash examples/scaletune/run_dlm_7b_with_salloc.sh --profiling-mode torch   (default)
#   PROFILING_MODE=scaletune bash examples/scaletune/profile_dlm_7b_seq_sweep.sh
#
# Torch / ScaleTune artifacts go under MegaDLMs repo root: <slug>/torch_profiler, scaletune_roofline, scaletune_comm.
#
# Default seq lengths: 1024, 128k, 512k, 1m
#
# Usage:
#   bash examples/scaletune/profile_dlm_7b_seq_sweep.sh
#
# Optional env overrides:
#   SEQ_SWEEP_VALUES="1024 128k 512k 1m"
#   TRAIN_ITERS_OVERRIDE=5
#   PROFILE_STEP_START=1
#   PROFILE_STEP_END=3
#   PROFILE_RANKS=0
#   PROFILING_MODE=torch|scaletune
#   RECOMPUTE_ACTIVATIONS=0|1
#   SWEEP_OUTPUT_ROOT=/path/to/output
#   SEQ_TIMEOUT_SECONDS=3600

set -euo pipefail

CURRENT_CHILD_PID=""
cleanup_on_interrupt() {
    echo ""
    echo "Interrupt received. Cleaning up sweep subprocesses..."
    if [[ -n "${CURRENT_CHILD_PID}" ]]; then
        kill -TERM "${CURRENT_CHILD_PID}" 2>/dev/null || true
    fi
    # Kill any direct child processes spawned by this sweep script.
    pkill -TERM -P $$ 2>/dev/null || true
    sleep 0.2
    pkill -KILL -P $$ 2>/dev/null || true
    exit 130
}
trap cleanup_on_interrupt INT TERM

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATASETS_DIR="${DATASETS_DIR:-$HOME/WORK/datasets/gpt_dataset}"
CKPT_DIR="${CKPT_DIR:-/tmp/${USER}/megadlm_profile_tmp}"
LAUNCH_SCRIPT="${PROJECT_DIR}/examples/scaletune/run_dlm_7b_with_salloc.sh"
SEQ_SWEEP_VALUES="${SEQ_SWEEP_VALUES:-1024 128k 512k 1m}"
TRAIN_ITERS_OVERRIDE="${TRAIN_ITERS_OVERRIDE:-5}"
PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
TOKENIZER="${TOKENIZER:-${DATASETS_DIR}/tokenizer.model}"
SEQ_TIMEOUT_SECONDS="${SEQ_TIMEOUT_SECONDS:-3600}"
PROFILING_MODE="${PROFILING_MODE:-torch}"
RECOMPUTE_ACTIVATIONS="${RECOMPUTE_ACTIVATIONS:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${PROJECT_DIR}/dlm_7b_seq_sweep_${TIMESTAMP}}"

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
    echo "ERROR: launcher not found: ${LAUNCH_SCRIPT}" >&2
    exit 2
fi

if [[ -z "${PROJECT_DIR:-}" ]]; then
    echo "ERROR: PROJECT_DIR is required." >&2
    exit 2
fi

parse_seq_to_int() {
    local raw="$1"
    local lower
    lower="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${lower}" =~ ^[0-9]+$ ]]; then
        echo "${lower}"
        return 0
    fi
    if [[ "${lower}" =~ ^([0-9]+)k$ ]]; then
        echo "$((BASH_REMATCH[1] * 1024))"
        return 0
    fi
    if [[ "${lower}" =~ ^([0-9]+)m$ ]]; then
        echo "$((BASH_REMATCH[1] * 1024 * 1024))"
        return 0
    fi
    return 1
}

mkdir -p "${SWEEP_OUTPUT_ROOT}"
SUMMARY_FILE="${SWEEP_OUTPUT_ROOT}/summary.txt"
{
    echo "DLM 7B sequence profiling sweep"
    echo "time=${TIMESTAMP}"
    echo "seq_values=${SEQ_SWEEP_VALUES}"
    echo "train_iters_override=${TRAIN_ITERS_OVERRIDE}"
    echo "profile_steps=[${PROFILE_STEP_START}, ${PROFILE_STEP_END})"
    echo "profile_ranks=${PROFILE_RANKS}"
    echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
    echo "micro_batch_size=${MICRO_BATCH_SIZE}"
    echo "tokenizer_type=${TOKENIZER_TYPE}"
    echo "tokenizer=${TOKENIZER}"
    echo "seq_timeout_seconds=${SEQ_TIMEOUT_SECONDS}"
    echo "profiling_mode=${PROFILING_MODE}"
    echo "recompute_activations=${RECOMPUTE_ACTIVATIONS}"
    echo ""
} > "${SUMMARY_FILE}"

echo "Sweep output root: ${SWEEP_OUTPUT_ROOT}"
echo "Summary file: ${SUMMARY_FILE}"
echo ""

for seq_raw in ${SEQ_SWEEP_VALUES}; do
    if ! seq_int="$(parse_seq_to_int "${seq_raw}")"; then
        echo "Skip invalid seq value: ${seq_raw}" | tee -a "${SUMMARY_FILE}"
        continue
    fi

    run_tag="seq_${seq_raw}"
    run_name="dlm_training_7b_${run_tag}"
    run_root="${SWEEP_OUTPUT_ROOT}/${run_tag}"
    run_log="${run_root}/run.log"
    mkdir -p "${run_root}"

    echo "====================================================================" | tee -a "${SUMMARY_FILE}"
    echo "Start profiling for ${seq_raw} (SEQ_LENGTH=${seq_int})" | tee -a "${SUMMARY_FILE}"
    echo "run_name=${run_name}" | tee -a "${SUMMARY_FILE}"
    echo "run_log=${run_log}" | tee -a "${SUMMARY_FILE}"
    echo "(artifacts: ${PROJECT_DIR}/<slug>/torch_profiler, scaletune_roofline, scaletune_comm)" | tee -a "${SUMMARY_FILE}"
    echo "===================================================================="

    set +e
    if command -v timeout >/dev/null 2>&1; then
        timeout "${SEQ_TIMEOUT_SECONDS}" env \
            RUN_NAME="${run_name}" \
            SEQ_LENGTH="${seq_int}" \
            TRAIN_ITERS_OVERRIDE="${TRAIN_ITERS_OVERRIDE}" \
            GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
            MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
            DATALOADER_NUM_WORKERS=0 \
            NO_CREATE_ATTENTION_MASK_IN_DATALOADER=1 \
            RECOMPUTE_ACTIVATIONS="${RECOMPUTE_ACTIVATIONS}" \
            DATASETS_DIR="${DATASETS_DIR}" \
            CKPT_DIR="${CKPT_DIR}" \
            DISABLE_CHECKPOINT_SAVE=1 \
            PROFILE_STEP_START="${PROFILE_STEP_START}" \
            PROFILE_STEP_END="${PROFILE_STEP_END}" \
            PROFILE_RANKS="${PROFILE_RANKS}" \
            TOKENIZER_TYPE="${TOKENIZER_TYPE}" \
            TOKENIZER="${TOKENIZER}" \
            bash "${LAUNCH_SCRIPT}" --profiling-mode "${PROFILING_MODE}" > "${run_log}" 2>&1 &
        CURRENT_CHILD_PID=$!
        wait "${CURRENT_CHILD_PID}"
        status=$?
        CURRENT_CHILD_PID=""
    else
        RUN_NAME="${run_name}" \
        SEQ_LENGTH="${seq_int}" \
        TRAIN_ITERS_OVERRIDE="${TRAIN_ITERS_OVERRIDE}" \
        GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
        DATALOADER_NUM_WORKERS=0 \
        NO_CREATE_ATTENTION_MASK_IN_DATALOADER=1 \
        RECOMPUTE_ACTIVATIONS="${RECOMPUTE_ACTIVATIONS}" \
        DATASETS_DIR="${DATASETS_DIR}" \
        CKPT_DIR="${CKPT_DIR}" \
        DISABLE_CHECKPOINT_SAVE=1 \
        PROFILE_STEP_START="${PROFILE_STEP_START}" \
        PROFILE_STEP_END="${PROFILE_STEP_END}" \
        PROFILE_RANKS="${PROFILE_RANKS}" \
        TOKENIZER_TYPE="${TOKENIZER_TYPE}" \
        TOKENIZER="${TOKENIZER}" \
        bash "${LAUNCH_SCRIPT}" --profiling-mode "${PROFILING_MODE}" > "${run_log}" 2>&1 &
        CURRENT_CHILD_PID=$!
        wait "${CURRENT_CHILD_PID}"
        status=$?
        CURRENT_CHILD_PID=""
    fi
    set -e

    if [[ "${status}" -eq 0 ]]; then
        echo "status=SUCCESS seq=${seq_raw} log=${run_log}" | tee -a "${SUMMARY_FILE}"
    elif [[ "${status}" -eq 124 ]]; then
        echo "status=TIMEOUT seq=${seq_raw} timeout=${SEQ_TIMEOUT_SECONDS}s log=${run_log}" | tee -a "${SUMMARY_FILE}"
    else
        echo "status=FAILED seq=${seq_raw} exit_code=${status} log=${run_log}" | tee -a "${SUMMARY_FILE}"
    fi
done

echo "" | tee -a "${SUMMARY_FILE}"
echo "Sweep finished. See summary: ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
