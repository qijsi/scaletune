#!/usr/bin/env bash
# Unified profiling sweep for four AIPerf scaletune models:
#   1. Megatron-LM  Qwen3-32B dense
#   2. Megatron-LM  Qwen3-30B-A3B MoE
#   3. Megatron-Bridge  Nemotron-3-Nano (tiny yaml)
#   4. MegaDLMs  DLM Qwen3-32B
#
# Shared benchmark profile (defaults):
#   PROFILING_MODE=torch, TRAIN_ITERS=4, profile window [1, 3)
#   TP=4, PP=2 (8 GPUs), USE_MOCK_DATA=0, real gpt_dataset, SEQ_LEN=4096
#
# Usage:
#   bash scaletune/run_four_models_profile.sh
#   DRY_RUN=1 bash scaletune/run_four_models_profile.sh
#   MODELS="qwen3-32b,qwen3-30b-a3b" bash scaletune/run_four_models_profile.sh
#   CONTINUE_ON_ERROR=0 bash scaletune/run_four_models_profile.sh
#
# Environment overrides (applied to every model unless a per-model block sets otherwise):
#   TRAIN_ITERS, PROFILE_STEP_START, PROFILE_STEP_END
#   TP, PP, USE_MOCK_DATA, SEQ_LEN / SEQ_LENGTH, MICRO_BS, GLOBAL_BS
#   DATA_ROOT, DATA_PREFIX, PROFILE_OUTPUT_ROOT, SLURM_ARGS, BATCH_DIR
#
# Output layout (default):
#   ${PROFILE_OUTPUT_ROOT}/batch_four_models_<timestamp>/
#     qwen3-32b/          profiling traces for model 1
#     qwen3-30b-a3b/      profiling traces for model 2
#     nemotron3-nano/     profiling traces for model 3
#     dlm-qwen3-32b/      profiling traces for model 4
#     logs/               per-model launcher logs
#     batch_manifest.json model -> artifact dir (for analyze_four_models_hlt.py)
#
# HLT analysis on the batch directory:
#   python3 scaletune/tools/analyze_four_models_hlt.py --batch-dir <batch_dir>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIPERF_ROOT="${AIPERF_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAUNCHER="${AIPERF_ROOT}/scaletune/run_with_srun.sh"

if [[ ! -f "${LAUNCHER}" ]]; then
    echo "ERROR: missing launcher ${LAUNCHER}" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Unified profiling profile
# ---------------------------------------------------------------------------
export PROFILING_MODE="${PROFILING_MODE:-torch}"
export TRAIN_ITERS="${TRAIN_ITERS:-4}"
export PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
export PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
export USE_MOCK_DATA="${USE_MOCK_DATA:-0}"
export TP="${TP:-4}"
export PP="${PP:-2}"
export SCALETUNE_DEFAULT_GPUS_PER_NODE="${SCALETUNE_DEFAULT_GPUS_PER_NODE:-8}"
export MICRO_BS="${MICRO_BS:-1}"
export GLOBAL_BS="${GLOBAL_BS:-8}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${MICRO_BS}}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${GLOBAL_BS}}"
export SEQ_LEN="${SEQ_LEN:-4096}"
export SEQ_LENGTH="${SEQ_LENGTH:-${SEQ_LEN}}"
export DATA_ROOT="${DATA_ROOT:-${HOME}/gpt_dataset}"
export DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT}/redpajama_text_document}"
export PROFILE_OUTPUT_ROOT="${PROFILE_OUTPUT_ROOT:-${AIPERF_ROOT}/scaletune_runs}"
export TRAIN_ITERS_OVERRIDE="${TRAIN_ITERS_OVERRIDE:-${TRAIN_ITERS}}"
export SCALETUNE_LOCK_PARALLEL="${SCALETUNE_LOCK_PARALLEL:-1}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
MODELS="${MODELS:-qwen3-32b,qwen3-30b-a3b,nemotron3-nano,dlm-qwen3-32b}"

BATCH_TS="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="${BATCH_DIR:-${PROFILE_OUTPUT_ROOT}/batch_four_models_${BATCH_TS}}"
BATCH_LOG_DIR="${BATCH_DIR}/logs"
mkdir -p "${BATCH_LOG_DIR}"
export BATCH_DIR BATCH_TS

if [[ "${TRAIN_ITERS}" -lt "${PROFILE_STEP_END}" ]]; then
    echo "WARNING: TRAIN_ITERS (${TRAIN_ITERS}) < PROFILE_STEP_END (${PROFILE_STEP_END}); profiler may miss steps." >&2
fi

echo "================================================================"
echo "AIPerf four-model unified profiling batch"
echo "  PROFILING_MODE=${PROFILING_MODE}"
echo "  TRAIN_ITERS=${TRAIN_ITERS}  profile window=[${PROFILE_STEP_START}, ${PROFILE_STEP_END})"
echo "  TP=${TP}  PP=${PP}  GPUs=${SCALETUNE_DEFAULT_GPUS_PER_NODE}"
echo "  USE_MOCK_DATA=${USE_MOCK_DATA}  SEQ_LEN=${SEQ_LEN}"
echo "  MICRO_BS=${MICRO_BS}  GLOBAL_BS=${GLOBAL_BS}"
echo "  DATA_PREFIX=${DATA_PREFIX}"
echo "  PROFILE_OUTPUT_ROOT=${PROFILE_OUTPUT_ROOT}"
echo "  BATCH_DIR=${BATCH_DIR}"
echo "  BATCH_LOG_DIR=${BATCH_LOG_DIR}"
echo "  MODELS=${MODELS}"
echo "================================================================"
echo ""

_write_batch_manifest() {
    local manifest="${BATCH_DIR}/batch_manifest.json"
    python3 - <<'PY' "${manifest}" "${BATCH_DIR}" "${BATCH_TS}" "${PROFILING_MODE}" "${MODELS}"
import json, os, sys
manifest, batch_dir, batch_ts, profiling_mode, models = sys.argv[1:6]
model_list = [m.strip() for m in models.split(",") if m.strip()]
entries = {}
for key in model_list:
    sub = os.path.join(batch_dir, key)
    if os.path.isdir(sub):
        entries[key] = os.path.abspath(sub)
payload = {
    "batch_dir": os.path.abspath(batch_dir),
    "timestamp": batch_ts,
    "profiling_mode": profiling_mode,
    "models": entries,
}
with open(manifest, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"[batch] wrote {manifest}")
PY
}

_run_model() {
    local key="$1"
    local log_file="${BATCH_LOG_DIR}/${key}.log"
    local model_out="${BATCH_DIR}/${key}"
    local -a cmd=()
    mkdir -p "${model_out}"

    case "${key}" in
        qwen3-32b)
            export TP PP SCALETUNE_DEFAULT_GPUS_PER_NODE SCALETUNE_LOCK_PARALLEL
            cmd=(
                bash "${LAUNCHER}"
                -f megatron-lm
                -m qwen3-32b
                -p "${PROFILING_MODE}"
                --profile-step-start "${PROFILE_STEP_START}"
                --profile-step-end "${PROFILE_STEP_END}"
            )
            ;;
        qwen3-30b-a3b)
            # MoE on 8 GPUs: TP=4 x PP=2, EP=1 (EP*TP*PP = 8)
            export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
            export EP="${EP:-${EXPERT_MODEL_PARALLEL_SIZE}}"
            export TP PP SCALETUNE_DEFAULT_GPUS_PER_NODE SCALETUNE_LOCK_PARALLEL
            cmd=(
                bash "${LAUNCHER}"
                -f megatron-lm
                -m qwen3-30b-a3b
                -p "${PROFILING_MODE}"
                --profile-step-start "${PROFILE_STEP_START}"
                --profile-step-end "${PROFILE_STEP_END}"
            )
            ;;
        nemotron3-nano)
            export SCALETUNE_SRUN_NTASKS="${SCALETUNE_SRUN_NTASKS:-1}"
            export SCALETUNE_SRUN_NTASKS_PER_NODE="${SCALETUNE_SRUN_NTASKS_PER_NODE:-1}"
            export MODEL_SIZE="${MODEL_SIZE:-tiny}"
            export GPUS_PER_NODE="${GPUS_PER_NODE:-${SCALETUNE_DEFAULT_GPUS_PER_NODE}}"
            export BRIDGE_NANO_TP="${BRIDGE_NANO_TP:-${TP}}"
            export BRIDGE_NANO_PP="${BRIDGE_NANO_PP:-${PP}}"
            export BRIDGE_NANO_EP="${BRIDGE_NANO_EP:-1}"
            export BRIDGE_NANO_SEQ_LENGTH="${BRIDGE_NANO_SEQ_LENGTH:-${SEQ_LEN}}"
            export BRIDGE_NANO_MICRO_BS="${BRIDGE_NANO_MICRO_BS:-${MICRO_BS}}"
            export BRIDGE_NANO_GLOBAL_BS="${BRIDGE_NANO_GLOBAL_BS:-${GLOBAL_BS}}"
            cmd=(
                bash "${LAUNCHER}"
                -f megatron-bridge
                -m nemotron3-nano30b
                -s "${MODEL_SIZE}"
                -p "${PROFILING_MODE}"
            )
            ;;
        dlm-qwen3-32b)
            export MODEL_PARALLEL_SIZE="${MODEL_PARALLEL_SIZE:-${TP}}"
            # DLM difflm-noshift does not support PP>1 (labels/mask only on first/last stage).
            export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
            export GPUS_PER_NODE="${GPUS_PER_NODE:-$((MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE))}"
            export SCALETUNE_SRUN_NTASKS="${SCALETUNE_SRUN_NTASKS:-1}"
            export SCALETUNE_SRUN_NTASKS_PER_NODE="${SCALETUNE_SRUN_NTASKS_PER_NODE:-1}"
            export SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-1}"
            cmd=(
                bash "${LAUNCHER}"
                -f megadlms
                -m qwen3-32b
                -p "${PROFILING_MODE}"
            )
            ;;
        *)
            echo "ERROR: unknown model key '${key}'" >&2
            return 2
            ;;
    esac

    echo "----------------------------------------------------------------"
    echo "[${key}] starting at $(date -Is)"
    echo "  artifact_dir: ${model_out}"
    echo "  log: ${log_file}"
    printf '  cmd:'
    printf ' %q' "${cmd[@]}"
    echo ""
    echo "----------------------------------------------------------------"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[${key}] DRY_RUN=1 — skipped"
        return 0
    fi

    set +e
    (
        export AIPERF_ROOT
        # Keep all profiling artifacts under this model's batch subdirectory.
        export PROFILE_OUTPUT_ROOT="${model_out}"
        export SCALETUNE_ROOFLINE_OUTPUT_DIR="${model_out}"
        export PROFILE_TENSORBOARD_DIR="${model_out}"
        export PROFILE_DIR="${model_out}"
        export PROFILE_RUN_STAMP="${BATCH_TS}"
        "${cmd[@]}"
    ) 2>&1 | tee "${log_file}"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ "${rc}" -eq 0 ]]; then
        echo "[${key}] OK"
    else
        echo "[${key}] FAILED (exit ${rc})" >&2
    fi
    return "${rc}"
}

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"

failed=()
passed=()
for model_key in "${MODEL_LIST[@]}"; do
    model_key="$(echo "${model_key}" | tr -d '[:space:]')"
    [[ -z "${model_key}" ]] && continue
    if _run_model "${model_key}"; then
        passed+=("${model_key}")
    else
        failed+=("${model_key}")
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
            echo "Stopping batch (CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR})." >&2
            break
        fi
    fi
    echo ""
done

_write_batch_manifest

echo "================================================================"
echo "Batch summary"
echo "  passed (${#passed[@]}): ${passed[*]:-none}"
echo "  failed (${#failed[@]}): ${failed[*]:-none}"
echo "  batch_dir: ${BATCH_DIR}"
echo "  logs: ${BATCH_LOG_DIR}"
echo "  HLT: python3 ${AIPERF_ROOT}/scaletune/tools/analyze_four_models_hlt.py --batch-dir ${BATCH_DIR}"
echo "================================================================"

if [[ "${#failed[@]}" -gt 0 ]]; then
    exit 1
fi
