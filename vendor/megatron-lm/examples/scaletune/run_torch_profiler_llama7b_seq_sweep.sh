#!/usr/bin/env bash
set -uo pipefail

# Loop-run torch profiler with LLaMA 7B settings and different sequence lengths.
#
# Usage:
#   bash examples/scaletune/run_torch_profiler_llama7b_seq_sweep.sh
#
# Optional env overrides:
#   TRAIN_ITERS=3 PROFILE_STEP_START=1 PROFILE_STEP_END=2 PROFILE_RANKS="0"
#   TP=2 PP=2 MICRO_BS=2 GLOBAL_BS=32 CONTINUE_ON_ERROR=1
#   LONG_SEQ_POLICY=auto|allow|skip  # default: auto (allow on H100, skip on others)
#   ALLOW_UNSAFE_LONG_SEQ=1          # legacy override, equivalent to LONG_SEQ_POLICY=allow
#   SEQ_LIST_OVERRIDE="1024"         # run only selected seq lengths (space-separated)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_SCRIPT="${ROOT}/examples/scaletune/run_with_salloc.sh"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: run script not found: ${RUN_SCRIPT}" >&2
    exit 2
fi
# LLaMA 7B-style configuration.
export MODEL_FAMILY="${MODEL_FAMILY:-llama}"
export NUM_LAYERS="${NUM_LAYERS:-8}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
export NUM_HEADS="${NUM_HEADS:-32}"
export MAX_POS_EMB="${MAX_POS_EMB:-131072}"
export TP="${TP:-2}"
export PP="${PP:-1}"
export CP="${CP:-1}"
export MICRO_BS="${MICRO_BS:-1}"
export GLOBAL_BS="${GLOBAL_BS:-8}"
export TRAIN_ITERS="${TRAIN_ITERS:-3}"
export USE_BF16="${USE_BF16:-1}"
export RECOMPUTE_ACTIVATIONS="${RECOMPUTE_ACTIVATIONS:-0}"
export USE_DISTRIBUTED_OPTIMIZER="${USE_DISTRIBUTED_OPTIMIZER:-0}"
# Avoid allocator internal assert with distributed optimizer + expandable segments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
# Stability workaround for TransformerEngine multi-tensor grad norm kernel.
export MEGATRON_DISABLE_MULTI_TENSOR_GRAD_NORM="${MEGATRON_DISABLE_MULTI_TENSOR_GRAD_NORM:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
ALLOW_UNSAFE_LONG_SEQ="${ALLOW_UNSAFE_LONG_SEQ:-0}"
LONG_SEQ_POLICY="${LONG_SEQ_POLICY:-auto}"
BASE_TP="${TP}"
BASE_PP="${PP}"
BASE_CP="${CP}"
BASE_MICRO_BS="${MICRO_BS}"
BASE_GLOBAL_BS="${GLOBAL_BS}"

detect_gpu_name() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true
    fi
}

GPU_NAME="$(detect_gpu_name)"
LONG_SEQ_EFFECTIVE_ALLOW="0"

if [[ "${ALLOW_UNSAFE_LONG_SEQ}" == "1" ]]; then
    LONG_SEQ_EFFECTIVE_ALLOW="1"
elif [[ "${LONG_SEQ_POLICY}" == "allow" ]]; then
    LONG_SEQ_EFFECTIVE_ALLOW="1"
elif [[ "${LONG_SEQ_POLICY}" == "skip" ]]; then
    LONG_SEQ_EFFECTIVE_ALLOW="0"
else
    # auto: allow on H100, keep conservative on other GPUs.
    if [[ "${GPU_NAME}" == *"H100"* ]]; then
        LONG_SEQ_EFFECTIVE_ALLOW="1"
    fi
fi

# Torch profiler config.
export PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
export PROFILE_STEP_END="${PROFILE_STEP_END:-2}"
export PROFILE_RANKS="${PROFILE_RANKS:-0}"

# Sequence length sweep: 1024, 128k, 512k, 1M
SEQ_LIST=(1024 131072 524288 1048576)
if [[ -n "${SEQ_LIST_OVERRIDE:-}" ]]; then
    read -r -a SEQ_LIST <<< "${SEQ_LIST_OVERRIDE}"
fi

echo "============================================================"
echo "LLaMA 7B torch profiler sequence sweep"
echo "Run script: ${RUN_SCRIPT}"
echo "TP=${TP} PP=${PP} CP=${CP} MICRO_BS=${MICRO_BS} GLOBAL_BS=${GLOBAL_BS}"
echo "TRAIN_ITERS=${TRAIN_ITERS} PROFILE_STEP_START=${PROFILE_STEP_START} PROFILE_STEP_END=${PROFILE_STEP_END} PROFILE_RANKS=${PROFILE_RANKS}"
echo "GPU_NAME=${GPU_NAME:-unknown} LONG_SEQ_POLICY=${LONG_SEQ_POLICY} LONG_SEQ_EFFECTIVE_ALLOW=${LONG_SEQ_EFFECTIVE_ALLOW}"
echo "SEQ_LIST=${SEQ_LIST[*]}"
echo "============================================================"

for seq in "${SEQ_LIST[@]}"; do
    tag="seq${seq}"
    echo ""
    echo ">>> Running ${tag}"

    export SEQ_LEN="${seq}"
    export PROFILE_TENSORBOARD_DIR="${ROOT}/torch_profiler_traces/${tag}"
    mkdir -p "${PROFILE_TENSORBOARD_DIR}"
    export TP="${BASE_TP}"
    export PP="${BASE_PP}"
    export CP="${BASE_CP}"
    export MICRO_BS="${BASE_MICRO_BS}"
    export GLOBAL_BS="${BASE_GLOBAL_BS}"

    # Guardrail for dense-attention LLaMA at extreme context lengths.
    if [[ "${seq}" -ge 131072 && "${LONG_SEQ_EFFECTIVE_ALLOW}" != "1" ]]; then
        echo ">>> SKIP ${tag}"
        echo "    reason: dense-attention LLaMA at seq>=128k is very likely to OOM on non-H100 GPUs"
        echo "    set LONG_SEQ_POLICY=allow (or ALLOW_UNSAFE_LONG_SEQ=1) to force this run"
        continue
    fi

    # Dense attention memory is O(seq^2). For very long contexts, enable CP and reduce batch aggressively.
    if [[ "${seq}" -ge 131072 ]]; then
        if [[ "${LONG_SEQ_EFFECTIVE_ALLOW}" == "1" && "${CP_LONG_CONTEXT:-8}" -le 1 ]]; then
            echo ">>> SKIP ${tag}"
            echo "    reason: seq>=128k for dense LLaMA requires context parallelism; set CP_LONG_CONTEXT>1"
            continue
        fi
        export MICRO_BS=1
        export GLOBAL_BS=1
        export TP="${TP_LONG_CONTEXT:-1}"
        export PP="${PP_LONG_CONTEXT:-1}"
        export CP="${CP_LONG_CONTEXT:-8}"
        echo "    long-context overrides: TP=${TP} PP=${PP} CP=${CP} MICRO_BS=${MICRO_BS} GLOBAL_BS=${GLOBAL_BS}"
    fi

    if bash "${RUN_SCRIPT}" --model llama --profiling-mode torch; then
        echo ">>> Finished ${tag}"
        echo "    traces: ${PROFILE_TENSORBOARD_DIR}"
    else
        rc=$?
        echo ">>> FAILED ${tag} (exit=${rc})"
        echo "    likely cause: one rank exited first (commonly OOM), peers then report NCCL/TCP reset."
        echo "    traces: ${PROFILE_TENSORBOARD_DIR}"
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
            echo "    continue to next seq (CONTINUE_ON_ERROR=1)"
            continue
        fi
        exit "${rc}"
    fi
done

echo ""
echo "All runs complete."
echo "TensorBoard root: ${ROOT}/torch_profiler_traces"
