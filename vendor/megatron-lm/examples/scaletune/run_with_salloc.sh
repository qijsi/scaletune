#!/bin/bash
# Run Megatron pretrain (default: Nemotron-H / Mamba 8B) with ScaleTune roofline on
# Slurm-allocated nodes (from salloc), using srun or mpirun.
#
# Usage:
#   1. First, allocate nodes with salloc:
#      salloc -N 2 -n 2 --gpus-per-node=8 --time=02:00:00 --partition=a01
#
#   2. Then run this script:
#      bash examples/scaletune/run_with_salloc.sh
#      bash examples/scaletune/run_with_salloc.sh -m qwen2-7b
#      bash examples/scaletune/run_with_salloc.sh --model llama
#      bash examples/scaletune/run_with_salloc.sh --model llama3-8b   # alias for Llama 3 8B
#      MODEL_FAMILY=qwen2-7b bash examples/scaletune/run_with_salloc.sh
#      MODEL_FAMILY=falcon-mamba-7b bash examples/scaletune/run_with_salloc.sh
#      bash examples/scaletune/run_with_salloc.sh -m qwen2-7b --profiling-mode scaletune \\
#          --profile-step-start 0 --profile-step-end 3
#      bash examples/scaletune/run_with_salloc.sh -m qwen3-32b
#      bash examples/scaletune/run_with_salloc.sh -m qwen3-30b-a3b
#      bash examples/scaletune/run_with_salloc.sh -m nemotron3-nano30b
#
# Profiling (--profiling-mode):
#   torch     — PyTorch Profiler only (Kineto trace); disables ScaleTune comm / parallel-dim NVTX.
#   scaletune — Adds --profile --use-pytorch-profiler --tensorboard-dir (rank*.pt.trace.json next to roofline JSON).
#   both      — same as scaletune (backward-compatible alias).
# Put known flags (--model, --profiling-mode, …) before other Megatron args so this script parses them.
#
# Default trace output: roofline JSON and Chrome traces share one directory (see PROFILE_TENSORBOARD_DIR).
# Dir name: scaletune_<model>_<framework>_tp*_pp*-virstage*_cp*_dp*_ep*_etp*_edp*_rec*_seq*_mbs*_<stamp> (CP/DP/EP/ETP/EDP default 1).
# Override: PROFILE_TENSORBOARD_DIR, SCALETUNE_ROOFLINE_OUTPUT_DIR, or PROFILE_RUN_STAMP.
#
# The script will automatically detect your Slurm allocation and run on those nodes.

set -euo pipefail

# Parse command line arguments
# Keep unrecognized flags and pass them through to pretrain_*.py.
PROFILING_MODE=""
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_FAMILY="${2:-}"
            if [[ -z "${MODEL_FAMILY}" ]]; then
                echo "ERROR: --model requires a value (llama | llama3-8b | mamba8b | falcon-mamba-7b | qwen2-7b | qwen3-32b | qwen3-30b-a3b | nemotron3-nano30b)" >&2
                exit 2
            fi
            shift 2
            ;;
        --model=*)
            MODEL_FAMILY="${1#*=}"
            shift
            ;;
        --profiling-mode|--profiling-mod)
            PROFILING_MODE="${2:-}"
            shift 2
            ;;
        --profiling-mode=*|--profiling-mod=*)
            PROFILING_MODE="${1#*=}"
            shift
            ;;
        --profile-step-start)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --profile-step-start requires an integer (Megatron training iter)" >&2
                exit 2
            fi
            PROFILE_STEP_START="${2}"
            shift 2
            ;;
        --profile-step-start=*)
            PROFILE_STEP_START="${1#*=}"
            shift
            ;;
        --profile-step-end)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --profile-step-end requires an integer (Megatron training iter, exclusive)" >&2
                exit 2
            fi
            PROFILE_STEP_END="${2}"
            shift 2
            ;;
        --profile-step-end=*)
            PROFILE_STEP_END="${1#*=}"
            shift
            ;;
        *)
            # Preserve unknown args for the training script and continue parsing.
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Normalize --profiling-mode (trim, lowercase) so Scaletune / SCALEtune / extra spaces work
PROFILING_MODE="$(echo "${PROFILING_MODE:-}" | xargs)"
PROFILING_MODE="${PROFILING_MODE,,}"

# ============================================================================
# Configuration
# ============================================================================
CONDA_ENV="${CONDA_ENV:-aiperf_llm}"
CONDA_PATH="/home/fit/zhaijdzq/WORK/miniconda3"

export SCALETUNE_ROOFLINE_DEBUG_COMM=1
export MEGATRON_P2P_WAIT_TRACE_METADATA=1

# Model family: mamba8b (Nemotron-H 8B hybrid) | falcon-mamba-7b | llama / llama3-8b (Llama 3 8B) | qwen2-7b | qwen3-32b | qwen3-30b-a3b | nemotron3-nano30b
MODEL_FAMILY="${MODEL_FAMILY:-qwen2-7b}"
case "${MODEL_FAMILY}" in
    llama3-8b|llama3_8b) MODEL_FAMILY="llama" ;;
    falcon_mamba_7b|falconmamba7b) MODEL_FAMILY="falcon-mamba-7b" ;;
    qwen3_32b|Qwen3-32B) MODEL_FAMILY="qwen3-32b" ;;
    qwen3_30b_a3b|Qwen3-30B-A3B) MODEL_FAMILY="qwen3-30b-a3b" ;;
    nemotron3_nano30b|Nemotron3-Nano-30B|Nemotron-3-Nano-30B) MODEL_FAMILY="nemotron3-nano30b" ;;
esac

# Model configuration (defaults depend on MODEL_FAMILY unless you set overrides)
# Only set defaults if not already specified
if [[ -z "${NUM_LAYERS:-}" ]]; then
    if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
        NUM_LAYERS=52
    elif [[ "${MODEL_FAMILY}" == "falcon-mamba-7b" ]]; then
        NUM_LAYERS=64
    elif [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
        NUM_LAYERS=28
    elif [[ "${MODEL_FAMILY}" == "qwen3-32b" ]]; then
        NUM_LAYERS=64
    elif [[ "${MODEL_FAMILY}" == "qwen3-30b-a3b" ]]; then
        NUM_LAYERS=48
    elif [[ "${MODEL_FAMILY}" == "nemotron3-nano30b" ]]; then
        NUM_LAYERS=56
    else
        NUM_LAYERS=32  # Llama 3 8B (and legacy llama branch)
    fi
fi

# Set other defaults (override with env: HIDDEN_SIZE, NUM_HEADS)
if [[ -z "${HIDDEN_SIZE:-}" ]]; then
    case "${MODEL_FAMILY}" in
        qwen3-32b|nemotron3-nano30b) HIDDEN_SIZE=5120 ;;
        qwen3-30b-a3b) HIDDEN_SIZE=2048 ;;
        *) HIDDEN_SIZE=4096 ;;
    esac
fi
if [[ -z "${NUM_HEADS:-}" ]]; then
    case "${MODEL_FAMILY}" in
        qwen3-32b) NUM_HEADS=64 ;;
        qwen3-30b-a3b) NUM_HEADS=32 ;;
        nemotron3-nano30b) NUM_HEADS=40 ;;
        *) NUM_HEADS=32 ;;
    esac
fi
# Llama 3 8B: HF max_position_embeddings 8192 (RoPE base 500000, GQA 8, FFN 14336).
# Default SEQ_LEN matches other Megatron ScaleTune models (1024) for apples-to-apples benchmarks;
# set SEQ_LEN=8192 (or export) to stress native context length.
# Qwen3-32B / Qwen3-30B-A3B: HF max_position_embeddings 40960 (reduce SEQ_LEN if OOM).
if [[ "${MODEL_FAMILY}" == "llama" ]]; then
    MAX_POS_EMB="${MAX_POS_EMB:-8192}"
    SEQ_LEN="${SEQ_LEN:-1024}"
elif [[ "${MODEL_FAMILY}" == "qwen3-32b" || "${MODEL_FAMILY}" == "qwen3-30b-a3b" ]]; then
    MAX_POS_EMB="${MAX_POS_EMB:-40960}"
    SEQ_LEN="${SEQ_LEN:-4096}"
elif [[ "${MODEL_FAMILY}" == "nemotron3-nano30b" ]]; then
    MAX_POS_EMB="${MAX_POS_EMB:-8192}"
    SEQ_LEN="${SEQ_LEN:-4096}"
else
    MAX_POS_EMB="${MAX_POS_EMB:-4096}"
    SEQ_LEN="${SEQ_LEN:-1024}"
fi

# Llama 3 8B overrides (Meta config; align with examples/inference/llama_mistral/run_text_generation_llama3.sh)
LLAMA3_FFN_HIDDEN_SIZE="${LLAMA3_FFN_HIDDEN_SIZE:-14336}"
LLAMA3_NUM_QUERY_GROUPS="${LLAMA3_NUM_QUERY_GROUPS:-8}"
LLAMA3_ROPE_BASE="${LLAMA3_ROPE_BASE:-500000}"
# Nemotron-3 Nano ~30B dense (approximate; override from NVIDIA model card if needed)
NEMOTRON3_FFN_HIDDEN_SIZE="${NEMOTRON3_FFN_HIDDEN_SIZE:-27648}"
NEMOTRON3_NUM_QUERY_GROUPS="${NEMOTRON3_NUM_QUERY_GROUPS:-8}"
NEMOTRON3_ROPE_BASE="${NEMOTRON3_ROPE_BASE:-1000000}"
# Qwen3-32B dense (Hugging Face Qwen/Qwen3-32B config.json)
QWEN3_32B_FFN_HIDDEN_SIZE="${QWEN3_32B_FFN_HIDDEN_SIZE:-25600}"
QWEN3_32B_NUM_QUERY_GROUPS="${QWEN3_32B_NUM_QUERY_GROUPS:-8}"
# Qwen3-30B-A3B MoE (see examples/post_training/modelopt/conf/qwen/Qwen3-30B-A3B.sh)
QWEN3_30B_A3B_FFN_HIDDEN_SIZE="${QWEN3_30B_A3B_FFN_HIDDEN_SIZE:-6144}"
QWEN3_30B_A3B_NUM_EXPERTS="${QWEN3_30B_A3B_NUM_EXPERTS:-128}"
QWEN3_30B_A3B_MOE_FFN_HIDDEN_SIZE="${QWEN3_30B_A3B_MOE_FFN_HIDDEN_SIZE:-768}"
QWEN3_30B_A3B_MOE_TOPK="${QWEN3_30B_A3B_MOE_TOPK:-8}"
QWEN3_30B_A3B_VOCAB_DIVISIBLE="${QWEN3_30B_A3B_VOCAB_DIVISIBLE:-1187}"
MICRO_BS="${MICRO_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-8}"
# Short profiling runs: default 4 iters (must be > last --profile-step-end).
TRAIN_ITERS="${TRAIN_ITERS:-4}"
LR="${LR:-3e-4}"
TP="${TP:-2}"
PP="${PP:-2}"
# Parallel dims encoded in trace/output dir names (defaults 1 unless set).
CP="${CP:-2}"
DP="${DP:-1}"
# Expert-parallel size for MoE (also used in output dir slug as ep*).
if [[ "${MODEL_FAMILY}" == "qwen3-30b-a3b" ]]; then
    EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
    EP="${EP:-${EXPERT_MODEL_PARALLEL_SIZE}}"
else
    EP="${EP:-1}"
fi
ETP="${ETP:-1}"
EDP="${EDP:-1}"
# Pipeline schedule controls:
# - non-interleaved 1F1B (default): leave both virtual-stage vars empty
# - interleaved 1F1B: set one of:
#     NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE
#     NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE="${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE:-}"
NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK="${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK:-}"
# Directory naming override for virtual pipeline stage count.
# If unset, it is inferred from virtual pipeline flags and falls back to 1.
VIR_STAGE="${VIR_STAGE:-}"
# Set to 1 to add --no-overlap-p2p-communication.
NO_OVERLAP_P2P_COMM="${NO_OVERLAP_P2P_COMM:-0}"
# Set to 1 to add --overlap-p2p-communication-warmup-flush.
OVERLAP_P2P_COMM_WARMUP_FLUSH="${OVERLAP_P2P_COMM_WARMUP_FLUSH:-0}"

# Nemotron-H 8B (Mamba hybrid) — see examples/nemotron_h/train_nemotron_h_8b.sh
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-14336}"
HYBRID_ATTN_RATIO="${HYBRID_ATTN_RATIO:-0.08}"
HYBRID_MLP_RATIO="${HYBRID_MLP_RATIO:-0.46}"
MAMBA_STATE_DIM="${MAMBA_STATE_DIM:-128}"
# d_conv / expand / dt_rank are not CLI flags in this Megatron tree; MambaMixer uses d_conv=4, expand=2.
MAMBA_NUM_GROUPS="${MAMBA_NUM_GROUPS:-8}"
MAMBA_HEAD_DIM="${MAMBA_HEAD_DIM:-64}"
# import_module() expects: module_path attribute_name (must exist in mamba_layer_specs.py)
MAMBA_SPEC_MODULE="${MAMBA_SPEC_MODULE:-megatron.core.models.mamba.mamba_layer_specs}"
MAMBA_SPEC_NAME="${MAMBA_SPEC_NAME:-mamba_stack_spec}"

# Data configuration
DATA_ROOT="${DATA_ROOT:-${HOME}/WORK/datasets/gpt_dataset}"
DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT}/redpajama_text_document}"
case "${MODEL_FAMILY}" in
    qwen3-32b)
        TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
        TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-32B}"
        ;;
    qwen3-30b-a3b)
        TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
        TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-30B-A3B}"
        ;;
    nemotron3-nano30b)
        TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
        TOKENIZER_MODEL="${TOKENIZER_MODEL:-nvidia/Nemotron-3-Nano-30B-v1}"
        ;;
    *)
        TOKENIZER_MODEL="${TOKENIZER_MODEL:-${DATA_ROOT}/tokenizer.model}"
        TOKENIZER_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
        ;;
esac
USE_MOCK_DATA="${USE_MOCK_DATA:-0}"

# Memory / activation checkpointing (needed before profiling output dir names use RECOMPUTE_ACTIVATIONS)
USE_BF16="${USE_BF16:-1}"
RECOMPUTE_ACTIVATIONS="${RECOMPUTE_ACTIVATIONS:-0}"
EMPTY_UNUSED_MEMORY_LEVEL="${EMPTY_UNUSED_MEMORY_LEVEL:-1}"

# Output directories
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Short model tag for directory names (from MODEL_FAMILY).
case "${MODEL_FAMILY}" in
    mamba8b) PROFILE_MODEL_TAG="mamba" ;;
    falcon-mamba-7b) PROFILE_MODEL_TAG="falconmamba7b" ;;
    qwen2-7b) PROFILE_MODEL_TAG="qwen" ;;
    qwen3-32b) PROFILE_MODEL_TAG="qwen3_32b" ;;
    qwen3-30b-a3b) PROFILE_MODEL_TAG="qwen3_30b_a3b" ;;
    nemotron3-nano30b) PROFILE_MODEL_TAG="nemotron3_nano30b" ;;
    llama) PROFILE_MODEL_TAG="llama3" ;;
    *) PROFILE_MODEL_TAG="${MODEL_FAMILY}" ;;
esac
# Fixed timestamp for one run (override to correlate dirs: PROFILE_RUN_STAMP=20260101_120000 ...)
PROFILE_RUN_STAMP="${PROFILE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
# Stack tag in output dir names (megatron vs megadlms, etc.); AIPerf unified launcher may set this.
PROFILE_FRAMEWORK_TAG="${PROFILE_FRAMEWORK_TAG:-megatron}"
PROFILE_FRAMEWORK_TAG="$(echo "${PROFILE_FRAMEWORK_TAG}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_' | tr -s '_')"
# Strip leading/trailing underscores so joining does not produce "__" segments.
PROFILE_FRAMEWORK_TAG="${PROFILE_FRAMEWORK_TAG#_}"
PROFILE_FRAMEWORK_TAG="${PROFILE_FRAMEWORK_TAG%_}"
[[ -n "${PROFILE_FRAMEWORK_TAG}" ]] || PROFILE_FRAMEWORK_TAG="megatron"
# Infer virtual stage count for output directory naming.
if [[ -n "${VIR_STAGE}" ]]; then
    VIRTUAL_STAGE_COUNT="${VIR_STAGE}"
elif [[ -n "${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK}" ]]; then
    VIRTUAL_STAGE_COUNT="${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK}"
elif [[ -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" ]] \
    && [[ "${NUM_LAYERS}" =~ ^[0-9]+$ ]] \
    && [[ "${PP}" =~ ^[0-9]+$ ]] \
    && [[ "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" =~ ^[0-9]+$ ]] \
    && [[ "${PP}" -gt 0 ]] \
    && [[ "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" -gt 0 ]]; then
    layers_per_pp_rank=$((NUM_LAYERS / PP))
    if [[ "${layers_per_pp_rank}" -gt 0 ]]; then
        VIRTUAL_STAGE_COUNT=$((layers_per_pp_rank / NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE))
    fi
fi
if [[ -z "${VIRTUAL_STAGE_COUNT:-}" || "${VIRTUAL_STAGE_COUNT}" -le 0 ]]; then
    VIRTUAL_STAGE_COUNT=1
fi

# Parallel + recompute + seq + micro-batch + time (for experiment comparison).
PROFILE_RUN_SUFFIX="${PROFILE_FRAMEWORK_TAG}_tp${TP}_pp${PP}-virstage${VIRTUAL_STAGE_COUNT}_cp${CP}_dp${DP}_ep${EP}_etp${ETP}_edp${EDP}_rec${RECOMPUTE_ACTIVATIONS}_seq${SEQ_LEN}_mbs${MICRO_BS}_${PROFILE_RUN_STAMP}"
case "${PROFILING_MODE}" in
    torch) PROFILE_MODE_TAG="torch" ;;
    both) PROFILE_MODE_TAG="both" ;;
    scaletune|*) PROFILE_MODE_TAG="scaletune" ;;
esac
PROFILE_RUN_SUFFIX="$(echo "${PROFILE_MODE_TAG}_${PROFILE_MODEL_TAG}_${PROFILE_RUN_SUFFIX}" | tr -s '_')"

# ScaleTune roofline output directory (Chrome traces / roofline JSON share this path by default).
# Default encodes model tag + parallel config + recompute + seq + micro_bs + timestamp.
# Override with SCALETUNE_ROOFLINE_OUTPUT_DIR to keep a fixed path.
if [[ -z "${SCALETUNE_ROOFLINE_OUTPUT_DIR:-}" ]]; then
    SCALETUNE_ROOFLINE_OUTPUT_DIR="${ROOT}/${PROFILE_RUN_SUFFIX}"
fi

# PyTorch Profiler traces directory (Kineto / Chrome trace)
# Passed via --tensorboard-dir when using --profiling-mode torch | scaletune | both.
# Default: same as SCALETUNE_ROOFLINE_OUTPUT_DIR so rank*.pt.trace.json sits next to roofline_*.json
# (ScaleTuneProfiler._tensorboard_trace_dir only searches that directory).
# For --profiling-mode scaletune|both, PROFILE_TENSORBOARD_DIR is set again when appending --profile.
if [[ -z "${PROFILE_TENSORBOARD_DIR:-}" ]]; then
    PROFILE_TENSORBOARD_DIR="${SCALETUNE_ROOFLINE_OUTPUT_DIR}"
fi

# Roofline JSON for these training iterations. With Kineto warmup=1, start profile one iter earlier
# than the first roofline iter (see --profile-step-start/end below).
SCALETUNE_ROOFLINE_ITERATIONS="${SCALETUNE_ROOFLINE_ITERATIONS:-1,2}"

# ============================================================================
# Check Slurm allocation
# ============================================================================
# Stale SLURM_JOB_ID (job completed / purged) still appears in many shells; do not
# use it for srun --jobid= or trust SLURM_JOB_NODELIST from the same env.
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

if [[ -n "${SLURM_JOB_ID:-}" ]] && ! slurm_job_id_accepts_new_step; then
    echo "WARNING: SLURM_JOB_ID=${SLURM_JOB_ID} is not active in squeue (state ended, pending, or invalid)."
    echo "         Unsetting stale Slurm job variables; will try to detect a running job next."
    unset SLURM_JOB_ID
    unset SLURM_JOB_NODELIST SLURM_NODELIST SLURM_JOB_NUM_NODES SLURM_NNODES 2>/dev/null || true
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # Try to get job info from squeue (user's current jobs)
    CURRENT_JOB=$(squeue -u "${USER}" -t R -h -o "%i %N" 2>/dev/null | head -1)
    if [[ -n "${CURRENT_JOB}" ]]; then
        SLURM_JOB_ID=$(echo "${CURRENT_JOB}" | awk '{print $1}')
        SLURM_JOB_NODELIST=$(echo "${CURRENT_JOB}" | awk '{print $2}')
        SLURM_JOB_NUM_NODES=$(echo "${CURRENT_JOB}" | awk '{print $2}' | tr ',' '\n' | wc -l)
        echo "Detected running Slurm job from squeue:"
    else
        echo "ERROR: No Slurm job found!"
        echo ""
        echo "Please allocate nodes first with salloc:"
        echo "  salloc -N 2 -n 2 --gpus-per-node=8 --time=02:00:00 --partition=a01"
        echo ""
        echo "Or use srun to start an interactive session:"
        echo "  srun -N 2 -n 2 --gpus-per-node=8 --time=02:00:00 --partition=a01 --pty bash"
        echo ""
        echo "Then run this script in the same shell session."
        exit 2
    fi
fi

if [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Note: Qwen2 7B supports native long-context 32k-128k with GQA and RoPE base=1M"
fi
echo "============================================================================"
echo "Running on Slurm allocation"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Nodes: ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-?}}"
echo "  Node list: ${SLURM_JOB_NODELIST:-?}"
if [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Note: Qwen2 7B supports native long-context 32k-128k with GQA and RoPE base=1M"
fi
echo "============================================================================"

# ============================================================================
# Load OpenMPI module
# ============================================================================
if ! command -v mpirun &> /dev/null; then
    echo "Loading OpenMPI module..."
    if command -v module &> /dev/null; then
        module load mpi/openmpi-4.1.4-gcc12-cuda12.6 2>/dev/null || \
        module load openmpi 2>/dev/null || {
            echo "ERROR: Could not load OpenMPI module"
            echo "  Try: module load mpi/openmpi-4.1.4-gcc12-cuda12.6"
            exit 2
        }
    else
        echo "ERROR: module command not found"
        echo "  Please load OpenMPI manually: module load mpi/openmpi-4.1.4-gcc12-cuda12.6"
        exit 2
    fi
fi

echo "Using mpirun: $(which mpirun)"
echo "  $(mpirun --version 2>&1 | head -1)"

# ============================================================================
# Create hostfile from Slurm node list
# ============================================================================
HOSTFILE=$(mktemp)
if command -v scontrol &> /dev/null; then
    scontrol show hostname "${SLURM_JOB_NODELIST}" > "${HOSTFILE}"
else
    # Fallback: parse node list manually
    echo "${SLURM_JOB_NODELIST}" | tr ',' '\n' > "${HOSTFILE}"
fi

NUM_NODES=$(wc -l < "${HOSTFILE}")

# GPUs per node -> world size = NUM_NODES * GPUS_PER_NODE (override: export GPUS_PER_NODE=N)
# Uses Slurm job/step environment and scontrol/squeue when GPUS_PER_NODE is unset.
resolve_gpus_per_node_from_slurm() {
    local g="" gres
    if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
        echo "${SLURM_GPUS_PER_NODE}"
        return 0
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol &>/dev/null; then
        g=$(
            scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
                | grep -oE 'gres/gpu=[0-9]+' \
                | head -1 \
                | cut -d= -f2
        )
        if [[ -z "${g}" ]]; then
            g=$(
                scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
                    | tr ',' '\n' \
                    | grep -E 'gres/gpu' \
                    | head -1 \
                    | grep -oE '[0-9]+' \
                    | head -1
            )
        fi
    fi
    if [[ -z "${g}" ]] && [[ -n "${SLURM_JOB_ID:-}" ]] && command -v squeue &>/dev/null; then
        gres=$(squeue -j "${SLURM_JOB_ID}" -h -o "%b" 2>/dev/null | head -1)
        if [[ -n "${gres}" ]] && [[ "${gres}" != "(null)" ]]; then
            g=$(echo "${gres}" | grep -oE '[0-9]+$' | head -1)
        fi
    fi
    if [[ -z "${g}" ]] && [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        g="${SLURM_GPUS_ON_NODE}"
    fi
    if [[ -z "${g}" ]] && command -v nvidia-smi &>/dev/null; then
        g=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    if [[ -n "${g}" ]] && [[ "${g}" =~ ^[0-9]+$ ]] && [[ "${g}" -gt 0 ]]; then
        echo "${g}"
        return 0
    fi
    return 1
}

if [[ -z "${GPUS_PER_NODE:-}" ]]; then
    if g_detected=$(resolve_gpus_per_node_from_slurm); then
        GPUS_PER_NODE="${g_detected}"
    else
        GPUS_PER_NODE=8
        echo "WARNING: Could not detect GPUs per node from Slurm (scontrol/squeue/SLURM_* / nvidia-smi); defaulting GPUS_PER_NODE=${GPUS_PER_NODE}. Set GPUS_PER_NODE to override."
    fi
fi

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

MASTER_ADDR=$(head -n 1 "${HOSTFILE}")
# Avoid fixed 6000 (often busy); pick a free port unless user set MASTER_PORT.
if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()" 2>/dev/null || echo "$((29500 + RANDOM % 1000))")
fi

echo "Hostfile: ${HOSTFILE}"
echo "  Nodes: ${NUM_NODES}"
echo "  GPUs per node: ${GPUS_PER_NODE}"
echo "  Total GPUs: ${TOTAL_GPUS}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"

# ============================================================================
# Resolve Slurm partition (required for srun on clusters with no default partition)
# Priority: SLURM_JOB_PARTITION -> scontrol show job -> squeue -> PARTITION env
# ============================================================================
SLURM_RESOLVED_PARTITION=""
if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
    SLURM_RESOLVED_PARTITION="${SLURM_JOB_PARTITION}"
elif [[ -n "${PARTITION:-}" ]]; then
    SLURM_RESOLVED_PARTITION="${PARTITION}"
elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol &>/dev/null; then
    # Job record contains "Partition=name" (possibly comma-separated)
    SLURM_RESOLVED_PARTITION=$(
        scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
            | grep -oE 'Partition=[^[:space:]]+' \
            | head -1 \
            | cut -d= -f2
    )
elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v squeue &>/dev/null; then
    SLURM_RESOLVED_PARTITION=$(squeue -j "${SLURM_JOB_ID}" -h -o "%P" 2>/dev/null | head -1)
fi
# Use first partition only if comma-separated; strip squeue default marker '*'
SLURM_RESOLVED_PARTITION="${SLURM_RESOLVED_PARTITION%%,*}"
SLURM_RESOLVED_PARTITION="${SLURM_RESOLVED_PARTITION//\*/}"

if [[ -n "${SLURM_RESOLVED_PARTITION}" ]]; then
    echo "Slurm partition (for srun): ${SLURM_RESOLVED_PARTITION}"
else
    echo "WARNING: Could not resolve partition. Set PARTITION=a01 or ensure SLURM_JOB_ID is valid."
fi

# ============================================================================
# Build mpirun command
# ============================================================================
# Get OpenMPI installation prefix
OMPI_PREFIX=$(dirname "$(dirname "$(which mpirun)")")

# ---------------------------------------------------------------------------
# Environment for NCCL / ScaleTune / PyTorch (must NOT use OpenMPI "-x" with srun)
#
# Slurm "srun" uses "-x" / "--exclude" for NODE exclusion, not env export.
# Passing "-x NCCL_DEBUG=ERROR" makes Slurm try to exclude a host named
# "NCCL_DEBUG=ERROR" -> "Invalid node name specified".
# Export vars in the shell so srun inherits them (--export=ALL is default).
# ---------------------------------------------------------------------------
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker*}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# NCCL uses NVML for topology; with Slurm GPU binding / CUDA_VISIBLE_DEVICES remapping,
# PCI bus lookups can fail (nvmlDeviceGetHandleByPciBusId: Not Found). Disable NVML in NCCL.
export NCCL_NVML_DISABLE="${NCCL_NVML_DISABLE:-1}"
# Megatron validate_args: assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export WORLD_SIZE="${TOTAL_GPUS}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Add AIPerf-LLM root to PYTHONPATH for standalone scaletune_profiler
AIPERF_LLM_ROOT="$(cd "${ROOT}" && pwd)"
if [[ -d "${AIPERF_LLM_ROOT}/scaletune_profiler" ]]; then
    PYTHONPATH="${AIPERF_LLM_ROOT}:${PYTHONPATH}"
    export PYTHONPATH
fi

# ScaleTune / comm profiling vs pure Torch Profiler (see header: --profiling-mode torch | scaletune | both)
if [[ "${PROFILING_MODE}" == "torch" ]]; then
    # Kineto trace only — no wrapped dist / NVTX parallel-dim labels, no hierarchical roofline JSON.
    export SCALETUNE_HIERARCHICAL_ROOFLINE="${SCALETUNE_HIERARCHICAL_ROOFLINE:-0}"
    export MEGATRON_ScaleTune_ROOFLINE="${MEGATRON_ScaleTune_ROOFLINE:-0}"
    export MEGATRON_COMM_DIM_PROFILE="${MEGATRON_COMM_DIM_PROFILE:-0}"
    export MEGATRON_ENHANCED_COMM_PROFILE="${MEGATRON_ENHANCED_COMM_PROFILE:-0}"
    export SCALETUNE_ENHANCED_COMM_PROFILE="${SCALETUNE_ENHANCED_COMM_PROFILE:-0}"
elif [[ "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    # PyTorch Profiler + Megatron parallel_dim / NVTX (enhanced_comm_profiler).
    export SCALETUNE_HIERARCHICAL_ROOFLINE="${SCALETUNE_HIERARCHICAL_ROOFLINE:-1}"
    export MEGATRON_ScaleTune_ROOFLINE="${MEGATRON_ScaleTune_ROOFLINE:-1}"
    export MEGATRON_COMM_DIM_PROFILE="${MEGATRON_COMM_DIM_PROFILE:-1}"
    export MEGATRON_ENHANCED_COMM_PROFILE="${MEGATRON_ENHANCED_COMM_PROFILE:-1}"
    export SCALETUNE_ENHANCED_COMM_PROFILE="${SCALETUNE_ENHANCED_COMM_PROFILE:-1}"
    # Optional: torch.cuda.synchronize() after each wrapped collective so Chrome traces
    # nest NCCL under ``tp|AR`` python_function rows (analyze_nvtx_comm_time.py). Slow.
    # export MEGATRON_COMM_PROFILE_TRACE_SYNC="${MEGATRON_COMM_PROFILE_TRACE_SYNC:-1}"
else
    # No --profiling-mode: default ScaleTune roofline + comm dim (previous behavior).
    export SCALETUNE_HIERARCHICAL_ROOFLINE="${SCALETUNE_HIERARCHICAL_ROOFLINE:-1}"
    export MEGATRON_ScaleTune_ROOFLINE="${MEGATRON_ScaleTune_ROOFLINE:-1}"
    export MEGATRON_COMM_DIM_PROFILE="${MEGATRON_COMM_DIM_PROFILE:-1}"
    export MEGATRON_ENHANCED_COMM_PROFILE="${MEGATRON_ENHANCED_COMM_PROFILE:-0}"
    export SCALETUNE_ENHANCED_COMM_PROFILE="${SCALETUNE_ENHANCED_COMM_PROFILE:-0}"
fi

export SCALETUNE_ROOFLINE_OUTPUT_DIR="${SCALETUNE_ROOFLINE_OUTPUT_DIR}"
export SCALETUNE_ROOFLINE_ITERATIONS="${SCALETUNE_ROOFLINE_ITERATIONS}"
# Standalone scaletune_profiler (AIPerf-LLM root): enables the standalone profiler
# in Megatron training.py.  Set to 1 to prefer the standalone profiler over
# the embedded Megatron-LM version.
export SCALETUNE_PROFILING="${SCALETUNE_PROFILING:-1}"
export TORCH_COMPILE_DEBUG="${TORCH_COMPILE_DEBUG:-0}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING="${TORCHINDUCTOR_COORDINATE_DESCENT_TUNING:-0}"
export TORCHINDUCTOR_FREEZING="${TORCHINDUCTOR_FREEZING:-0}"
export CONDA_ENV="${CONDA_ENV}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
if [[ "${MODEL_FAMILY}" == "mamba8b" || "${MODEL_FAMILY}" == "falcon-mamba-7b" ]]; then
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton_cache_${USER:-user}}"
    mkdir -p "${TRITON_CACHE_DIR}"
    export TRITON_CACHE_MANAGER="${TRITON_CACHE_MANAGER:-megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager}"
fi

# Nested srun fails when the parent step only reserved 1 CPU (common with salloc shells).
# If this process is already inside a Slurm step with one task per GPU (SLURM_NTASKS == TOTAL_GPUS),
# run the wrapper directly on each rank (e.g. run_with_srun.sh Mode 2 uses one outer srun only).
SCALETUNE_USE_EMBEDDED_SLURM_STEP=0
if [[ "${SCALETUNE_FORCE_INNER_SRUN:-0}" != "1" ]] \
    && [[ -n "${SLURM_STEP_ID:-}" ]] \
    && [[ "${SLURM_NTASKS:-0}" =~ ^[0-9]+$ ]] \
    && [[ "${TOTAL_GPUS}" =~ ^[0-9]+$ ]] \
    && [[ "${SLURM_NTASKS:-0}" -eq "${TOTAL_GPUS}" ]] \
    && [[ "${TOTAL_GPUS}" -gt 0 ]]; then
    SCALETUNE_USE_EMBEDDED_SLURM_STEP=1
fi

# Use srun instead of mpirun for better Slurm integration
# srun automatically sets up WORLD_SIZE, RANK, etc. for PyTorch
LAUNCH_CMD=""
if [[ "${SCALETUNE_USE_EMBEDDED_SLURM_STEP}" == "1" ]]; then
    echo "Using embedded Slurm step (SLURM_NTASKS=${SLURM_NTASKS} == TOTAL_GPUS=${TOTAL_GPUS}); skipping nested srun."
elif command -v srun &> /dev/null; then
    echo "Using srun for better PyTorch integration..."
    LAUNCH_CMD="srun"
    # Inside an active allocation (salloc / sbatch shell), bind the step to this job
    # and pass partition when the site has no default partition.
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        LAUNCH_CMD+=" --jobid=${SLURM_JOB_ID}"
    else
        echo "WARNING: SLURM_JOB_ID not set; srun may fail outside an allocation."
    fi
    if [[ -n "${SLURM_RESOLVED_PARTITION}" ]]; then
        LAUNCH_CMD+=" --partition=${SLURM_RESOLVED_PARTITION}"
    fi
    LAUNCH_CMD+=" --ntasks=${TOTAL_GPUS}"
    LAUNCH_CMD+=" --ntasks-per-node=${GPUS_PER_NODE}"
    # Use --gres=gpu:N for GPU allocation (required by some Slurm clusters)
    # Use actual allocated GPUs from Slurm if available, otherwise fall back to GPUS_PER_NODE
    if [[ -n "${SRUN_GPU_BIND:-}" ]]; then
        LAUNCH_CMD+=" ${SRUN_GPU_BIND}"
    elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        # Use actual GPU count from Slurm allocation
        LAUNCH_CMD+=" --gres=gpu:${SLURM_GPUS_ON_NODE}"
    elif [[ -n "${SLURM_JOB_GRES_GPUS:-}" ]]; then
        # Alternative: extract from SLURM_JOB_GRES (format: gpu:node:8 or gpu:8)
        LAUNCH_CMD+=" --gres=gpu:${SLURM_JOB_GRES_GPUS}"
    else
        # Fallback to detected GPUS_PER_NODE
        LAUNCH_CMD+=" --gres=gpu:${GPUS_PER_NODE}"
    fi
elif ! command -v srun &> /dev/null; then
    # Fallback to mpirun with proper environment setup (OpenMPI "-x" exports env)
    LAUNCH_CMD="mpirun"
    LAUNCH_CMD+=" --hostfile ${HOSTFILE}"
    LAUNCH_CMD+=" -np ${TOTAL_GPUS}"
    LAUNCH_CMD+=" -npernode ${GPUS_PER_NODE}"
    LAUNCH_CMD+=" --prefix ${OMPI_PREFIX}"
    LAUNCH_CMD+=" -bind-to none"
    LAUNCH_CMD+=" -map-by slot"
    LAUNCH_CMD+=" --mca btl_vader_single_copy_mechanism none"
    LAUNCH_CMD+=" -x WORLD_SIZE=${TOTAL_GPUS}"
    LAUNCH_CMD+=" -x MASTER_ADDR=${MASTER_ADDR}"
    LAUNCH_CMD+=" -x MASTER_PORT=${MASTER_PORT}"
    LAUNCH_CMD+=" -x NCCL_DEBUG=${NCCL_DEBUG}"
    LAUNCH_CMD+=" -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
    LAUNCH_CMD+=" -x NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}"
    LAUNCH_CMD+=" -x NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
    LAUNCH_CMD+=" -x NCCL_NVML_DISABLE=${NCCL_NVML_DISABLE}"
    LAUNCH_CMD+=" -x CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
    LAUNCH_CMD+=" -x PYTHONPATH=${PYTHONPATH}"
    LAUNCH_CMD+=" -x SCALETUNE_HIERARCHICAL_ROOFLINE=${SCALETUNE_HIERARCHICAL_ROOFLINE}"
    LAUNCH_CMD+=" -x MEGATRON_ScaleTune_ROOFLINE=${MEGATRON_ScaleTune_ROOFLINE}"
    LAUNCH_CMD+=" -x MEGATRON_COMM_DIM_PROFILE=${MEGATRON_COMM_DIM_PROFILE}"
    LAUNCH_CMD+=" -x MEGATRON_ENHANCED_COMM_PROFILE=${MEGATRON_ENHANCED_COMM_PROFILE}"
    LAUNCH_CMD+=" -x SCALETUNE_ENHANCED_COMM_PROFILE=${SCALETUNE_ENHANCED_COMM_PROFILE}"
    LAUNCH_CMD+=" -x SCALETUNE_ROOFLINE_OUTPUT_DIR=${SCALETUNE_ROOFLINE_OUTPUT_DIR}"
    LAUNCH_CMD+=" -x SCALETUNE_ROOFLINE_ITERATIONS=${SCALETUNE_ROOFLINE_ITERATIONS}"
    LAUNCH_CMD+=" -x TORCH_COMPILE_DEBUG=${TORCH_COMPILE_DEBUG}"
    LAUNCH_CMD+=" -x TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS}"
    LAUNCH_CMD+=" -x TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=${TORCHINDUCTOR_COORDINATE_DESCENT_TUNING}"
    LAUNCH_CMD+=" -x TORCHINDUCTOR_FREEZING=${TORCHINDUCTOR_FREEZING}"
    LAUNCH_CMD+=" -x PATH=${PATH}"
    LAUNCH_CMD+=" -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    LAUNCH_CMD+=" -x CONDA_ENV=${CONDA_ENV}"
    LAUNCH_CMD+=" -x NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE}"
    if [[ "${MODEL_FAMILY}" == "mamba8b" || "${MODEL_FAMILY}" == "falcon-mamba-7b" ]]; then
        LAUNCH_CMD+=" -x TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
        LAUNCH_CMD+=" -x TRITON_CACHE_MANAGER=${TRITON_CACHE_MANAGER}"
    fi
    LAUNCH_CMD+=" -x HOME=${HOME}"
    LAUNCH_CMD+=" -x USER=${USER:-}"
fi

# ============================================================================
# Create wrapper script for conda activation
# Use shared filesystem path accessible from all nodes
# ============================================================================
WRAPPER_DIR="${ROOT}/.scaletune_tmp"
mkdir -p "${WRAPPER_DIR}"
WRAPPER_SCRIPT="${WRAPPER_DIR}/wrapper_$$_${SLURM_JOB_ID}.sh"

cat > "${WRAPPER_SCRIPT}" << WRAPPER_EOF
#!/bin/bash
set -euo pipefail

# Hardcoded paths
CONDA_PATH="${CONDA_PATH}"
CONDA_ENV="${CONDA_ENV}"
ROOT="${ROOT}"

# Activate conda
if [[ -f "\${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    source "\${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "\${CONDA_ENV}"
else
    echo "ERROR: conda.sh not found at \${CONDA_PATH}/etc/profile.d/conda.sh" >&2
    exit 2
fi

# Slurm srun sets SLURM_* but not RANK / WORLD_SIZE / LOCAL_RANK. OpenMPI sets OMPI_*.
# Megatron reads RANK from env (default 0). Without this, every rank is 0 and all bind TCPStore -> EADDRINUSE.
if [[ -n "\${SLURM_PROCID:-}" ]]; then
    export RANK="\${SLURM_PROCID}"
elif [[ -n "\${OMPI_COMM_WORLD_RANK:-}" ]]; then
    export RANK="\${OMPI_COMM_WORLD_RANK}"
fi
if [[ -n "\${SLURM_NTASKS:-}" ]]; then
    export WORLD_SIZE="\${SLURM_NTASKS}"
elif [[ -n "\${OMPI_COMM_WORLD_SIZE:-}" ]]; then
    export WORLD_SIZE="\${OMPI_COMM_WORLD_SIZE}"
fi
# NCCL needs to see node GPU topology; Slurm's per-task CVD masks other GPUs and breaks NVML in NCCL.
# With run_with_salloc.sh using --gpus-per-node (not --gpus-per-task=1), unset CVD and pick device by SLURM_LOCALID.
# Set KEEP_CUDA_VISIBLE_DEVICES=1 to skip unset (debug / exotic schedulers).
if [[ -n "\${SLURM_PROCID:-}" ]] && [[ "\${KEEP_CUDA_VISIBLE_DEVICES:-0}" != "1" ]]; then
    unset CUDA_VISIBLE_DEVICES
    export NCCL_NVML_DISABLE=1
    export LOCAL_RANK="\${SLURM_LOCALID:-0}"
elif [[ -n "\${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "\${CUDA_VISIBLE_DEVICES}" != *","* ]]; then
    export LOCAL_RANK=0
elif [[ -n "\${SLURM_LOCALID:-}" ]]; then
    export LOCAL_RANK="\${SLURM_LOCALID}"
elif [[ -n "\${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]]; then
    export LOCAL_RANK="\${OMPI_COMM_WORLD_LOCAL_RANK}"
fi

# Run training
cd "\${ROOT}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-}"
if [[ -z "\${TRAIN_SCRIPT}" ]]; then
    if [[ "${MODEL_FAMILY}" == "mamba8b" || "${MODEL_FAMILY}" == "falcon-mamba-7b" ]]; then
        TRAIN_SCRIPT="pretrain_mamba.py"
    elif [[ "${MODEL_FAMILY}" == "qwen3-30b-a3b" ]]; then
        TRAIN_SCRIPT="pretrain_gpt.py"
    else
        TRAIN_SCRIPT="pretrain_llama.py"
    fi
fi
exec python "\${TRAIN_SCRIPT}" "\$@"
WRAPPER_EOF

chmod +x "${WRAPPER_SCRIPT}"

# Cleanup on exit
cleanup() {
    rm -f "${HOSTFILE}" "${WRAPPER_SCRIPT}"
    rm -rf "${WRAPPER_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# ============================================================================
# Build training arguments
# ============================================================================
MEMORY_ARGS=()
if [[ "${USE_BF16}" == "1" ]]; then
    MEMORY_ARGS+=(--bf16)
fi
if [[ "${RECOMPUTE_ACTIVATIONS}" == "1" ]]; then
    MEMORY_ARGS+=(--recompute-activations)
fi
if [[ "${EMPTY_UNUSED_MEMORY_LEVEL}" != "0" ]]; then
    MEMORY_ARGS+=(--empty-unused-memory-level "${EMPTY_UNUSED_MEMORY_LEVEL}")
fi

DISTRIBUTED_ARGS=(
    --distributed-backend nccl
    --tp-comm-bootstrap-backend nccl
    --disable-gloo-process-groups
)

PIPELINE_SCHEDULE_ARGS=()
if [[ -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" && -n "${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK}" ]]; then
    echo "ERROR: set only one of NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE or NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK." >&2
    exit 2
fi
if [[ -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" ]]; then
    PIPELINE_SCHEDULE_ARGS+=(--num-layers-per-virtual-pipeline-stage "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}")
fi
if [[ -n "${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK}" ]]; then
    PIPELINE_SCHEDULE_ARGS+=(--num-virtual-stages-per-pipeline-rank "${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK}")
fi
if [[ "${NO_OVERLAP_P2P_COMM}" == "1" ]]; then
    PIPELINE_SCHEDULE_ARGS+=(--no-overlap-p2p-communication)
fi
if [[ "${OVERLAP_P2P_COMM_WARMUP_FLUSH}" == "1" ]]; then
    PIPELINE_SCHEDULE_ARGS+=(--overlap-p2p-communication-warmup-flush)
fi

TOK_ARGS=()
if [[ "${USE_MOCK_DATA}" == "1" && ( "${MODEL_FAMILY}" == "mamba8b" || "${MODEL_FAMILY}" == "falcon-mamba-7b" ) ]]; then
    # Tokenizer set only in DATA_ARGS (NullTokenizer + mock)
    :
else
    TOK_ARGS=(--tokenizer-type "${TOKENIZER_TYPE}")
    if [[ -n "${TOKENIZER_MODEL}" ]]; then
        TOK_ARGS+=(--tokenizer-model "${TOKENIZER_MODEL}")
    fi
fi

DATA_ARGS=()
if [[ "${USE_MOCK_DATA}" == "1" ]]; then
    if [[ "${MODEL_FAMILY}" == "mamba8b" || "${MODEL_FAMILY}" == "falcon-mamba-7b" ]]; then
        # pretrain_mamba + mock: NullTokenizer avoids GPT2 BPE path without vocab files
        DATA_ARGS+=(--mock-data --tokenizer-type NullTokenizer --vocab-size "${VOCAB_SIZE:-32000}")
    else
        DATA_ARGS+=(--mock-data --vocab-size "${VOCAB_SIZE:-32000}")
    fi
else
    DATA_ARGS=(
        --data-path "${DATA_PREFIX}"
        --split "${SPLIT:-969,30,1}"
        --data-cache-path "${DATA_CACHE_PATH:-${DATA_ROOT}/.megatron_data_cache}"
    )
fi

BASE_TRAIN_ARGS=(
    --tensor-model-parallel-size "${TP}"
    --pipeline-model-parallel-size "${PP}"
    --context-parallel-size "${CP}"
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --num-attention-heads "${NUM_HEADS}"
    --max-position-embeddings "${MAX_POS_EMB}"
    --seq-length "${SEQ_LEN}"
    --micro-batch-size "${MICRO_BS}"
    --global-batch-size "${GLOBAL_BS}"
    --train-iters "${TRAIN_ITERS}"
    --lr "${LR}"
    "${MEMORY_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${PIPELINE_SCHEDULE_ARGS[@]}"
    "${TOK_ARGS[@]}"
    "${DATA_ARGS[@]}"
    --skip-eval
)
# MoE: expert tensor parallel (see --expert-model-parallel-size in Megatron)
EXTRA_PARALLEL_ARGS=()
if [[ "${MODEL_FAMILY}" == "qwen3-30b-a3b" ]]; then
    EXTRA_PARALLEL_ARGS+=(--expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}")
fi

if [[ "${MODEL_FAMILY}" == "mamba8b" || "${MODEL_FAMILY}" == "falcon-mamba-7b" ]]; then
    COMMON_ARGS=(
        "${BASE_TRAIN_ARGS[@]}"
        "${EXTRA_PARALLEL_ARGS[@]}"
        --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
        --position-embedding-type rope
        --rotary-percent 1.0
        --rotary-base 10000
        --mamba-state-dim "${MAMBA_STATE_DIM}"
        --mamba-head-dim "${MAMBA_HEAD_DIM}"
        --mamba-num-groups "${MAMBA_NUM_GROUPS}"
        --hybrid-attention-ratio "${HYBRID_ATTN_RATIO}"
        --hybrid-mlp-ratio "${HYBRID_MLP_RATIO}"
        --normalization RMSNorm
        --swiglu
        --untie-embeddings-and-output-weights
        --transformer-impl local
        --spec "${MAMBA_SPEC_MODULE}" "${MAMBA_SPEC_NAME}"
        --disable-bias-linear
        --use-rotary-position-embeddings
        --no-bias-gelu-fusion
        --no-bias-dropout-fusion
        --no-masked-softmax-fusion
        --no-gradient-accumulation-fusion
    )
elif [[ "${MODEL_FAMILY}" == "qwen3-30b-a3b" ]]; then
    # Qwen3-30B-A3B MoE — align with examples/post_training/modelopt/conf/qwen/Qwen3-30B-A3B.sh (uses pretrain_gpt.py)
    COMMON_ARGS=(
        "${BASE_TRAIN_ARGS[@]}"
        "${EXTRA_PARALLEL_ARGS[@]}"
        --ffn-hidden-size "${QWEN3_30B_A3B_FFN_HIDDEN_SIZE}"
        --num-experts "${QWEN3_30B_A3B_NUM_EXPERTS}"
        --moe-ffn-hidden-size "${QWEN3_30B_A3B_MOE_FFN_HIDDEN_SIZE}"
        --moe-router-topk "${QWEN3_30B_A3B_MOE_TOPK}"
        --moe-router-dtype fp32
        --moe-aux-loss-coeff 1e-3
        --moe-token-dispatcher-type alltoall
        --moe-router-load-balancing-type aux_loss
        --moe-layer-recompute
        --make-vocab-size-divisible-by "${QWEN3_30B_A3B_VOCAB_DIVISIBLE}"
        --group-query-attention
        --num-query-groups 4
        --kv-channels 128
        --qk-layernorm
        --position-embedding-type rope
        --no-rope-fusion
        --rotary-percent 1.0
        --rotary-base 1000000
        --normalization RMSNorm
        --swiglu
        --untie-embeddings-and-output-weights
        --transformer-impl transformer_engine
        --disable-bias-linear
        --use-rotary-position-embeddings
        --no-bias-swiglu-fusion
        --no-masked-softmax-fusion
        --no-bias-gelu-fusion
        --no-bias-dropout-fusion
        --no-gradient-accumulation-fusion
        --sequence-parallel
    )
elif [[ "${MODEL_FAMILY}" == "qwen3-32b" ]]; then
    # Qwen3-32B dense — Hugging Face Qwen/Qwen3-32B (5120 x 64 layers, GQA 8, RoPE 1M)
    COMMON_ARGS=(
        "${BASE_TRAIN_ARGS[@]}"
        "${EXTRA_PARALLEL_ARGS[@]}"
        --ffn-hidden-size "${QWEN3_32B_FFN_HIDDEN_SIZE}"
        --group-query-attention
        --num-query-groups "${QWEN3_32B_NUM_QUERY_GROUPS}"
        --qk-layernorm
        --position-embedding-type rope
        --rotary-percent 1.0
        --rotary-base 1000000
        --normalization RMSNorm
        --swiglu
        --untie-embeddings-and-output-weights
        --transformer-impl transformer_engine
        --disable-bias-linear
        --use-rotary-position-embeddings
        --no-bias-gelu-fusion
        --no-bias-dropout-fusion
        --no-masked-softmax-fusion
        --no-gradient-accumulation-fusion
    )
elif [[ "${MODEL_FAMILY}" == "nemotron3-nano30b" ]]; then
    # Nemotron-3 Nano ~30B dense (approximate Llama-class shape; override layers/hidden/ffn from NVIDIA model card if needed)
    COMMON_ARGS=(
        "${BASE_TRAIN_ARGS[@]}"
        "${EXTRA_PARALLEL_ARGS[@]}"
        --ffn-hidden-size "${NEMOTRON3_FFN_HIDDEN_SIZE}"
        --group-query-attention
        --num-query-groups "${NEMOTRON3_NUM_QUERY_GROUPS}"
        --position-embedding-type rope
        --rotary-percent 1.0
        --rotary-base "${NEMOTRON3_ROPE_BASE}"
        --normalization RMSNorm
        --swiglu
        --untie-embeddings-and-output-weights
        --transformer-impl transformer_engine
        --disable-bias-linear
        --use-rotary-position-embeddings
        --no-bias-gelu-fusion
        --no-bias-dropout-fusion
        --no-masked-softmax-fusion
        --no-gradient-accumulation-fusion
    )
elif [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    # Qwen2 7B uses LLaMA architecture
    # Key difference: higher RoPE base (1M vs 10k) for better long-context extrapolation
    # Context parallelism (CP>1) requires TEDotProductAttention; local Megatron attention does not support CP.
    COMMON_ARGS=(
        "${BASE_TRAIN_ARGS[@]}"
        "${EXTRA_PARALLEL_ARGS[@]}"
        --position-embedding-type rope
        --rotary-percent 1.0
        --rotary-base 1000000
        --swiglu
        --untie-embeddings-and-output-weights
        --transformer-impl transformer_engine
        --disable-bias-linear
        --use-rotary-position-embeddings
        --no-bias-gelu-fusion
        --no-bias-dropout-fusion
        --no-masked-softmax-fusion
        --no-gradient-accumulation-fusion
    )
else
    # Llama 3 8B (long-context native 8k; GQA 8 KV heads; FFN 14336; RoPE theta 500k)
    # Transformer Engine (TENorm) supports RMSNorm; requires TE import (see pretrain_llama.py).
    COMMON_ARGS=(
        "${BASE_TRAIN_ARGS[@]}"
        "${EXTRA_PARALLEL_ARGS[@]}"
        --ffn-hidden-size "${LLAMA3_FFN_HIDDEN_SIZE}"
        --group-query-attention
        --num-query-groups "${LLAMA3_NUM_QUERY_GROUPS}"
        --position-embedding-type rope
        --rotary-percent 1.0
        --rotary-base "${LLAMA3_ROPE_BASE}"
        --normalization RMSNorm
        --swiglu
        --untie-embeddings-and-output-weights
        --transformer-impl transformer_engine
        --disable-bias-linear
        --use-rotary-position-embeddings
        --no-bias-gelu-fusion
        --no-bias-dropout-fusion
        --no-masked-softmax-fusion
        --no-gradient-accumulation-fusion
    )
fi

# PyTorch Profiler: --profiling-mode torch | scaletune | both (env for ScaleTune differs above)
# scaletune|both: must enable Megatron --profile + torch.profiler so refine_profile_after_training_profiler_step
# sees prof.events() and rank*.pt.trace.json lands under PROFILE_TENSORBOARD_DIR (default = roofline dir).
if [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    if [[ "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
        PROFILE_TENSORBOARD_DIR="${PROFILE_TENSORBOARD_DIR:-${SCALETUNE_ROOFLINE_OUTPUT_DIR}}"
        export PROFILE_TENSORBOARD_DIR
    fi
    mkdir -p "${PROFILE_TENSORBOARD_DIR}"
    # Omit --profile-ranks so Megatron defaults to all ranks (see arguments.py validate_args).
    # Warmup consumes the first iter in [start,end). For SCALETUNE_ROOFLINE_ITERATIONS=1,2 use [0,3):
    # iter0 warmup, iter1+iter2 active — both roofline JSONs get real GPU events.
    PROFILE_STEP_START="${PROFILE_STEP_START:-0}"
    PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
    COMMON_ARGS+=(
        --profile
        --use-pytorch-profiler
        --tensorboard-dir "${PROFILE_TENSORBOARD_DIR}"
        --profile-step-start "${PROFILE_STEP_START}"
        --profile-step-end "${PROFILE_STEP_END}"
    )
fi

# Forward all unknown CLI args to Megatron training scripts.
if [[ "${#PASSTHROUGH_ARGS[@]}" -gt 0 ]]; then
    COMMON_ARGS+=("${PASSTHROUGH_ARGS[@]}")
fi

# Fail fast if scaletune|both was requested but --profile was not appended (should not happen)
if [[ "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    _have_profile=0
    for _a in "${COMMON_ARGS[@]}"; do
        if [[ "${_a}" == "--profile" ]]; then _have_profile=1; break; fi
    done
    if [[ "${_have_profile}" -ne 1 ]]; then
        echo "ERROR: --profiling-mode ${PROFILING_MODE} requires torch profiler flags; internal COMMON_ARGS error." >&2
        exit 2
    fi
fi

# ============================================================================
# Print configuration
# ============================================================================
echo ""
if [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Note: Qwen2 7B supports native long-context 32k-128k with GQA and RoPE base=1M"
fi
echo "============================================================================"
if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
    echo "ScaleTune Mamba 8B Nemotron-H Profiling Configuration"
elif [[ "${MODEL_FAMILY}" == "falcon-mamba-7b" ]]; then
    echo "ScaleTune Falcon Mamba 7B Profiling Configuration"
elif [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Qwen2 7B Long-Context Profiling Configuration"
elif [[ "${MODEL_FAMILY}" == "qwen3-32b" ]]; then
    echo "Qwen3 32B Dense Profiling Configuration"
elif [[ "${MODEL_FAMILY}" == "qwen3-30b-a3b" ]]; then
    echo "Qwen3 30B-A3B MoE Profiling Configuration (pretrain_gpt.py)"
elif [[ "${MODEL_FAMILY}" == "nemotron3-nano30b" ]]; then
    echo "Nemotron-3 Nano ~30B Dense Profiling Configuration (defaults approximate; override from model card)"
else
    echo "ScaleTune Llama 3 8B Profiling Configuration"
fi
if [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Note: Qwen2 7B supports native long-context 32k-128k with GQA and RoPE base=1M"
fi
echo "============================================================================"
echo "MODEL_FAMILY=${MODEL_FAMILY}"
echo "Model: layers=${NUM_LAYERS} hidden=${HIDDEN_SIZE} heads=${NUM_HEADS}"
if [[ "${MODEL_FAMILY}" == "mamba8b" || "${MODEL_FAMILY}" == "falcon-mamba-7b" ]]; then
    echo "Mamba: ffn=${FFN_HIDDEN_SIZE} hybrid_attn=${HYBRID_ATTN_RATIO} hybrid_mlp=${HYBRID_MLP_RATIO} spec=${MAMBA_SPEC_MODULE}.${MAMBA_SPEC_NAME}"
elif [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Qwen2 7B: GQA with 4 KV heads, RoPE base=1M, native 32k-128k context"
elif [[ "${MODEL_FAMILY}" == "qwen3-32b" ]]; then
    echo "Qwen3 32B: GQA (${QWEN3_32B_NUM_QUERY_GROUPS} KV groups), ffn=${QWEN3_32B_FFN_HIDDEN_SIZE}, RoPE base=1M, max_pos=${MAX_POS_EMB}"
elif [[ "${MODEL_FAMILY}" == "qwen3-30b-a3b" ]]; then
    echo "Qwen3 30B-A3B MoE: experts=${QWEN3_30B_A3B_NUM_EXPERTS} topk=${QWEN3_30B_A3B_MOE_TOPK} moe_ffn=${QWEN3_30B_A3B_MOE_FFN_HIDDEN_SIZE} expert_mp=${EXPERT_MODEL_PARALLEL_SIZE}"
elif [[ "${MODEL_FAMILY}" == "nemotron3-nano30b" ]]; then
    echo "Nemotron-3 Nano ~30B: GQA (${NEMOTRON3_NUM_QUERY_GROUPS} KV groups), ffn=${NEMOTRON3_FFN_HIDDEN_SIZE}, RoPE base=${NEMOTRON3_ROPE_BASE}"
elif [[ "${MODEL_FAMILY}" == "llama" ]]; then
    echo "Llama 3 8B: GQA (${LLAMA3_NUM_QUERY_GROUPS} query groups / KV heads), ffn=${LLAMA3_FFN_HIDDEN_SIZE}, RoPE base=${LLAMA3_ROPE_BASE}, native 8k context"
fi
echo "Parallel: TP=${TP} PP=${PP} CP=${CP} DP=${DP} EP=${EP} ETP=${ETP} EDP=${EDP}"
if [[ -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" ]]; then
    echo "Pipeline schedule: interleaved 1F1B (num-layers-per-virtual-pipeline-stage=${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE})"
elif [[ -n "${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK}" ]]; then
    echo "Pipeline schedule: interleaved 1F1B (num-virtual-stages-per-pipeline-rank=${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK})"
else
    echo "Pipeline schedule: non-interleaved 1F1B"
fi
echo "Pipeline P2P overlap: no-overlap=${NO_OVERLAP_P2P_COMM} warmup-flush-overlap=${OVERLAP_P2P_COMM_WARMUP_FLUSH}"
echo "Sequence: seq_len=${SEQ_LEN} micro_bs=${MICRO_BS} global_bs=${GLOBAL_BS}"
echo "Training: iters=${TRAIN_ITERS} lr=${LR}"
echo "Memory: bf16=${USE_BF16} recompute=${RECOMPUTE_ACTIVATIONS}"
if [[ "${USE_MOCK_DATA}" == "1" ]]; then
    echo "Data: MOCK"
else
    echo "Data: ${DATA_PREFIX}"
    echo "Tokenizer: ${TOKENIZER_MODEL}"
fi
echo "Output: ${SCALETUNE_ROOFLINE_OUTPUT_DIR}"
if [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    echo "PyTorch Chrome traces (--tensorboard-dir): ${PROFILE_TENSORBOARD_DIR}"
    echo "GPU kernel profiling: ENABLED (traces will include CUDA kernel events)"
    echo "Profile train-iters window: [${PROFILE_STEP_START:-0}, ${PROFILE_STEP_END:-3}) (Kineto warmup=1 consumes the first iter in this window)"
    echo "ScaleTune roofline JSON iterations: ${SCALETUNE_ROOFLINE_ITERATIONS}"
    if [[ "${PROFILING_MODE}" == "torch" ]]; then
        echo "Profiling mode: torch — Megatron parallel-dim NVTX / comm wrapping: OFF (pure Kineto)"
    else
        echo "Profiling mode: ${PROFILING_MODE} — Megatron parallel-dim NVTX (enhanced_comm_profiler): ON"
    fi
fi
if [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Note: Qwen2 7B supports native long-context 32k-128k with GQA and RoPE base=1M"
fi
echo "============================================================================"
echo ""

# ============================================================================
# Run training
# ============================================================================
if [[ "${SCALETUNE_USE_EMBEDDED_SLURM_STEP}" == "1" ]]; then
    if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
        echo "Starting training (embedded Slurm step, SLURM_NTASKS=${SLURM_NTASKS}; no nested srun)..."
        echo ""
    fi
    exec bash "${WRAPPER_SCRIPT}" "${COMMON_ARGS[@]}"
fi

echo "Starting training with ${LAUNCH_CMD}..."
echo ""

eval "${LAUNCH_CMD}" bash "${WRAPPER_SCRIPT}" "${COMMON_ARGS[@]}"

echo ""
if [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Note: Qwen2 7B supports native long-context 32k-128k with GQA and RoPE base=1M"
fi
echo "============================================================================"
echo "Training complete!"
echo "Output: ${SCALETUNE_ROOFLINE_OUTPUT_DIR}/roofline_cluster_iter*.json"
if [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Note: Qwen2 7B supports native long-context 32k-128k with GQA and RoPE base=1M"
fi
if [[ "${MODEL_FAMILY}" == "qwen2-7b" ]]; then
    echo "Note: Qwen2 7B supports native long-context 32k-128k with GQA and RoPE base=1M"
fi
echo "============================================================================"
