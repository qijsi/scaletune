#!/bin/bash
# Run Megatron pretrain (default: LLaMA-style GPT) with ScaleTune roofline on
# Slurm-allocated nodes (from salloc), using srun or mpirun.
#
# Usage:
#   1. First, allocate nodes with salloc (adjust partition / GPU count for your site):
#      salloc -N 2 -n 2 --gpus-per-node=8 --time=02:00:00 --partition=a01
#
#   2. Then run this script from the MegaDLMs repo (or any cwd):
#      bash examples/scaletune/run_with_salloc.sh
#      bash examples/scaletune/run_with_salloc.sh --model mamba8b
#      bash examples/scaletune/run_with_salloc.sh -m llama
#
#   Profiling mode (ScaleTune roofline vs PyTorch Profiler vs baseline):
#      bash examples/scaletune/run_with_salloc.sh --profiling-mode scaletune   # default
#      bash examples/scaletune/run_with_salloc.sh --profiling-mode torch       # torch.profiler TensorBoard traces
#      bash examples/scaletune/run_with_salloc.sh --profiling-mode none
#      bash examples/scaletune/run_with_salloc.sh --profiling-mode both      # both (high overhead)
#
#   Megatron-LM tree is resolved as:
#     - MEGATRON_ROOT if set, else
#     - sibling ../Megatron-LM next to this MegaDLMs checkout (WORK/Megatron-LM).
#
#   Model can also be set with MODEL_FAMILY=llama (CLI overrides env).
#
# GPU count: GPUS_PER_NODE and world size are derived from Slurm (scontrol/squeue/env)
# unless you override GPUS_PER_NODE in the environment.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -m, --model FAMILY       Model family: llama | mamba8b (default: llama, or \$MODEL_FAMILY)
      --profiling-mode M   scaletune | torch | none | both (default: scaletune, or \$PROFILING_MODE)
  -h, --help               Show this help

Environment:
  MODEL_FAMILY         Same as --model if no CLI flag
  PROFILING_MODE       Same as --profiling-mode if no CLI flag
  GPUS_PER_NODE        Force GPUs per node (disables Slurm auto-detect)
  MEGATRON_ROOT        Path to Megatron-LM repo (required if not next to MegaDLMs)
  PROFILE_OUTPUT_ROOT  Base dir for profiling runs (default: MegaDLMs repo root)
  CONDA_PATH           Conda install prefix (optional; auto-detected if unset)
  SKIP_CONDA_ACTIVATE  Set to 1 if conda env is already active on compute nodes
  PROFILE_FRAMEWORK_TAG  Short name in profiling dir slug (default: megadlms)

Profiling layout: PROFILE_OUTPUT_ROOT (default: MegaDLMs repo root), then
  <slug>/torch_profiler, <slug>/scaletune_roofline, <slug>/scaletune_comm
  Slug: mode, framework (megadlms|..., \$PROFILE_FRAMEWORK_TAG), model (llama|qwen|mamba8b), tp, pp, recompute, seq, mbs, timestamp.
  Override dirs with PROFILE_TENSORBOARD_DIR, SCALETUNE_ROOFLINE_OUTPUT_DIR, SCALETUNE_OUTPUT_DIR.
Torch: PROFILE_STEP_START, PROFILE_STEP_END, PROFILE_RANKS
EOF
}

# Model family: mamba8b (Nemotron-H 8B hybrid) | llama (LLaMA-style GPT)
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
PROFILING_MODE="${PROFILING_MODE:-scaletune}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_FAMILY="${2:-}"
            if [[ -z "${MODEL_FAMILY}" ]]; then
                echo "ERROR: --model requires a value (llama | mamba8b)" >&2
                exit 2
            fi
            shift 2
            ;;
        --model=*)
            MODEL_FAMILY="${1#*=}"
            shift
            ;;
        --profiling-mode)
            PROFILING_MODE="${2:-}"
            if [[ -z "${PROFILING_MODE}" ]]; then
                echo "ERROR: --profiling-mode requires a value (scaletune | torch | none | both)" >&2
                exit 2
            fi
            shift 2
            ;;
        --profiling-mode=*)
            PROFILING_MODE="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${MODEL_FAMILY}" in
    llama|mamba8b) ;;
    *)
        echo "ERROR: MODEL_FAMILY must be llama or mamba8b, got: ${MODEL_FAMILY}" >&2
        exit 2
        ;;
esac

case "${PROFILING_MODE}" in
    scaletune|torch|none|both) ;;
    *)
        echo "ERROR: PROFILING_MODE must be scaletune, torch, none, or both, got: ${PROFILING_MODE}" >&2
        exit 2
        ;;
esac

# ============================================================================
# Configuration
# ============================================================================
CONDA_ENV="${CONDA_ENV:-aiperf_llm}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

resolve_conda_base() {
    if [[ -n "${CONDA_PATH:-}" ]] && [[ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
        echo "${CONDA_PATH}"
        return 0
    fi
    if [[ -n "${CONDA_EXE:-}" ]]; then
        local base
        base="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
        if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
            echo "${base}"
            return 0
        fi
    fi
    if command -v conda &>/dev/null; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${base}" ]] && [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
            echo "${base}"
            return 0
        fi
    fi
    for base in "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/mambaforge" "${HOME}/miniforge3"; do
        if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
            echo "${base}"
            return 0
        fi
    done
    return 1
}

if [[ "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
    if ! _conda_base="$(resolve_conda_base)"; then
        echo "ERROR: Could not find conda (conda.sh). Set CONDA_PATH to your install prefix," >&2
        echo "  or export SKIP_CONDA_ACTIVATE=1 if the training env is already active on all nodes." >&2
        exit 2
    fi
    CONDA_PATH="${_conda_base}"
fi

# Model configuration (defaults depend on MODEL_FAMILY unless you set overrides)
NUM_LAYERS="${NUM_LAYERS:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
NUM_HEADS="${NUM_HEADS:-32}"
MAX_POS_EMB="${MAX_POS_EMB:-4096}"
SEQ_LEN="${SEQ_LEN:-1024}"
MICRO_BS="${MICRO_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-2}"
LR="${LR:-3e-4}"
TP="${TP:-2}"
PP="${PP:-2}"
RECOMPUTE_ACTIVATIONS="${RECOMPUTE_ACTIVATIONS:-0}"

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

if [[ -z "${NUM_LAYERS}" ]]; then
    if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
        NUM_LAYERS=52
    else
        NUM_LAYERS=32
    fi
fi

# Data configuration
DATA_ROOT="${DATA_ROOT:-${HOME}/WORK/datasets/gpt_dataset}"
DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT}/redpajama_text_document}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${DATA_ROOT}/tokenizer.model}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
USE_MOCK_DATA="${USE_MOCK_DATA:-0}"

# Megatron-LM repo root (training scripts live there, not in MegaDLMs)
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${MEGATRON_ROOT:-}" ]]; then
    ROOT="$(cd "${MEGATRON_ROOT}" && pwd)"
elif [[ -d "${_SCRIPT_DIR}/../../../Megatron-LM" ]]; then
    ROOT="$(cd "${_SCRIPT_DIR}/../../../Megatron-LM" && pwd)"
else
    echo "ERROR: Megatron-LM repo not found. Set MEGATRON_ROOT to your checkout path." >&2
    echo "  Example: export MEGATRON_ROOT=/path/to/Megatron-LM" >&2
    exit 2
fi

# MegaDLMs repo root (this script lives under examples/scaletune/)
MEGA_DLMS_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

# Named profiling runs under MegaDLMs repo root (override base with PROFILE_OUTPUT_ROOT).
PROFILE_OUTPUT_ROOT="${PROFILE_OUTPUT_ROOT:-${MEGA_DLMS_ROOT}}"
TS_PROF="$(date +%Y%m%d_%H%M%S)"
PROFILE_FRAMEWORK_TAG="${PROFILE_FRAMEWORK_TAG:-megadlms}"
PROFILE_FRAMEWORK_TAG="$(echo "${PROFILE_FRAMEWORK_TAG}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_' | tr -s '_')"
TOK_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
if echo "${TOK_TYPE}" | grep -qi qwen; then
    PROFILE_MODEL_TAG="qwen"
elif [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
    PROFILE_MODEL_TAG="mamba8b"
else
    PROFILE_MODEL_TAG="llama"
fi

PROFILE_SLUG=""
PROF_BASE=""
if [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    PROFILE_SLUG="${PROFILING_MODE}_${PROFILE_FRAMEWORK_TAG}_${PROFILE_MODEL_TAG}_tp${TP}_pp${PP}_recompute${RECOMPUTE_ACTIVATIONS}_seq${SEQ_LEN}_mbs${MICRO_BS}_${TS_PROF}"
    PROF_BASE="${PROFILE_OUTPUT_ROOT}/${PROFILE_SLUG}"
    echo "Profiling artifact root: ${PROF_BASE}"
fi

# ScaleTune roofline JSON directory
if [[ -z "${SCALETUNE_ROOFLINE_OUTPUT_DIR:-}" ]]; then
    if [[ -n "${PROF_BASE}" ]] && [[ "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
        SCALETUNE_ROOFLINE_OUTPUT_DIR="${PROF_BASE}/scaletune_roofline"
    elif [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
        SCALETUNE_ROOFLINE_OUTPUT_DIR="${ROOT}/roofline_out_mamba8b"
    else
        SCALETUNE_ROOFLINE_OUTPUT_DIR="${ROOT}/roofline_out_llama7b"
    fi
fi
SCALETUNE_ROOFLINE_ITERATIONS="${SCALETUNE_ROOFLINE_ITERATIONS:-1,2}"

# PyTorch Profiler (Megatron training loop) — used when PROFILING_MODE is torch or both
if [[ -n "${PROF_BASE}" ]] && [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "both" ]]; then
    PROFILE_TENSORBOARD_DIR="${PROFILE_TENSORBOARD_DIR:-${PROF_BASE}/torch_profiler}"
fi
PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
# End is exclusive in the training loop: [--profile-step-start, --profile-step-end) (Megatron-aligned).
PROFILE_STEP_END="${PROFILE_STEP_END:-2}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_TRAIN_ARGS=()

case "${PROFILING_MODE}" in
    scaletune)
        export SCALETUNE_HIERARCHICAL_ROOFLINE=1
        export MEGATRON_ScaleTune_ROOFLINE=1
        export MEGATRON_COMM_DIM_PROFILE=1
        export MEGATRON_DETAILED_COMM_LOG="${MEGATRON_DETAILED_COMM_LOG:-1}"
        if [[ -n "${PROF_BASE}" ]]; then
            export SCALETUNE_OUTPUT_DIR="${SCALETUNE_OUTPUT_DIR:-${PROF_BASE}/scaletune_comm}"
        else
            export SCALETUNE_OUTPUT_DIR="${SCALETUNE_OUTPUT_DIR:-${ROOT}/comm_logs}"
        fi
        PROFILE_TRAIN_ARGS=()
        ;;
    torch)
        export SCALETUNE_HIERARCHICAL_ROOFLINE=0
        export MEGATRON_ScaleTune_ROOFLINE=0
        export MEGATRON_COMM_DIM_PROFILE=0
        export MEGATRON_DETAILED_COMM_LOG=0
        mkdir -p "${PROFILE_TENSORBOARD_DIR}"
        read -r -a _profile_ranks_arr <<< "${PROFILE_RANKS}"
        PROFILE_TRAIN_ARGS=(
            --profile
            --use-pytorch-profiler
            --tensorboard-dir "${PROFILE_TENSORBOARD_DIR}"
            --profile-step-start "${PROFILE_STEP_START}"
            --profile-step-end "${PROFILE_STEP_END}"
            --profile-ranks "${_profile_ranks_arr[@]}"
            --log-throughput
            --log-memory-to-tensorboard
            --log-interval "${PROFILE_LOG_INTERVAL:-1}"
        )
        ;;
    none)
        export SCALETUNE_HIERARCHICAL_ROOFLINE=0
        export MEGATRON_ScaleTune_ROOFLINE=0
        export MEGATRON_COMM_DIM_PROFILE=0
        export MEGATRON_DETAILED_COMM_LOG=0
        PROFILE_TRAIN_ARGS=()
        ;;
    both)
        export SCALETUNE_HIERARCHICAL_ROOFLINE=1
        export MEGATRON_ScaleTune_ROOFLINE=1
        export MEGATRON_COMM_DIM_PROFILE=1
        export MEGATRON_DETAILED_COMM_LOG="${MEGATRON_DETAILED_COMM_LOG:-1}"
        if [[ -n "${PROF_BASE}" ]]; then
            export SCALETUNE_OUTPUT_DIR="${SCALETUNE_OUTPUT_DIR:-${PROF_BASE}/scaletune_comm}"
        else
            export SCALETUNE_OUTPUT_DIR="${SCALETUNE_OUTPUT_DIR:-${ROOT}/comm_logs}"
        fi
        mkdir -p "${PROFILE_TENSORBOARD_DIR}"
        read -r -a _profile_ranks_arr <<< "${PROFILE_RANKS}"
        PROFILE_TRAIN_ARGS=(
            --profile
            --use-pytorch-profiler
            --tensorboard-dir "${PROFILE_TENSORBOARD_DIR}"
            --profile-step-start "${PROFILE_STEP_START}"
            --profile-step-end "${PROFILE_STEP_END}"
            --profile-ranks "${_profile_ranks_arr[@]}"
            --log-throughput
            --log-memory-to-tensorboard
            --log-interval "${PROFILE_LOG_INTERVAL:-1}"
        )
        echo "WARNING: PROFILING_MODE=both enables ScaleTune and PyTorch Profiler (high overhead). Prefer separate runs for a fair comparison." >&2
        ;;
esac

if [[ "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    mkdir -p "${SCALETUNE_ROOFLINE_OUTPUT_DIR}" "${SCALETUNE_OUTPUT_DIR}"
fi

# Memory optimization
USE_BF16="${USE_BF16:-1}"
EMPTY_UNUSED_MEMORY_LEVEL="${EMPTY_UNUSED_MEMORY_LEVEL:-1}"

# ============================================================================
# Check Slurm allocation
# ============================================================================
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

echo "============================================================================"
echo "Running on Slurm allocation"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Nodes: ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-?}}"
echo "  Node list: ${SLURM_JOB_NODELIST:-?}"
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

if [[ -n "${SLURM_NNODES:-}" ]] && [[ "${SLURM_NNODES}" -ne "${NUM_NODES}" ]]; then
    echo "WARNING: SLURM_NNODES (${SLURM_NNODES}) != hostfile node count (${NUM_NODES}); using hostfile for launch."
fi

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
# ScaleTune / comm profiling flags are set by PROFILING_MODE case above.
export SCALETUNE_HIERARCHICAL_ROOFLINE
export MEGATRON_ScaleTune_ROOFLINE
export MEGATRON_COMM_DIM_PROFILE
export MEGATRON_DETAILED_COMM_LOG
export SCALETUNE_ROOFLINE_OUTPUT_DIR="${SCALETUNE_ROOFLINE_OUTPUT_DIR}"
export SCALETUNE_ROOFLINE_ITERATIONS="${SCALETUNE_ROOFLINE_ITERATIONS}"
export TORCH_COMPILE_DEBUG="${TORCH_COMPILE_DEBUG:-0}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING="${TORCHINDUCTOR_COORDINATE_DESCENT_TUNING:-0}"
export TORCHINDUCTOR_FREEZING="${TORCHINDUCTOR_FREEZING:-0}"
export CONDA_ENV="${CONDA_ENV}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton_cache_${USER:-user}}"
    mkdir -p "${TRITON_CACHE_DIR}"
    export TRITON_CACHE_MANAGER="${TRITON_CACHE_MANAGER:-megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager}"
fi

# Use srun instead of mpirun for better Slurm integration
# srun automatically sets up WORLD_SIZE, RANK, etc. for PyTorch
if command -v srun &> /dev/null; then
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
    # Do NOT use --gpus-per-task=1 alone: it masks peer GPUs; NCCL NVML PCI lookup then fails.
    # Override with e.g. SRUN_GPU_BIND='--gpus-per-task=1' only if you know your site needs it (use wrapper unset CVD).
    if [[ -n "${SRUN_GPU_BIND:-}" ]]; then
        LAUNCH_CMD+=" ${SRUN_GPU_BIND}"
    else
        LAUNCH_CMD+=" --gpus-per-node=${GPUS_PER_NODE}"
    fi
else
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
    LAUNCH_CMD+=" -x MEGATRON_DETAILED_COMM_LOG=${MEGATRON_DETAILED_COMM_LOG:-0}"
    if [[ -n "${SCALETUNE_OUTPUT_DIR:-}" ]]; then
        LAUNCH_CMD+=" -x SCALETUNE_OUTPUT_DIR=${SCALETUNE_OUTPUT_DIR}"
    fi
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
    if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
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

# Paths (expanded when this wrapper is generated)
CONDA_PATH="${CONDA_PATH:-}"
CONDA_ENV="${CONDA_ENV}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE}"
ROOT="${ROOT}"

# Activate conda unless caller already has the training env on PATH (all nodes)
if [[ "\${SKIP_CONDA_ACTIVATE}" == "1" ]]; then
    :
elif [[ -f "\${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    source "\${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "\${CONDA_ENV}"
else
    echo "ERROR: conda.sh not found at \${CONDA_PATH}/etc/profile.d/conda.sh (set CONDA_PATH or SKIP_CONDA_ACTIVATE=1)" >&2
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
    if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
        TRAIN_SCRIPT="pretrain_mamba.py"
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

TOK_ARGS=()
if [[ "${USE_MOCK_DATA}" == "1" && "${MODEL_FAMILY}" == "mamba8b" ]]; then
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
    if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
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
    "${TOK_ARGS[@]}"
    "${DATA_ARGS[@]}"
    --skip-eval
)

if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
    COMMON_ARGS=(
        "${BASE_TRAIN_ARGS[@]}"
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
else
    COMMON_ARGS=(
        "${BASE_TRAIN_ARGS[@]}"
        --swiglu
    )
fi

if [[ "${#PROFILE_TRAIN_ARGS[@]}" -gt 0 ]]; then
    COMMON_ARGS+=("${PROFILE_TRAIN_ARGS[@]}")
fi

if [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "both" ]]; then
    if [[ "${TRAIN_ITERS}" -lt "${PROFILE_STEP_END}" ]]; then
        echo "WARNING: TRAIN_ITERS (${TRAIN_ITERS}) < PROFILE_STEP_END (${PROFILE_STEP_END}); PyTorch Profiler may not capture the intended steps." >&2
    fi
fi

# ============================================================================
# Print configuration
# ============================================================================
echo ""
echo "============================================================================"
if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
    echo "ScaleTune Mamba 8B (Nemotron-H) Profiling Configuration"
else
    echo "ScaleTune LLaMA 7B Profiling Configuration"
fi
echo "============================================================================"
echo "MODEL_FAMILY=${MODEL_FAMILY}"
echo "PROFILING_MODE=${PROFILING_MODE}  (scaletune | torch | none | both)"
if [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "both" ]]; then
    echo "PyTorch Profiler: tensorboard_dir=${PROFILE_TENSORBOARD_DIR} steps=[${PROFILE_STEP_START},${PROFILE_STEP_END}) ranks=${PROFILE_RANKS}"
fi
echo "Model: layers=${NUM_LAYERS} hidden=${HIDDEN_SIZE} heads=${NUM_HEADS}"
if [[ "${MODEL_FAMILY}" == "mamba8b" ]]; then
    echo "Mamba: ffn=${FFN_HIDDEN_SIZE} hybrid_attn=${HYBRID_ATTN_RATIO} hybrid_mlp=${HYBRID_MLP_RATIO} spec=${MAMBA_SPEC_MODULE}.${MAMBA_SPEC_NAME}"
fi
echo "Parallel: TP=${TP} PP=${PP}"
echo "Sequence: seq_len=${SEQ_LEN} micro_bs=${MICRO_BS} global_bs=${GLOBAL_BS}"
echo "Training: iters=${TRAIN_ITERS} lr=${LR}"
echo "Memory: bf16=${USE_BF16} recompute=${RECOMPUTE_ACTIVATIONS}"
if [[ "${USE_MOCK_DATA}" == "1" ]]; then
    echo "Data: MOCK"
else
    echo "Data: ${DATA_PREFIX}"
    echo "Tokenizer: ${TOKENIZER_MODEL}"
fi
if [[ "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    echo "ScaleTune roofline output: ${SCALETUNE_ROOFLINE_OUTPUT_DIR}"
    echo "ScaleTune comm output: ${SCALETUNE_OUTPUT_DIR}"
fi
if [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "both" ]]; then
    echo "PyTorch Profiler traces (TensorBoard): ${PROFILE_TENSORBOARD_DIR}"
fi
if [[ "${PROFILING_MODE}" == "none" ]]; then
    echo "Profiling: disabled (ScaleTune + torch profiler off)"
fi
echo "============================================================================"
echo ""

# ============================================================================
# Run training
# ============================================================================
echo "Starting training with ${LAUNCH_CMD}..."
echo ""

eval "${LAUNCH_CMD}" bash "${WRAPPER_SCRIPT}" "${COMMON_ARGS[@]}"

echo ""
echo "============================================================================"
echo "Training complete!"
if [[ "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    echo "ScaleTune roofline: ${SCALETUNE_ROOFLINE_OUTPUT_DIR}/roofline_cluster_iter*.json"
    echo "ScaleTune comm: ${SCALETUNE_OUTPUT_DIR}"
fi
if [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "both" ]]; then
    echo "PyTorch Profiler: tensorboard --logdir ${PROFILE_TENSORBOARD_DIR}"
    echo "  Chrome traces: ${PROFILE_TENSORBOARD_DIR}/rank*.pt.trace.json"
    echo "  Sidecar: peak_memory.json in the same dir (if enabled); TFLOP/s via training log (--log-throughput)"
fi
echo "============================================================================"
