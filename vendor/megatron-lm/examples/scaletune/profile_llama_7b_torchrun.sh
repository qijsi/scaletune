#!/usr/bin/env bash
# Profile LLaMA pretrain with ScaleTune hierarchical roofline + comm_dim tags.
#
# Supports two execution modes:
# 1. Local torchrun (default): Uses torchrun on current node
# 2. Slurm + mpirun: Uses mpirun on Slurm-allocated nodes from login node
#
# Usage (local):
#   bash examples/scaletune/profile_llama_7b_torchrun.sh
#
# Usage (Slurm + mpirun from login node):
#   RUN_MODE=slurm_mpirun SLURM_JOB_ID=<job_id> bash examples/scaletune/profile_llama_7b_torchrun.sh
#   # Or with sbatch:
#   sbatch -N 2 -n 2 --gpus-per-node=8 examples/scaletune/profile_llama_7b_torchrun.sh
#
# Real data + tokenizer (default): DATA_ROOT=${HOME}/WORK/datasets/gpt_dataset,
# DATA_PREFIX=${DATA_ROOT}/redpajama_text_document, tokenizer from tokenizer.model or tokenizer.json.
#
# Conda environment: aiperf_llm (configurable via CONDA_ENV)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Conda environment
CONDA_ENV="${CONDA_ENV:-aiperf_llm}"

# Execution mode: local (torchrun) or slurm_mpirun
RUN_MODE="${RUN_MODE:-local}"

# Reduce allocator fragmentation OOMs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Megatron validate_args: assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# Avoid NCCL NVML PCI lookup failures under Slurm GPU binding (nvmlDeviceGetHandleByPciBusId).
export NCCL_NVML_DISABLE="${NCCL_NVML_DISABLE:-1}"

# ---------------------------------------------------------------------------
# comm_dim + hierarchical roofline (must be set before Python starts)
# ---------------------------------------------------------------------------
export SCALETUNE_HIERARCHICAL_ROOFLINE=1
export MEGATRON_ScaleTune_ROOFLINE=1
export MEGATRON_COMM_DIM_PROFILE=1
unset SCALETUNE_COMM_DIM_PROFILE 2>/dev/null || true

export SCALETUNE_ROOFLINE_OUTPUT_DIR="${SCALETUNE_ROOFLINE_OUTPUT_DIR:-${ROOT}/roofline_out_llama7b}"
export SCALETUNE_ROOFLINE_ITERATIONS="${SCALETUNE_ROOFLINE_ITERATIONS:-1,2}"
mkdir -p "${SCALETUNE_ROOFLINE_OUTPUT_DIR}"

# Disable torch.compile autotuning
export TORCH_COMPILE_DEBUG=0
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
export TORCHINDUCTOR_FREEZING=0

# ---------------------------------------------------------------------------
# Slurm + mpirun mode setup
# ---------------------------------------------------------------------------
if [[ "${RUN_MODE}" == "slurm_mpirun" ]]; then
    # Check if running under Slurm
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: RUN_MODE=slurm_mpirun requires SLURM_JOB_ID to be set"
        echo "  Example: RUN_MODE=slurm_mpirun SLURM_JOB_ID=12345 bash $0"
        exit 2
    fi
    
    # Load OpenMPI module if mpirun not in PATH
    if ! command -v mpirun &> /dev/null; then
        if command -v module &> /dev/null; then
            echo "Loading OpenMPI module..."
            module load mpi/openmpi-4.1.4-gcc12-cuda12.6 2>/dev/null || \
            module load openmpi 2>/dev/null || \
            echo "Warning: Could not load OpenMPI module, mpirun may not be available"
        fi
    fi
    
    # Verify mpirun is available
    if ! command -v mpirun &> /dev/null; then
        echo "ERROR: mpirun not found after loading modules"
        echo "  Try: module load mpi/openmpi-4.1.4-gcc12-cuda12.6"
        echo "  Or install OpenMPI in your conda environment: conda install -c conda-forge openmpi"
        exit 2
    fi
    
    MPI_VERSION=$(mpirun --version 2>&1 | head -1)
    echo "Using mpirun: $(which mpirun)"
    echo "  ${MPI_VERSION}"
    
    # Get node list from Slurm
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        # Parse Slurm node list (handles compact format like node[01-04])
        if command -v scontrol &> /dev/null; then
            HOSTFILE=$(mktemp)
            scontrol show hostname "${SLURM_JOB_NODELIST}" > "${HOSTFILE}"
        else
            # Fallback: assume simple format
            HOSTFILE=$(mktemp)
            echo "${SLURM_JOB_NODELIST}" | tr ',' '\n' > "${HOSTFILE}"
        fi
    elif [[ -f "${SLURM_HOSTFILE:-}" ]]; then
        HOSTFILE="${SLURM_HOSTFILE}"
    else
        echo "ERROR: Cannot determine hostfile for mpirun"
        echo "  Set SLURM_JOB_NODELIST or SLURM_HOSTFILE"
        exit 2
    fi
    
    # Count nodes and GPUs
    NUM_NODES=$(wc -l < "${HOSTFILE}")
    GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
    TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
    
    # Set master node (first node in list)
    MASTER_ADDR=$(head -n 1 "${HOSTFILE}")
    MASTER_PORT="${MASTER_PORT:-6000}"
    
    # MPI configuration
    # Use one process per GPU (each process will see CUDA_VISIBLE_DEVICES)
    MPI_CMD="mpirun"
    MPI_CMD+=" --hostfile ${HOSTFILE}"
    MPI_CMD+=" -np ${NUM_NODES}"  # One process per node, each will spawn GPU processes via torchrun
    MPI_CMD+=" --map-by ppr:${GPUS_PER_NODE}:node"  # Map GPU processes per node
    MPI_CMD+=" -x MASTER_ADDR=${MASTER_ADDR}"
    MPI_CMD+=" -x MASTER_PORT=${MASTER_PORT}"
    MPI_CMD+=" -x CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
    MPI_CMD+=" -x NCCL_NVML_DISABLE=${NCCL_NVML_DISABLE}"
    MPI_CMD+=" -x PYTHONPATH=${PYTHONPATH}"
    MPI_CMD+=" -x SCALETUNE_HIERARCHICAL_ROOFLINE=${SCALETUNE_HIERARCHICAL_ROOFLINE}"
    MPI_CMD+=" -x MEGATRON_ScaleTune_ROOFLINE=${MEGATRON_ScaleTune_ROOFLINE}"
    MPI_CMD+=" -x MEGATRON_COMM_DIM_PROFILE=${MEGATRON_COMM_DIM_PROFILE}"
    MPI_CMD+=" -x SCALETUNE_ROOFLINE_OUTPUT_DIR=${SCALETUNE_ROOFLINE_OUTPUT_DIR}"
    MPI_CMD+=" -x SCALETUNE_ROOFLINE_ITERATIONS=${SCALETUNE_ROOFLINE_ITERATIONS}"
    MPI_CMD+=" -x TORCH_COMPILE_DEBUG=${TORCH_COMPILE_DEBUG}"
    MPI_CMD+=" -x TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS}"
    MPI_CMD+=" -x TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=${TORCHINDUCTOR_COORDINATE_DESCENT_TUNING}"
    MPI_CMD+=" -x TORCHINDUCTOR_FREEZING=${TORCHINDUCTOR_FREEZING}"
    
    # Conda activation for mpirun
    CONDA_PATH="${CONDA_PATH:-/home/fit/zhaijdzq/WORK/miniconda3}"
    # Use absolute path in the command string to avoid variable expansion issues
    CONDA_ACTIVATE=". ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"
    
    # Export all necessary environment variables to remote nodes
    MPI_CMD+=" -x PATH=${PATH}"
    MPI_CMD+=" -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    MPI_CMD+=" -x CONDA_PATH=${CONDA_PATH}"
    MPI_CMD+=" -x CONDA_ENV=${CONDA_ENV}"
    MPI_CMD+=" -x HOME=${HOME}"
    MPI_CMD+=" -x USER=${USER:-}"
    
    echo "============================================================================"
    echo "ScaleTune profiling with Slurm + mpirun"
    echo "  Job ID: ${SLURM_JOB_ID}"
    echo "  Nodes: ${NUM_NODES}"
    echo "  GPUs per node: ${GPUS_PER_NODE}"
    echo "  Total GPUs: ${TOTAL_GPUS}"
    echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
    echo "  Hostfile: ${HOSTFILE}"
    echo "  Conda env: ${CONDA_ENV}"
    echo "============================================================================"
    
    # Cleanup temp hostfile on exit
    trap "rm -f ${HOSTFILE}" EXIT
fi

# ---------------------------------------------------------------------------
# Model and data configuration
# ---------------------------------------------------------------------------
TP="${TP:-2}"
PP="${PP:-2}"

SEQ_LEN="${SEQ_LEN:-1024}"
MICRO_BS="${MICRO_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-2}"
LR="${LR:-3e-4}"

NUM_LAYERS="${NUM_LAYERS:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
NUM_HEADS="${NUM_HEADS:-32}"
MAX_POS_EMB="${MAX_POS_EMB:-4096}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"

DATA_ROOT="${DATA_ROOT:-${HOME}/WORK/datasets/gpt_dataset}"
DATA_PREFIX="${DATA_PREFIX:-}"
if [[ -z "${DATA_PREFIX}" ]]; then
    DATA_PREFIX="${DATA_ROOT}/redpajama_text_document"
fi
SPLIT="${SPLIT:-969,30,1}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${DATA_ROOT}/.megatron_data_cache}"

USE_MOCK_DATA="${USE_MOCK_DATA:-0}"
DISABLE_GLOO_GROUPS="${DISABLE_GLOO_GROUPS:-1}"
SKIP_EVAL="${SKIP_EVAL:-1}"
USE_BF16="${USE_BF16:-1}"
RECOMPUTE_ACTIVATIONS="${RECOMPUTE_ACTIVATIONS:-1}"
EMPTY_UNUSED_MEMORY_LEVEL="${EMPTY_UNUSED_MEMORY_LEVEL:-1}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-pretrain_llama.py}"
if [[ ! -f "${ROOT}/${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: training script not found: ${ROOT}/${TRAIN_SCRIPT}"
    exit 2
fi

# Build argument arrays
EXTRA_ARGS=()
if [[ "${SKIP_EVAL}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-eval)
fi

DISTRIBUTED_ARGS=(
    --distributed-backend nccl
    --tp-comm-bootstrap-backend nccl
)
if [[ "${DISABLE_GLOO_GROUPS}" == "1" ]]; then
    DISTRIBUTED_ARGS+=(--disable-gloo-process-groups)
fi

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

TOK_ARGS=()
DATA_ARGS=()
if [[ "${USE_MOCK_DATA}" == "1" ]]; then
    DATA_ARGS+=(--mock-data)
    TOK_ARGS+=(--tokenizer-type NullTokenizer --vocab-size "${VOCAB_SIZE}")
else
    if [[ ! -d "${DATA_ROOT}" ]]; then
        echo "ERROR: DATA_ROOT is not a directory: ${DATA_ROOT}"
        echo "  Set DATA_ROOT or USE_MOCK_DATA=1 for synthetic data."
        exit 2
    fi
    mkdir -p "${DATA_CACHE_PATH}"
    DATA_ARGS+=(
        --data-path "${DATA_PREFIX}"
        --split "${SPLIT}"
        --data-cache-path "${DATA_CACHE_PATH}"
    )
    TOKENIZER_TYPE="${TOKENIZER_TYPE:-}"
    TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"
    if [[ -z "${TOKENIZER_MODEL}" && -f "${DATA_ROOT}/tokenizer.json" ]]; then
        TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
        TOKENIZER_MODEL="${DATA_ROOT}"
    elif [[ -z "${TOKENIZER_MODEL}" && -f "${DATA_ROOT}/tokenizer.model" ]]; then
        TOKENIZER_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
        TOKENIZER_MODEL="${DATA_ROOT}/tokenizer.model"
    fi
    TOKENIZER_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
    TOK_ARGS+=(--tokenizer-type "${TOKENIZER_TYPE}")
    if [[ -n "${TOKENIZER_MODEL}" ]]; then
        TOK_ARGS+=(--tokenizer-model "${TOKENIZER_MODEL}")
    else
        echo "ERROR: Real data requires --tokenizer-model."
        echo "  Set TOKENIZER_MODEL to your .model file or HF folder."
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# Print configuration
# ---------------------------------------------------------------------------
echo "============================================================================"
echo "ScaleTune comm_dim + hierarchical roofline (LLaMA profile)"
echo "  Mode: ${RUN_MODE}"
echo "  TRAIN_SCRIPT=${TRAIN_SCRIPT}"
if [[ "${RUN_MODE}" == "slurm_mpirun" ]]; then
    echo "  nproc_per_node=${GPUS_PER_NODE}  TP=${TP} PP=${PP}"
else
    echo "  nproc_per_node=${MEGATRON_GPN:-8}  TP=${TP} PP=${PP}"
fi
echo "  num_layers=${NUM_LAYERS} hidden=${HIDDEN_SIZE} heads=${NUM_HEADS} max_pos=${MAX_POS_EMB}"
echo "  SEQ_LEN=${SEQ_LEN} MICRO_BS=${MICRO_BS} GLOBAL_BS=${GLOBAL_BS} TRAIN_ITERS=${TRAIN_ITERS} LR=${LR}"
echo "  memory: USE_BF16=${USE_BF16} RECOMPUTE_ACTIVATIONS=${RECOMPUTE_ACTIVATIONS} EMPTY_UNUSED_MEMORY_LEVEL=${EMPTY_UNUSED_MEMORY_LEVEL}"
echo "  torch.compile: disabled"
echo "  comm profiler: enhanced (MEGATRON_COMM_DIM_PROFILE=1)"
echo "  ScaleTune profiler: enabled"
if [[ "${USE_MOCK_DATA}" == "1" ]]; then
    echo "  data: MOCK (--mock-data)"
    echo "  tokenizer: NullTokenizer vocab_size=${VOCAB_SIZE}"
else
    echo "  data-path: ${DATA_PREFIX}  split=${SPLIT}  cache=${DATA_CACHE_PATH}"
    echo "  tokenizer: type=${TOKENIZER_TYPE:-?} model=${TOKENIZER_MODEL}"
fi
echo "  OUT=${SCALETUNE_ROOFLINE_OUTPUT_DIR}"
echo "============================================================================"

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
if [[ "${RUN_MODE}" == "slurm_mpirun" ]]; then
    # Mpirun mode
    COMMON_ARGS=(
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
        --swiglu
        "${MEMORY_ARGS[@]}"
        "${DISTRIBUTED_ARGS[@]}"
        "${TOK_ARGS[@]}"
        "${DATA_ARGS[@]}"
        "${EXTRA_ARGS[@]}"
    )
    
    # Activate conda and run with mpirun
    # Use a wrapper script to ensure proper conda activation on remote nodes
    WRAPPER_SCRIPT=$(mktemp)
    cat > "${WRAPPER_SCRIPT}" << WRAPPER_EOF
#!/bin/bash
set -euo pipefail

# Use hardcoded absolute path to avoid variable expansion issues
CONDA_PATH="/home/fit/zhaijdzq/WORK/miniconda3"
CONDA_ENV="${CONDA_ENV}"
ROOT="${ROOT}"
TRAIN_SCRIPT="${TRAIN_SCRIPT}"

# Activate conda using absolute path
if [[ -f "\${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    source "\${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "\${CONDA_ENV}"
else
    echo "ERROR: conda.sh not found at \${CONDA_PATH}/etc/profile.d/conda.sh" >&2
    exit 2
fi

# Map Slurm / OpenMPI launcher env to PyTorch (Megatron defaults RANK to 0 if unset).
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
# Match run_with_salloc.sh: Slurm per-task GPU masking breaks NCCL NVML; unset CVD + SLURM_LOCALID.
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

cd "\${ROOT}"
exec python "\${TRAIN_SCRIPT}" "\$@"
WRAPPER_EOF
    chmod +x "${WRAPPER_SCRIPT}"
    
    # Run with mpirun - use bash explicitly
    eval "${MPI_CMD}" bash "${WRAPPER_SCRIPT}" "${COMMON_ARGS[@]}"
    
    # Cleanup
    rm -f "${WRAPPER_SCRIPT}"
else
    # Local torchrun mode
    MEGATRON_GPN="${MEGATRON_GPN:-8}"
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-6000}"
    
    CONDA_PATH="${CONDA_PATH:-/home/fit/zhaijdzq/WORK/miniconda3}"
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    
    torchrun \
        --nproc_per_node="${MEGATRON_GPN}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${TRAIN_SCRIPT}" \
        --tensor-model-parallel-size "${TP}" \
        --pipeline-model-parallel-size "${PP}" \
        --num-layers "${NUM_LAYERS}" \
        --hidden-size "${HIDDEN_SIZE}" \
        --num-attention-heads "${NUM_HEADS}" \
        --max-position-embeddings "${MAX_POS_EMB}" \
        --seq-length "${SEQ_LEN}" \
        --micro-batch-size "${MICRO_BS}" \
        --global-batch-size "${GLOBAL_BS}" \
        --train-iters "${TRAIN_ITERS}" \
        --lr "${LR}" \
        --swiglu \
        "${MEMORY_ARGS[@]}" \
        "${DISTRIBUTED_ARGS[@]}" \
        "${TOK_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
fi

echo "Done. Outputs:"
echo "  ${SCALETUNE_ROOFLINE_OUTPUT_DIR}/roofline_cluster_iter*.json"
