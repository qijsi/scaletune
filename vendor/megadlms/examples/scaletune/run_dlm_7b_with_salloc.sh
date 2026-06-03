#!/bin/bash
# Run DLM 7B pretrain on an existing Slurm allocation (from salloc/sbatch shell).
#
# Usage:
#   1) Allocate resources first:
#      salloc -N 2 -n 2 --gpus-per-node=8 --time=02:00:00 --partition=a01
#   2) Run from login node (or any node with same shared filesystem):
#      bash examples/scaletune/run_dlm_7b_with_salloc.sh
#      bash examples/scaletune/run_dlm_7b_with_salloc.sh --profiling-mode torch
#      bash examples/scaletune/run_dlm_7b_with_salloc.sh convert_ckpt
#      bash examples/scaletune/run_dlm_7b_with_salloc.sh convert_ckpt 12000
#
# Required env vars:
#   PROJECT_DIR   MegaDLMs repo root
#
# Optional but recommended:
#   DATASETS_DIR  Dataset root directory (default: ~/WORK/datasets/gpt_dataset)
#   CKPT_DIR      Checkpoint/log root (default: /tmp/${USER}/megadlm_profile_tmp)
#
# Optional env vars:
#   GPUS_PER_NODE, MASTER_PORT, PARTITION, CONDA_PATH, CONDA_ENV, SKIP_CONDA_ACTIVATE
#   MODEL_PARALLEL_SIZE, PIPELINE_MODEL_PARALLEL_SIZE, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE
#   PROFILE_TENSORBOARD_DIR, PROFILE_STEP_START, PROFILE_STEP_END, PROFILE_RANKS
#   RECOMPUTE_ACTIVATIONS (0|1), DL_MODEL_FAMILY (llama|qwen, optional; inferred from TOKENIZER_TYPE)
#   PROFILE_OUTPUT_ROOT (optional base dir; default: PROJECT_DIR / MegaDLMs repo root)

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [--profiling-mode torch|scaletune|none|both] [convert_ckpt] [ckpt_step]

Profiling output layout (torch / scaletune / both):
  \${PROFILE_OUTPUT_ROOT:-<repo>}/<slug>/
    torch:     torch_profiler/   (PyTorch profiler TensorBoard)
    scaletune: scaletune_roofline/  scaletune_comm/  (roofline JSON + comm logs)

Slug includes: profiling mode, model family (llama|qwen), tp, pp, recompute flag,
               seq length, micro batch size, timestamp.

Examples:
  bash examples/scaletune/run_dlm_7b_with_salloc.sh
  bash examples/scaletune/run_dlm_7b_with_salloc.sh --profiling-mode scaletune
  bash examples/scaletune/run_dlm_7b_with_salloc.sh --profiling-mode torch
  bash examples/scaletune/run_dlm_7b_with_salloc.sh --profiling-mode both
  bash examples/scaletune/run_dlm_7b_with_salloc.sh convert_ckpt
  bash examples/scaletune/run_dlm_7b_with_salloc.sh convert_ckpt 12000

Defaults:
  PROFILE_OUTPUT_ROOT=<MegaDLMs repo root (PROJECT_DIR)>
  DATASETS_DIR=~/WORK/datasets/gpt_dataset
  CKPT_DIR=/tmp/\${USER}/megadlm_profile_tmp
  DISABLE_CHECKPOINT_SAVE=1
EOF
}

CONDA_ENV="${CONDA_ENV:-aiperf_llm}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"
PROFILING_MODE="${PROFILING_MODE:-none}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profiling-mode)
            PROFILING_MODE="${2:-}"
            if [[ -z "${PROFILING_MODE}" ]]; then
                echo "ERROR: --profiling-mode requires a value." >&2
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
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

CONVERT_CHECKPOINT_ONLY="${POSITIONAL_ARGS[0]:-}"
CONVERT_CKPT_STEP="${POSITIONAL_ARGS[1]:-}"

if [[ "${CONVERT_CHECKPOINT_ONLY}" == "-h" || "${CONVERT_CHECKPOINT_ONLY}" == "--help" ]]; then
    usage
    exit 0
fi

case "${PROFILING_MODE}" in
    scaletune|torch|none|both) ;;
    *)
        echo "ERROR: profiling mode must be scaletune|torch|none|both, got ${PROFILING_MODE}" >&2
        exit 2
        ;;
esac

DATASETS_DIR="${DATASETS_DIR:-$HOME/WORK/datasets/gpt_dataset}"
CKPT_DIR="${CKPT_DIR:-/tmp/${USER}/megadlm_profile_tmp}"
DISABLE_CHECKPOINT_SAVE="${DISABLE_CHECKPOINT_SAVE:-1}"

TS_PROF="$(date +%Y%m%d_%H%M%S)"
PROFILE_FRAMEWORK_TAG="${PROFILE_FRAMEWORK_TAG:-megadlms}"
PROFILE_FRAMEWORK_TAG="$(echo "${PROFILE_FRAMEWORK_TAG}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_' | tr -s '_')"
TP_VAL="${MODEL_PARALLEL_SIZE:-2}"
PP_VAL="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
SEQ_VAL="${SEQ_LENGTH:-2048}"
MBS_VAL="${MICRO_BATCH_SIZE:-4}"
RECOMP_VAL="${RECOMPUTE_ACTIVATIONS:-0}"
TOK_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"

if [[ -n "${DL_MODEL_FAMILY:-}" ]]; then
    DL_MODEL_FAMILY="$(echo "${DL_MODEL_FAMILY}" | tr '[:upper:]' '[:lower:]')"
else
    if echo "${TOK_TYPE}" | grep -qi qwen; then
        DL_MODEL_FAMILY="qwen"
    else
        DL_MODEL_FAMILY="llama"
    fi
fi

PROFILE_OUTPUT_ROOT="${PROFILE_OUTPUT_ROOT:-${PROJECT_DIR}}"
PROFILE_SLUG=""
PROF_BASE=""
PROFILE_TENSORBOARD_RESOLVED=""
SCALETUNE_ROOF_RESOLVED=""
SCALETUNE_COMM_RESOLVED=""
WRAPPER_PROFILE_EXPORTS=""

if [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    PROFILE_SLUG="${PROFILING_MODE}_${PROFILE_FRAMEWORK_TAG}_${DL_MODEL_FAMILY}_tp${TP_VAL}_pp${PP_VAL}_recompute${RECOMP_VAL}_seq${SEQ_VAL}_mbs${MBS_VAL}_${TS_PROF}"
    PROF_BASE="${PROFILE_OUTPUT_ROOT}/${PROFILE_SLUG}"
    PROFILE_TENSORBOARD_RESOLVED="${PROFILE_TENSORBOARD_DIR:-${PROF_BASE}/torch_profiler}"
    SCALETUNE_ROOF_RESOLVED="${SCALETUNE_ROOFLINE_OUTPUT_DIR:-${PROF_BASE}/scaletune_roofline}"
    SCALETUNE_COMM_RESOLVED="${SCALETUNE_OUTPUT_DIR:-${PROF_BASE}/scaletune_comm}"

    if [[ "${PROFILING_MODE}" == "torch" || "${PROFILING_MODE}" == "both" ]]; then
        mkdir -p "${PROFILE_TENSORBOARD_RESOLVED}"
        WRAPPER_PROFILE_EXPORTS+="export PROFILE_TENSORBOARD_DIR=\"${PROFILE_TENSORBOARD_RESOLVED}\""$'\n'
    fi
    if [[ "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
        mkdir -p "${SCALETUNE_ROOF_RESOLVED}" "${SCALETUNE_COMM_RESOLVED}"
        WRAPPER_PROFILE_EXPORTS+="export SCALETUNE_ROOFLINE_OUTPUT_DIR=\"${SCALETUNE_ROOF_RESOLVED}\""$'\n'
        WRAPPER_PROFILE_EXPORTS+="export SCALETUNE_OUTPUT_DIR=\"${SCALETUNE_COMM_RESOLVED}\""$'\n'
    fi

    echo "Profiling artifact root: ${PROF_BASE}"
fi

DLM_SCRIPT="${PROJECT_DIR}/examples/dlm_training/dlm_pretrain_7b.sh"
if [[ ! -f "${DLM_SCRIPT}" ]]; then
    echo "ERROR: Script not found: ${DLM_SCRIPT}" >&2
    exit 2
fi

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
    return 1
}

if [[ "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
    if ! _conda_base="$(resolve_conda_base)"; then
        echo "ERROR: Could not find conda. Set CONDA_PATH or SKIP_CONDA_ACTIVATE=1." >&2
        exit 2
    fi
    CONDA_PATH="${_conda_base}"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    CURRENT_JOB=$(squeue -u "${USER}" -t R -h -o "%i %N" 2>/dev/null | head -1 || true)
    if [[ -n "${CURRENT_JOB}" ]]; then
        SLURM_JOB_ID=$(echo "${CURRENT_JOB}" | awk '{print $1}')
        SLURM_JOB_NODELIST=$(echo "${CURRENT_JOB}" | awk '{print $2}')
    else
        echo "ERROR: No active Slurm allocation found." >&2
        echo "Please allocate first, e.g.:" >&2
        echo "  salloc -N 2 -n 2 --gpus-per-node=8 --time=02:00:00 --partition=a01" >&2
        exit 2
    fi
fi

HOSTFILE=$(mktemp)
if command -v scontrol &>/dev/null; then
    scontrol show hostname "${SLURM_JOB_NODELIST}" > "${HOSTFILE}"
else
    echo "${SLURM_JOB_NODELIST}" | tr ',' '\n' > "${HOSTFILE}"
fi
NUM_NODES=$(wc -l < "${HOSTFILE}")

resolve_gpus_per_node_from_slurm() {
    local g="" gres
    if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
        echo "${SLURM_GPUS_PER_NODE}"
        return 0
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol &>/dev/null; then
        g=$(
            scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
                | awk 'match($0, /gres\/gpu=[0-9]+/) {print substr($0, RSTART+9, RLENGTH-9); exit}'
        )
    fi
    if [[ -z "${g}" ]] && [[ -n "${SLURM_JOB_ID:-}" ]] && command -v squeue &>/dev/null; then
        gres=$(squeue -j "${SLURM_JOB_ID}" -h -o "%b" 2>/dev/null | head -1)
        if [[ -n "${gres}" ]] && [[ "${gres}" != "(null)" ]]; then
            g=$(echo "${gres}" | awk -F: '{print $NF}' | awk '{print $1}')
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
        echo "WARNING: Cannot detect GPUs per node, defaulting to ${GPUS_PER_NODE}." >&2
    fi
fi

MASTER_ADDR=$(head -n 1 "${HOSTFILE}")
if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()" 2>/dev/null || echo "$((29500 + RANDOM % 1000))")
fi

SLURM_RESOLVED_PARTITION=""
if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
    SLURM_RESOLVED_PARTITION="${SLURM_JOB_PARTITION}"
elif [[ -n "${PARTITION:-}" ]]; then
    SLURM_RESOLVED_PARTITION="${PARTITION}"
elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol &>/dev/null; then
    SLURM_RESOLVED_PARTITION=$(
        scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
            | awk 'match($0, /Partition=[^[:space:]]+/) {p=substr($0, RSTART+10, RLENGTH-10); split(p, a, ","); print a[1]; exit}'
    )
fi

echo "============================================================================"
echo "DLM 7B on Slurm allocation"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Nodes: ${NUM_NODES}"
echo "  Node list: ${SLURM_JOB_NODELIST}"
echo "  GPUs per node: ${GPUS_PER_NODE}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Profiling mode: ${PROFILING_MODE}"
echo "  Model tag: ${DL_MODEL_FAMILY}  TP=${TP_VAL} PP=${PP_VAL} recompute=${RECOMP_VAL} seq=${SEQ_VAL} mbs=${MBS_VAL}"
if [[ -n "${PROF_BASE:-}" ]]; then
    echo "  Profiling artifact root: ${PROF_BASE}"
fi
echo "============================================================================"

WRAPPER_DIR="${PROJECT_DIR}/.dlm_tmp"
mkdir -p "${WRAPPER_DIR}"
WRAPPER_SCRIPT="${WRAPPER_DIR}/dlm7b_wrapper_$$_${SLURM_JOB_ID}.sh"

cat > "${WRAPPER_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail

CONDA_PATH="${CONDA_PATH:-}"
CONDA_ENV="${CONDA_ENV}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE}"

if [[ "\${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
    source "\${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "\${CONDA_ENV}"
fi

export PROJECT_DIR="${PROJECT_DIR}"
export CKPT_DIR="${CKPT_DIR}"
export DATASETS_DIR="${DATASETS_DIR}"
export GPUS_PER_NODE="${GPUS_PER_NODE}"
export NUM_NODES="${NUM_NODES}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export NODE_RANK="\${SLURM_PROCID:-\${SLURM_NODEID:-0}}"

# Optional passthrough knobs for 7B tuning
export MODEL_PARALLEL_SIZE="${MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export SEQ_LENGTH="${SEQ_VAL}"
export RECOMPUTE_ACTIVATIONS="${RECOMP_VAL}"
export DL_MODEL_FAMILY="${DL_MODEL_FAMILY}"
export PROFILING_MODE="${PROFILING_MODE}"
export PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
export PROFILE_STEP_END="${PROFILE_STEP_END:-2}"
export PROFILE_RANKS="${PROFILE_RANKS:-0}"
export DISABLE_CHECKPOINT_SAVE="${DISABLE_CHECKPOINT_SAVE:-1}"
export TOKENIZER_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
export TOKENIZER="${TOKENIZER:-${DATASETS_DIR}/tokenizer.model}"

${WRAPPER_PROFILE_EXPORTS}
if [[ "${PROFILING_MODE}" == "scaletune" || "${PROFILING_MODE}" == "both" ]]; then
    export SCALETUNE_HIERARCHICAL_ROOFLINE=1
    export MEGATRON_ScaleTune_ROOFLINE=1
    export MEGATRON_COMM_DIM_PROFILE=1
    export MEGATRON_DETAILED_COMM_LOG="${MEGATRON_DETAILED_COMM_LOG:-1}"
fi

cd "${PROJECT_DIR}"
if [[ -n "${CONVERT_CHECKPOINT_ONLY}" ]]; then
    if [[ -n "${CONVERT_CKPT_STEP}" ]]; then
        bash "${DLM_SCRIPT}" "${CONVERT_CHECKPOINT_ONLY}" "${CONVERT_CKPT_STEP}"
    else
        bash "${DLM_SCRIPT}" "${CONVERT_CHECKPOINT_ONLY}"
    fi
else
    bash "${DLM_SCRIPT}"
fi
EOF

chmod +x "${WRAPPER_SCRIPT}"
cleanup() {
    rm -f "${HOSTFILE}" "${WRAPPER_SCRIPT}"
}
trap cleanup EXIT

if ! command -v srun &>/dev/null; then
    echo "ERROR: srun not found. This launcher requires Slurm srun." >&2
    exit 2
fi

LAUNCH_CMD="srun"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCH_CMD+=" --jobid=${SLURM_JOB_ID}"
fi
if [[ -n "${SLURM_RESOLVED_PARTITION}" ]]; then
    LAUNCH_CMD+=" --partition=${SLURM_RESOLVED_PARTITION}"
fi
LAUNCH_CMD+=" --ntasks=${NUM_NODES}"
LAUNCH_CMD+=" --ntasks-per-node=1"
LAUNCH_CMD+=" --gpus-per-node=${GPUS_PER_NODE}"

echo "Launch command: ${LAUNCH_CMD}"
echo ""
eval "${LAUNCH_CMD}" bash "${WRAPPER_SCRIPT}"

echo "DLM 7B run finished."
