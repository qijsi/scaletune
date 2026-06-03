#!/usr/bin/env bash
# Unified ScaleTune / profiling entry for AIPerf-LLM.
#
# Dispatches to the canonical scripts still maintained in each repo:
#   - Megatron-LM/examples/scaletune/run_with_salloc.sh
#   - MegaDLMs/examples/scaletune/run_with_salloc.sh
#   - Megatron-Bridge/examples/scaletune/run_with_salloc.sh
#
# Usage (from anywhere):
#   export AIPERF_ROOT=/path/to/AIPerf-LLM   # optional if this file lives there
#   bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" --framework megadlms -m llama -p torch -b 2
#   bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" -f megatron-lm -m qwen2-7b --micro-batch-size 4
#   bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" -f megatron-lm -m qwen2-7b -p instrument
#   bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" -f megatron-lm -m qwen3-32b
#   bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" -f megatron-lm -m qwen3-30b-a3b
#   bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" -f megatron-bridge -m nemotron3-nano30b --model-size tiny
#   bash "${AIPERF_ROOT}/scaletune/run_with_salloc.sh" -f megatron-lm -m nemotron3-nano30b
#
# Copies of helper scripts live under scaletune/vendor/ for reference; the active
# implementations stay in Megatron-LM / MegaDLMs so training paths and PYTHONPATH stay correct.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIPERF_ROOT="${AIPERF_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

MEGATRON_ROOT="${MEGATRON_ROOT:-${AIPERF_ROOT}/Megatron-LM}"
MEGADLMS_ROOT="${MEGADLMS_ROOT:-${AIPERF_ROOT}/MegaDLMs}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${AIPERF_ROOT}/Megatron-Bridge}"

usage() {
    cat <<EOF
Usage: $(basename "$0") --framework {megatron-lm|megadlms|megatron-bridge} [options passed to the repo script...]

Options (handled by this launcher only):
  --framework NAME, -f NAME   Required unless AIPERF_FRAMEWORK is set: megatron-lm | megadlms | megatron-bridge
                              (aliases: --stack, AIPERF_STACK for backward compatibility)
  --profiling-mode M, -p M    Forwarded as --profiling-mode M to the repo script
                              Megatron-LM supports: torch | scaletune | both | instrument | none
  --memory-benefit 0|1        Forwarded to Megatron scripts (theory + saved-tensor JSON)
  --measured-vram 0|1         Forwarded (CUDA peak allocator + tensor/ZeRO-visible buckets)
  -b N, --micro-batch-size N  Sets MICRO_BS for the repo script (micro-batch-size)
  --model-size SIZE, -s SIZE  Megatron-Bridge only: full (default) | tiny (7-layer Nemotron-3 Nano)
  -h, --help                  This help

Environment:
  AIPERF_ROOT                 Default: parent of this script (AIPerf-LLM)
  MEGATRON_ROOT               Default: \$AIPERF_ROOT/Megatron-LM
  MEGADLMS_ROOT               Default: \$AIPERF_ROOT/MegaDLMs
  MEGATRON_BRIDGE_ROOT        Default: \$AIPERF_ROOT/Megatron-Bridge
  AIPERF_FRAMEWORK            Same as --framework if no CLI flag
  PROFILE_OUTPUT_ROOT         Default: \$AIPERF_ROOT/scaletune_runs (MegaDLMs script honors this)
  PROFILE_FRAMEWORK_TAG       Optional override for profiling dir slug (defaults: megatron | megadlms | megatron-bridge)
  MICRO_BS                    Micro batch size (same as -b if not using CLI)
  BRIDGE_USE_UV               megatron-bridge: 1 (default, use .venv) | 0 for conda/system python
  BRIDGE_VENV_LAUNCHER        megatron-bridge: auto (default) | venv-python | uv-run
  UV_BIN                      megatron-bridge: path to uv (optional)

Vendor copies (read-only reference): ${SCRIPT_DIR}/vendor/
Shared analysis tool: ${SCRIPT_DIR}/tools/analyze_nvtx_comm_time.py
EOF
}

# AIPERF_STACK kept as deprecated alias for AIPERF_FRAMEWORK
FRAMEWORK="${AIPERF_FRAMEWORK:-${AIPERF_STACK:-}}"
PASS=()
AIPERF_MICRO_BS_CLI=""
MODEL_FAMILY=""
MODEL_SIZE="${MODEL_SIZE:-full}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework|--stack|-f)
            FRAMEWORK="${2:-}"
            if [[ -z "${FRAMEWORK}" || "${FRAMEWORK}" == -* ]]; then
                echo "ERROR: $1 requires a value (megatron-lm, megadlms, or megatron-bridge)" >&2
                exit 2
            fi
            shift 2
            ;;
        --framework=*|--stack=*)
            FRAMEWORK="${1#*=}"
            shift
            ;;
        -f[!-]*)
            FRAMEWORK="${1#-f}"
            shift
            ;;
        --model|-m)
            MODEL_FAMILY="${2:-}"
            if [[ -z "${MODEL_FAMILY}" ]]; then
                echo "ERROR: $1 requires a value (e.g. nemotron3-nano30b | nemotron3-super)" >&2
                exit 2
            fi
            shift 2
            ;;
        --model=*)
            MODEL_FAMILY="${1#*=}"
            shift
            ;;
        -m[!-]*)
            MODEL_FAMILY="${1#-m}"
            shift
            ;;
        --model-size=*)
            MODEL_SIZE="${1#*=}"
            shift
            ;;
        --model-size|-s)
            ms="${2:-}"
            if [[ -z "${ms}" || "${ms}" == -* ]]; then
                echo "ERROR: $1 requires a value (full | tiny)" >&2
                exit 2
            fi
            MODEL_SIZE="${ms}"
            shift 2
            ;;
        -s[!-]*)
            MODEL_SIZE="${1#-s}"
            shift
            ;;
        --profiling-mode=*)
            PASS+=(--profiling-mode "${1#*=}")
            shift
            ;;
        -p[!-]*)
            PASS+=(--profiling-mode "${1#-p}")
            shift
            ;;
        --profiling-mode|-p)
            mode="${2:-}"
            if [[ -z "${mode}" || "${mode}" == -* ]]; then
                echo "ERROR: $1 requires a value (e.g. scaletune | torch | both | instrument | none)" >&2
                exit 2
            fi
            PASS+=(--profiling-mode "${mode}")
            shift 2
            ;;
        --memory-benefit=*)
            PASS+=(--memory-benefit "${1#*=}")
            shift
            ;;
        --measured-vram=*)
            PASS+=(--measured-vram "${1#*=}")
            shift
            ;;
        --measured-vram)
            mbv="${2:-1}"
            if [[ -z "${mbv}" || "${mbv}" == -* ]]; then
                echo "ERROR: $1 requires 0 or 1" >&2
                exit 2
            fi
            PASS+=(--measured-vram "${mbv}")
            shift 2
            ;;
        --no-measured-vram)
            PASS+=(--measured-vram 0)
            shift
            ;;
        --memory-benefit)
            mb="${2:-1}"
            if [[ -z "${mb}" || "${mb}" == -* ]]; then
                echo "ERROR: $1 requires 0 or 1" >&2
                exit 2
            fi
            PASS+=(--memory-benefit "${mb}")
            shift 2
            ;;
        --no-memory-benefit)
            PASS+=(--memory-benefit 0)
            shift
            ;;
        --micro-batch-size=*)
            AIPERF_MICRO_BS_CLI="${1#*=}"
            shift
            ;;
        -b[!-]*)
            AIPERF_MICRO_BS_CLI="${1#-b}"
            shift
            ;;
        -b|--micro-batch-size)
            mbs="${2:-}"
            if [[ -z "${mbs}" || "${mbs}" == -* ]]; then
                echo "ERROR: $1 requires a positive integer (micro batch size)" >&2
                exit 2
            fi
            AIPERF_MICRO_BS_CLI="${mbs}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PASS+=("$1")
            shift
            ;;
    esac
done

FRAMEWORK="${FRAMEWORK:-}"
if [[ -z "${FRAMEWORK}" ]]; then
    echo "ERROR: set --framework megatron-lm|megadlms|megatron-bridge (or -f), or export AIPERF_FRAMEWORK" >&2
    usage >&2
    exit 2
fi

if [[ -n "${AIPERF_MICRO_BS_CLI}" ]]; then
    if ! [[ "${AIPERF_MICRO_BS_CLI}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: micro batch size must be a positive integer, got: ${AIPERF_MICRO_BS_CLI}" >&2
        exit 2
    fi
    export MICRO_BS="${AIPERF_MICRO_BS_CLI}"
fi

export PROFILE_OUTPUT_ROOT="${PROFILE_OUTPUT_ROOT:-${AIPERF_ROOT}/scaletune_runs}"
mkdir -p "${PROFILE_OUTPUT_ROOT}"

case "${FRAMEWORK}" in
    megatron-lm|megatron)
        if [[ ! -d "${MEGATRON_ROOT}" ]]; then
            echo "ERROR: MEGATRON_ROOT not found: ${MEGATRON_ROOT}" >&2
            exit 2
        fi
        if [[ ! -f "${MEGATRON_ROOT}/examples/scaletune/run_with_salloc.sh" ]]; then
            echo "ERROR: missing ${MEGATRON_ROOT}/examples/scaletune/run_with_salloc.sh" >&2
            exit 2
        fi
        echo ">>> AIPerf unified launcher: framework=megatron-lm  repo=${MEGATRON_ROOT}"
        export PROFILE_FRAMEWORK_TAG="${PROFILE_FRAMEWORK_TAG:-megatron}"
        # Forward --model to downstream Megatron-LM script if provided
        if [[ -n "${MODEL_FAMILY}" ]]; then
            PASS=(--model "${MODEL_FAMILY}" "${PASS[@]}")
        fi
        cd "${MEGATRON_ROOT}"
        exec bash examples/scaletune/run_with_salloc.sh "${PASS[@]}"
        ;;
    megadlms|mega|dlm)
        if [[ ! -d "${MEGADLMS_ROOT}" ]]; then
            echo "ERROR: MEGADLMS_ROOT not found: ${MEGADLMS_ROOT}" >&2
            exit 2
        fi
        if [[ ! -f "${MEGADLMS_ROOT}/examples/scaletune/run_with_salloc.sh" ]]; then
            echo "ERROR: missing ${MEGADLMS_ROOT}/examples/scaletune/run_with_salloc.sh" >&2
            exit 2
        fi
        export MEGATRON_ROOT
        echo ">>> AIPerf unified launcher: framework=megadlms  repo=${MEGADLMS_ROOT}  MEGATRON_ROOT=${MEGATRON_ROOT}"
        export PROFILE_FRAMEWORK_TAG="${PROFILE_FRAMEWORK_TAG:-megadlms}"
        # Forward --model to downstream MegaDLMs script if provided
        if [[ -n "${MODEL_FAMILY}" ]]; then
            PASS=(--model "${MODEL_FAMILY}" "${PASS[@]}")
        fi
        cd "${MEGADLMS_ROOT}"
        exec bash examples/scaletune/run_with_salloc.sh "${PASS[@]}"
        ;;
    megatron-bridge|bridge|mbridge)
        if [[ ! -d "${MEGATRON_BRIDGE_ROOT}" ]]; then
            echo "ERROR: MEGATRON_BRIDGE_ROOT not found: ${MEGATRON_BRIDGE_ROOT}" >&2
            exit 2
        fi
        if [[ ! -f "${MEGATRON_BRIDGE_ROOT}/examples/scaletune/run_with_salloc.sh" ]]; then
            echo "ERROR: missing ${MEGATRON_BRIDGE_ROOT}/examples/scaletune/run_with_salloc.sh" >&2
            exit 2
        fi
        echo ">>> AIPerf unified launcher: framework=megatron-bridge  repo=${MEGATRON_BRIDGE_ROOT}"
        export PROFILE_FRAMEWORK_TAG="${PROFILE_FRAMEWORK_TAG:-megatron-bridge}"
        export AIPERF_ROOT
        export MODEL_SIZE
        export MICRO_BS
        export BRIDGE_USE_UV="${BRIDGE_USE_UV:-1}"
        export BRIDGE_VENV_LAUNCHER="${BRIDGE_VENV_LAUNCHER:-auto}"
        export UV_BIN="${UV_BIN:-}"
        export BRIDGE_PYTHON="${BRIDGE_PYTHON:-}"
        export CONDA_ENV="${CONDA_ENV:-}"
        export UV_OFFLINE="${UV_OFFLINE:-1}"
        BRIDGE_PASS=()
        if [[ -n "${MODEL_FAMILY}" ]]; then
            BRIDGE_PASS=(--model "${MODEL_FAMILY}")
        fi
        if [[ -n "${MODEL_SIZE}" && "${MODEL_SIZE}" != "full" ]]; then
            BRIDGE_PASS+=(--model-size "${MODEL_SIZE}")
        fi
        cd "${MEGATRON_BRIDGE_ROOT}"
        exec bash examples/scaletune/run_with_salloc.sh "${BRIDGE_PASS[@]}" "${PASS[@]}"
        ;;
    *)
        echo "ERROR: unknown framework '${FRAMEWORK}' (use megatron-lm, megadlms, or megatron-bridge)" >&2
        exit 2
        ;;
esac
