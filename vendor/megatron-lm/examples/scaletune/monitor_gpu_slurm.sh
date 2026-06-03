#!/bin/bash
# Monitor GPU usage on Slurm-allocated nodes
# Usage: bash examples/scaletune/monitor_gpu_slurm.sh [interval_seconds]

set -euo pipefail

INTERVAL="${1:-1}"  # Default: 1 second

echo "============================================================"
echo "Slurm GPU Monitor"
echo "============================================================"

# Get Slurm job info
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # Try to get from squeue
    SLURM_JOB_ID=$(squeue -u "${USER}" -t R -h -o "%i" 2>/dev/null | head -1)
    if [[ -z "${SLURM_JOB_ID}" ]]; then
        echo "ERROR: No active Slurm job found!"
        echo "Please run this script within a Slurm allocation or specify SLURM_JOB_ID"
        exit 1
    fi
fi

echo "Job ID: ${SLURM_JOB_ID}"

# Get node list - try multiple methods
SLURM_JOB_NODELIST=""

# Method 1: From environment (if running within job) - highest priority
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST}"
fi

# Method 2: From scontrol show job
if [[ -z "${SLURM_JOB_NODELIST}" ]]; then
    SLURM_JOB_NODELIST=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oE 'NodeList=[^[:space:]]+' | cut -d= -f2 | head -1)
fi

# Method 3: From squeue
if [[ -z "${SLURM_JOB_NODELIST}" ]]; then
    SLURM_JOB_NODELIST=$(squeue -j "${SLURM_JOB_ID}" -h -o "%n" 2>/dev/null | head -1)
fi

# Method 4: From scontrol show hostname
if [[ -z "${SLURM_JOB_NODELIST}" ]] && [[ -n "${SLURM_JOB_ID}" ]]; then
    SLURM_JOB_NODELIST=$(scontrol show hostname -l "${SLURM_JOB_ID}" 2>/dev/null | tr ' ' ',' | head -1)
fi

# Validate node list
if [[ -z "${SLURM_JOB_NODELIST}" ]] || [[ "${SLURM_JOB_NODELIST}" == "(null)" ]]; then
    echo "WARNING: Could not get node list from Slurm for job ${SLURM_JOB_ID}"
    echo ""
    echo "Debug info:"
    echo "  SLURM_JOB_ID=${SLURM_JOB_ID}"
    echo "  SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
    echo ""
    echo "Possible causes:"
    echo "  1. Job ${SLURM_JOB_ID} has ended"
    echo "  2. You are not running within a Slurm allocation"
    echo "  3. Slurm commands are not available"
    echo ""
    echo "Solutions:"
    echo "  Option A: Run within a Slurm allocation:"
    echo "    salloc -N 1 --gpus-per-node=8 --time=01:00:00 --partition=h01"
    echo "    bash examples/scaletune/monitor_gpu_slurm.sh"
    echo ""
    echo "  Option B: Manually specify the node:"
    echo "    MONITOR_NODE=g44 bash examples/scaletune/monitor_gpu_slurm.sh"
    echo ""
    echo "  Option C: Check if job exists:"
    echo "    squeue -j ${SLURM_JOB_ID}"
    echo "    scontrol show job ${SLURM_JOB_ID}"
    echo ""
    
    # If user provided manual node, use it
    if [[ -n "${MONITOR_NODE:-}" ]]; then
        echo "Using manually specified node: ${MONITOR_NODE}"
        SLURM_JOB_NODELIST="${MONITOR_NODE}"
    else
        exit 1
    fi
fi

echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Monitoring interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo "============================================================"
echo ""

# Function to monitor a single node
monitor_node() {
    local node=$1
    echo "Monitoring node: ${node}"
    echo "Press Ctrl+C to stop"
    echo ""
    
    while true; do
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
        
        # Method 1: Direct nvidia-smi (if on the node itself)
        if command -v nvidia-smi &>/dev/null; then
            nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits
        # Method 2: SSH to remote node
        elif ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no "${node}" \
            'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits' 2>/dev/null; then
            :
        # Method 3: srun to remote node (if within Slurm allocation)
        elif command -v srun &>/dev/null && [[ -n "${SLURM_JOB_ID}" ]]; then
            srun --jobid="${SLURM_JOB_ID}" --nodes=1 --ntasks=1 --gres=gpu:all \
                --nodelist="${node}" --exclusive nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || \
            echo "Failed to get GPU info from ${node}"
        else
            echo "ERROR: Cannot access GPU info on ${node}"
            echo "  - nvidia-smi not available locally"
            echo "  - SSH failed or not configured"
            echo "  - srun not available or job invalid"
        fi
        
        echo ""
        sleep "${INTERVAL}"
    done
}

# Parse node list (handle formats like "node[001-008]" or "node001,node002")
parse_nodes() {
    local nodelist="$1"
    
    # If contains brackets (e.g., node[001-008]), expand them
    if [[ "${nodelist}" == *"["*"]"* ]]; then
        # Use scontrol to expand
        scontrol show hostname "${nodelist}" 2>/dev/null || echo "${nodelist}"
    else
        # Comma-separated list
        echo "${nodelist}" | tr ',' '\n'
    fi
}

# Get all nodes
NODES=($(parse_nodes "${SLURM_JOB_NODELIST}"))
NUM_NODES=${#NODES[@]}

echo "Total nodes: ${NUM_NODES}"
echo ""

if [[ ${NUM_NODES} -eq 0 ]]; then
    echo "ERROR: No nodes found!"
    exit 1
elif [[ ${NUM_NODES} -eq 1 ]]; then
    # Single node - monitor directly
    monitor_node "${NODES[0]}"
else
    # Multiple nodes - monitor each in sequence
    echo "Monitoring ${NUM_NODES} nodes in sequence..."
    echo ""
    
    while true; do
        for node in "${NODES[@]}"; do
            echo ""
            echo "=== $(date '+%Y-%m-%d %H:%M:%S') - Node: ${node} ==="
            
            # Try SSH to remote node
            if ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no "${node}" \
                'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader' 2>/dev/null; then
                :
            else
                # Fallback: try srun
                if command -v srun &>/dev/null && [[ -n "${SLURM_JOB_ID}" ]]; then
                    srun --jobid="${SLURM_JOB_ID}" --nodes=1 --ntasks=1 --gres=gpu:1 \
                        --nodelist="${node}" nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader 2>/dev/null || \
                    echo "Failed to get GPU info from ${node}"
                else
                    echo "Failed to connect to ${node}"
                fi
            fi
            
            sleep "${INTERVAL}"
        done
    done
fi
