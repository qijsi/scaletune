#!/bin/bash
#SBATCH --job-name=scaletune_llama7b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --partition=a01
#SBATCH --output=slurm_logs/scaletune_llama7b_%j.out
#SBATCH --error=slurm_logs/scaletune_llama7b_%j.err

# ScaleTune LLaMA 7B profiling with Slurm + mpirun
# Submit with: sbatch examples/scaletune/profile_llama_7b_slurm.sh

set -euo pipefail

# Create log directory
mkdir -p slurm_logs

# Load OpenMPI module (required for mpirun)
module load mpi/openmpi-4.1.4-gcc12-cuda12.6 2>/dev/null || \
module load openmpi 2>/dev/null || \
echo "Warning: Could not load OpenMPI module"

# Verify mpirun is available
if ! command -v mpirun &> /dev/null; then
    echo "ERROR: mpirun not found"
    echo "  Try: module load mpi/openmpi-4.1.4-gcc12-cuda12.6"
    exit 2
fi

echo "Using mpirun: $(which mpirun)"
echo "  $(mpirun --version 2>&1 | head -1)"

# Activate conda
# Use absolute path and source (not dot) for better compatibility with Slurm
CONDA_PATH="/home/fit/zhaijdzq/WORK/miniconda3"
if [[ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate aiperf_llm
else
    echo "ERROR: conda.sh not found at ${CONDA_PATH}/etc/profile.d/conda.sh" >&2
    echo "  Check CONDA_PATH or conda installation" >&2
    exit 2
fi

# Set Slurm environment variables
export SLURM_JOB_ID="${SLURM_JOB_ID:-$$}"
export RUN_MODE=slurm_mpirun

# Optional: customize configuration
# export NUM_LAYERS=16
# export HIDDEN_SIZE=2048
# export SEQ_LEN=512
# export USE_MOCK_DATA=1

echo "Starting ScaleTune profiling with Slurm job ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Node list: ${SLURM_JOB_NODELIST}"

# Run the profiling script
bash examples/scaletune/profile_llama_7b_torchrun.sh

echo "Profiling complete"
