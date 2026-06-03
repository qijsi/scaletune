#!/usr/bin/env bash
# Run nvidia-smi on every node in a Slurm job allocation (from the login node).
#
# Resolves the job ID from: CLI argument, SLURM_JOB_ID, or the first RUNNING job
# for the current user. Uses srun --jobid so SSH to compute nodes is not required.
#
# Usage:
#   bash scaletune/tools/slurm_job_nvidia_smi.sh [JOBID]
#   watch -n 5 bash scaletune/tools/slurm_job_nvidia_smi.sh 12345
#
# If your site blocks concurrent job steps until you add overlap (Slurm 20.11+):
#   export SRUN_JOBID_EXTRA="--overlap"
#
set -euo pipefail

resolve_job_id() {
    if [[ -n "${1:-}" ]]; then
        echo "$1"
        return
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "${SLURM_JOB_ID}"
        return
    fi
    local jid
    jid="$(squeue -u "${USER}" -t R -h -o "%i" 2>/dev/null | head -1 || true)"
    if [[ -z "${jid}" ]]; then
        echo "ERROR: no job id (pass JOBID, run inside allocation, or have a RUNNING job in squeue)." >&2
        exit 2
    fi
    echo "${jid}"
}

job_id="$(resolve_job_id "${1:-}")"

# NumNodes can be "4" or "3-4" (min-max); use the first number for srun -N.
num_nodes_raw="$(scontrol show job "${job_id}" 2>/dev/null | awk -F= '/NumNodes=/{print $2; exit}' | tr -d ' ')"
if [[ -z "${num_nodes_raw}" ]]; then
    echo "ERROR: could not read NumNodes for job ${job_id} (wrong id or permission)." >&2
    exit 2
fi
num_nodes="${num_nodes_raw%%-*}"
if ! [[ "${num_nodes}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: unexpected NumNodes='${num_nodes_raw}' for job ${job_id}." >&2
    exit 2
fi

# Total tasks = one per node so each rank prints its host's GPUs.
srun_args=(
    --jobid="${job_id}"
    -N"${num_nodes}"
    -n"${num_nodes}"
    --ntasks-per-node=1
)

# Optional: e.g. SRUN_JOBID_EXTRA="--overlap" when the job already holds all resources.
if [[ -n "${SRUN_JOBID_EXTRA:-}" ]]; then
    read -r -a _srun_extra <<< "${SRUN_JOBID_EXTRA}"
    srun_args+=("${_srun_extra[@]}")
fi

echo "=== Job ${job_id}  NumNodes=${num_nodes_raw}  (${#srun_args[@]} srun args + launcher) ==="
echo

srun "${srun_args[@]}" bash -c 'echo "----- $(hostname) -----"; nvidia-smi; echo'
