#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_study_job.sbatch"
CONFIG_PATH="${CONFIG_PATH:-configs/metropt_study.yaml}"
IMAGE_PATH="${IMAGE_PATH:-/d/hpc/home/sasop/images/nianet-vaepymoo-current.sif}"
IMAGE_SYNC="${IMAGE_SYNC:-0}"
IMAGE_REF="${IMAGE_REF:-docker://spartan300/nianet:vaepymoo}"
IMAGE_LATEST="/d/hpc/home/sasop/images/nianet-vaepymoo-latest.sif"

for command in sbatch singularity; do
    command -v "${command}" >/dev/null || { echo "Missing command: ${command}" >&2; exit 1; }
done
[ -f "${WORKER}" ] || { echo "Missing worker: ${WORKER}" >&2; exit 1; }
[ -f "${CONFIG_PATH}" ] || { echo "Missing config: ${CONFIG_PATH}" >&2; exit 1; }
mkdir -p artifacts outputs logs /d/hpc/home/sasop/images /d/hpc/home/sasop/outputs

if [ "${IMAGE_SYNC}" = "1" ]; then
    singularity pull --force "${IMAGE_LATEST}" "${IMAGE_REF}"
    ln -sfn "${IMAGE_LATEST}" "${IMAGE_PATH}"
elif [ "${IMAGE_SYNC}" != "0" ]; then
    echo "IMAGE_SYNC must be 0 or 1" >&2
    exit 1
fi
[ -f "${IMAGE_PATH}" ] || { echo "Missing active image: ${IMAGE_PATH}" >&2; exit 1; }

submit() {
    local name="$1"
    local walltime="$2"
    local dependency="$3"
    shift 3
    local args=(--parsable --job-name="${name}" --time="${walltime}")
    if [ -n "${dependency}" ]; then
        args+=(--dependency="afterok:${dependency}" --kill-on-invalid-dep=yes)
    fi
    sbatch "${args[@]}" \
        --export="ALL,CONFIG_PATH=${CONFIG_PATH},IMAGE_PATH=${IMAGE_PATH},$*" \
        "${WORKER}"
}

prepare_job=$(submit "nianet-prepare" "04:00:00" "" "JOB_MODE=prepare")
search_job=$(submit "nianet-search" "3-10:00:00" "${prepare_job}" "JOB_MODE=search")

iforest_static_job=$(submit "nianet-if-static" "08:00:00" "${prepare_job}" \
    "JOB_MODE=workflow,WORKFLOW_ID=iforest_static")
iforest_per_job=$(submit "nianet-if-per" "12:00:00" "${prepare_job}" \
    "JOB_MODE=workflow,WORKFLOW_ID=iforest_per_maintenance")
sae_job=$(submit "nianet-sae" "2-00:00:00" "${prepare_job}" \
    "JOB_MODE=workflow,WORKFLOW_ID=sae_static")
vae_job=$(submit "nianet-vae" "2-00:00:00" "${prepare_job}" \
    "JOB_MODE=workflow,WORKFLOW_ID=vae_static")

previous="${search_job}"
nianet_jobs=()
for cycle_id in $(seq 0 21); do
    if [ "${cycle_id}" -eq 0 ]; then
        walltime="2-00:00:00"
    else
        walltime="08:00:00"
    fi
    job=$(submit "nianet-cycle-${cycle_id}" "${walltime}" "${previous}" \
        "JOB_MODE=cycle,WORKFLOW_ID=nianetvae_per_maintenance,CYCLE_ID=${cycle_id}")
    nianet_jobs+=("${job}")
    previous="${job}"
done
nianet_finalize_job=$(submit "nianet-finalize" "01:00:00" "${previous}" \
    "JOB_MODE=finalize,WORKFLOW_ID=nianetvae_per_maintenance")

all_workflows="${iforest_static_job}:${iforest_per_job}:${sae_job}:${vae_job}:${nianet_finalize_job}"
comparison_job=$(submit "nianet-compare" "01:00:00" "${all_workflows}" "JOB_MODE=compare")
validation_job=$(submit "nianet-validate" "01:00:00" "${comparison_job}" "JOB_MODE=validate-study")

echo "Submitted controlled study:"
echo "  prepare=${prepare_job}"
echo "  search=${search_job}"
echo "  baselines=${iforest_static_job}:${iforest_per_job}:${sae_job}:${vae_job}"
echo "  nianet_cycles=$(IFS=:; echo "${nianet_jobs[*]}")"
echo "  nianet_finalize=${nianet_finalize_job}"
echo "  comparison=${comparison_job}"
echo "  validation=${validation_job}"
echo "Monitor with: squeue --me"
