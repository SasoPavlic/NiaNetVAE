#!/bin/bash
# Phase 1 of a fresh controlled study: prepare, the four search-independent
# baselines, and a chained architecture-search ladder.
#
# The ladder exists because a search job cannot simply be re-run to continue.
# A job that terminates gracefully records its execution budget, and the engine
# resumes only when a later job raises that budget. Each ladder step therefore
# increases search.max_generations while keeping every controlled constant
# identical, so all steps share one study_id and one NSGA-III checkpoint.
#
# This script deliberately never submits nianetvae_per_maintenance. Starting
# that workflow permanently forbids extending the search.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_study_job.sbatch"
CONFIG_PATH="${CONFIG_PATH:-configs/metropt_study_v5.yaml}"
IMAGE_PATH="${IMAGE_PATH:?IMAGE_PATH must identify the immutable target SIF}"
SEARCH_WALLTIME="${SEARCH_WALLTIME:-4-00:00:00}"
BASELINE_WALLTIME="${BASELINE_WALLTIME:-02:00:00}"
LADDER="${LADDER:-configs/search_ladder/metropt_study_v5_gen025.yaml configs/search_ladder/metropt_study_v5_gen050.yaml configs/search_ladder/metropt_study_v5_gen075.yaml configs/metropt_study_v5.yaml}"

for command in sbatch singularity; do
    command -v "${command}" >/dev/null || { echo "Missing command: ${command}" >&2; exit 1; }
done
[ -f "${WORKER}" ] || { echo "Missing worker: ${WORKER}" >&2; exit 1; }
[ -f "${CONFIG_PATH}" ] || { echo "Missing config: ${CONFIG_PATH}" >&2; exit 1; }
[ -f "${IMAGE_PATH}" ] || { echo "Missing target image: ${IMAGE_PATH}" >&2; exit 1; }
for step_config in ${LADDER}; do
    [ -f "${step_config}" ] || { echo "Missing ladder config: ${step_config}" >&2; exit 1; }
done

read_study_id() {
    sed -n 's/^[[:space:]]*study_id:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}[[:space:]]*$/\1/p' "$1" | head -1
}

STUDY_ID=$(read_study_id "${CONFIG_PATH}")
[ -n "${STUDY_ID}" ] || { echo "Could not read artifacts.study_id from ${CONFIG_PATH}" >&2; exit 1; }
JOB_PREFIX="${JOB_PREFIX:-nianet-${STUDY_ID##*_}}"

# Every ladder step must belong to the same study, otherwise a later step would
# start a second search instead of resuming the checkpoint.
for step_config in ${LADDER}; do
    step_study=$(read_study_id "${step_config}")
    [ "${step_study}" = "${STUDY_ID}" ] || {
        echo "Ladder config ${step_config} targets ${step_study}, expected ${STUDY_ID}" >&2
        exit 1
    }
done

# Refuse to extend a search that has already been frozen by the NiaNetVAE workflow.
if [ -e "artifacts/${STUDY_ID}/workflows/nianetvae_per_maintenance/run_manifest.json" ]; then
    echo "Refusing: ${STUDY_ID} has already begun nianetvae_per_maintenance." >&2
    echo "The architecture search can no longer be extended. Use a new study_id." >&2
    exit 1
fi

mkdir -p artifacts outputs logs /d/hpc/home/sasop/outputs

submit() {
    local name="$1"
    local walltime="$2"
    local dependency="$3"
    local config="$4"
    shift 4
    local args=(--parsable --job-name="${name}" --time="${walltime}")
    if [ -n "${dependency}" ]; then
        args+=(--dependency="afterok:${dependency}" --kill-on-invalid-dep=yes)
    fi
    sbatch "${args[@]}" \
        --export="ALL,CONFIG_PATH=${config},IMAGE_PATH=${IMAGE_PATH},$*" \
        "${WORKER}"
}

prepare_job=$(submit "${JOB_PREFIX}-prepare" "01:00:00" "" "${CONFIG_PATH}" "JOB_MODE=prepare")

# The baselines never consult the searched architecture, so they run in parallel
# with the ladder and leave the search extendable.
iforest_static_job=$(submit "${JOB_PREFIX}-if-static" "${BASELINE_WALLTIME}" "${prepare_job}" \
    "${CONFIG_PATH}" "JOB_MODE=workflow,WORKFLOW_ID=iforest_static")
iforest_per_job=$(submit "${JOB_PREFIX}-if-per" "${BASELINE_WALLTIME}" "${prepare_job}" \
    "${CONFIG_PATH}" "JOB_MODE=workflow,WORKFLOW_ID=iforest_per_maintenance")
sae_job=$(submit "${JOB_PREFIX}-sae" "${BASELINE_WALLTIME}" "${prepare_job}" \
    "${CONFIG_PATH}" "JOB_MODE=workflow,WORKFLOW_ID=sae_static")
vae_job=$(submit "${JOB_PREFIX}-vae" "${BASELINE_WALLTIME}" "${prepare_job}" \
    "${CONFIG_PATH}" "JOB_MODE=workflow,WORKFLOW_ID=vae_static")

previous="${prepare_job}"
search_jobs=()
for step_config in ${LADDER}; do
    step_generations=$(sed -n 's/^[[:space:]]*max_generations:[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
        "${step_config}" | head -1)
    [ -n "${step_generations}" ] || {
        echo "Could not read search.max_generations from ${step_config}" >&2
        exit 1
    }
    job=$(submit "${JOB_PREFIX}-search-gen${step_generations}" "${SEARCH_WALLTIME}" "${previous}" \
        "${step_config}" "JOB_MODE=search")
    search_jobs+=("${job}")
    previous="${job}"
done

echo "Submitted ${STUDY_ID} phase 1 (prepare, baselines, search ladder):"
echo "  prepare=${prepare_job}"
echo "  baselines=${iforest_static_job}:${iforest_per_job}:${sae_job}:${vae_job}"
echo "  search_ladder=$(IFS=:; echo "${search_jobs[*]}")"
echo
echo "nianetvae_per_maintenance was NOT submitted, so the search stays extendable."
echo "Review the ladder, then run submit_nianet_workflow.sh for phase 2."
echo "Monitor with: squeue --me"
