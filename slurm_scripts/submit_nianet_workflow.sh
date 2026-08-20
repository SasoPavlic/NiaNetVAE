#!/bin/bash
# Phase 2 of a fresh controlled study: the sequential NiaNetVAE workflow and the
# study evidence that depends on it.
#
# Run this only after the architecture search has been reviewed and accepted.
# The first cycle creates the workflow manifest, after which the engine refuses
# to extend the search budget, so this step freezes the architecture for good.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_study_job.sbatch"
CONFIG_PATH="${CONFIG_PATH:-configs/metropt_study_v5.yaml}"
IMAGE_PATH="${IMAGE_PATH:?IMAGE_PATH must identify the immutable target SIF}"
FIRST_CYCLE_WALLTIME="${FIRST_CYCLE_WALLTIME:-08:00:00}"
CYCLE_WALLTIME="${CYCLE_WALLTIME:-04:00:00}"
FINAL_CYCLE="${FINAL_CYCLE:-21}"

for command in sbatch singularity; do
    command -v "${command}" >/dev/null || { echo "Missing command: ${command}" >&2; exit 1; }
done
[ -f "${WORKER}" ] || { echo "Missing worker: ${WORKER}" >&2; exit 1; }
[ -f "${CONFIG_PATH}" ] || { echo "Missing config: ${CONFIG_PATH}" >&2; exit 1; }
[ -f "${IMAGE_PATH}" ] || { echo "Missing target image: ${IMAGE_PATH}" >&2; exit 1; }

STUDY_ID=$(sed -n 's/^[[:space:]]*study_id:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}[[:space:]]*$/\1/p' \
    "${CONFIG_PATH}" | head -1)
[ -n "${STUDY_ID}" ] || { echo "Could not read artifacts.study_id from ${CONFIG_PATH}" >&2; exit 1; }
JOB_PREFIX="${JOB_PREFIX:-nianet-${STUDY_ID##*_}}"

# The workflow reads the selected architecture, so the search must have finished.
SEARCH_MANIFEST="artifacts/${STUDY_ID}/search/search_manifest.json"
SELECTED="artifacts/${STUDY_ID}/search/selected_architecture.json"
[ -f "${SEARCH_MANIFEST}" ] || { echo "Missing search manifest: ${SEARCH_MANIFEST}" >&2; exit 1; }
[ -f "${SELECTED}" ] || { echo "Missing selected architecture: ${SELECTED}" >&2; exit 1; }
grep -q '"status": "completed"' "${SEARCH_MANIFEST}" || {
    echo "Architecture search for ${STUDY_ID} is not completed. Refusing to freeze it." >&2
    exit 1
}

mkdir -p artifacts outputs logs /d/hpc/home/sasop/outputs

# The comparison consumes all five workflows. The four baselines completed in
# phase 1, possibly days ago, so verify them now rather than discovering a gap
# after twenty-two sequential cycles have already run.
for baseline in iforest_static iforest_per_maintenance sae_static vae_static; do
    manifest="artifacts/${STUDY_ID}/workflows/${baseline}/run_manifest.json"
    [ -f "${manifest}" ] || { echo "Missing baseline workflow: ${manifest}" >&2; exit 1; }
    grep -q '"status": "completed"' "${manifest}" || {
        echo "Baseline ${baseline} is not completed. Re-run it before phase 2." >&2
        exit 1
    }
done


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

previous=""
cycle_jobs=()
for cycle_id in $(seq 0 "${FINAL_CYCLE}"); do
    if [ "${cycle_id}" -eq 0 ]; then
        walltime="${FIRST_CYCLE_WALLTIME}"
    else
        walltime="${CYCLE_WALLTIME}"
    fi
    job=$(submit "${JOB_PREFIX}-cycle-${cycle_id}" "${walltime}" "${previous}" \
        "JOB_MODE=cycle,WORKFLOW_ID=nianetvae_per_maintenance,CYCLE_ID=${cycle_id}")
    cycle_jobs+=("${job}")
    previous="${job}"
done

finalize_job=$(submit "${JOB_PREFIX}-finalize" "02:00:00" "${previous}" \
    "JOB_MODE=finalize,WORKFLOW_ID=nianetvae_per_maintenance")
comparison_job=$(submit "${JOB_PREFIX}-compare" "02:00:00" "${finalize_job}" "JOB_MODE=compare")
validation_job=$(submit "${JOB_PREFIX}-validate" "01:00:00" "${comparison_job}" \
    "JOB_MODE=validate-study")

echo "Submitted ${STUDY_ID} phase 2 (NiaNetVAE workflow and evidence):"
echo "  cycles=$(IFS=:; echo "${cycle_jobs[*]}")"
echo "  finalize=${finalize_job}"
echo "  comparison=${comparison_job}"
echo "  validation=${validation_job}"
echo
echo "The architecture search for ${STUDY_ID} is now frozen."
echo "Monitor with: squeue --me"
