#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_per_maint_cycles.sbatch"
MANIFEST_SCRIPT="${SCRIPT_DIR}/build_cycle_manifest.sbatch"
IMAGE_DIR="/d/hpc/home/sasop/images"
IMAGE_REF="docker://spartan300/nianet:vaepymoo"
IMAGE_LATEST="${IMAGE_DIR}/nianet-vaepymoo-latest.sif"
IMAGE_CURRENT="${IMAGE_DIR}/nianet-vaepymoo-current.sif"
START_CYCLE="${START_CYCLE:-0}"
END_CYCLE="${END_CYCLE:-21}"
RESUME_FROM="${RESUME_FROM:-auto}"
DATASET_NAME="${DATASET_NAME:-MetroPT}"
EXPORT_ROOT="${EXPORT_ROOT:-logs/per_maint_models}"
DETACHED_SUBMIT="${DETACHED_SUBMIT:-0}"

# Optional detached mode:
#   ./submit_per_maint_pipeline.sh --detach
# This re-launches the script under nohup so SSH disconnects do not stop submission.
if [ "${1:-}" = "--detach" ] && [ "${DETACHED_SUBMIT}" != "1" ]; then
    mkdir -p outputs
    ts=$(date +%Y%m%d_%H%M%S)
    detach_log="outputs/submit_per_maint_pipeline_${ts}.log"
    nohup env DETACHED_SUBMIT=1 bash "$0" > "${detach_log}" 2>&1 < /dev/null &
    detach_pid=$!
    echo "Detached submission started."
    echo "  pid=${detach_pid}"
    echo "  log=${detach_log}"
    echo "Track progress with: tail -f ${detach_log}"
    exit 0
fi

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Missing ${TRAIN_SCRIPT}"
    exit 1
fi

if [ ! -f "${MANIFEST_SCRIPT}" ]; then
    echo "Missing ${MANIFEST_SCRIPT}"
    exit 1
fi

if ! command -v singularity >/dev/null 2>&1; then
    echo "singularity command not found in PATH."
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch command not found in PATH."
    exit 1
fi

if ! [[ "${START_CYCLE}" =~ ^[0-9]+$ && "${END_CYCLE}" =~ ^[0-9]+$ ]]; then
    echo "START_CYCLE and END_CYCLE must be integers. Received START_CYCLE=${START_CYCLE}, END_CYCLE=${END_CYCLE}"
    exit 1
fi

if [ "${START_CYCLE}" -gt "${END_CYCLE}" ]; then
    echo "Invalid cycle range: START_CYCLE=${START_CYCLE} > END_CYCLE=${END_CYCLE}"
    exit 1
fi

echo "Syncing image from ${IMAGE_REF} ..."
mkdir -p "${IMAGE_DIR}"
singularity pull --force "${IMAGE_LATEST}" "${IMAGE_REF}"
ln -sfn "${IMAGE_LATEST}" "${IMAGE_CURRENT}"
echo "Active image symlink: ${IMAGE_CURRENT} -> $(readlink -f "${IMAGE_CURRENT}")"

mkdir -p outputs logs
if [ ! -d "configs" ]; then
    echo "Missing ./configs directory in $(pwd)"
    exit 1
fi
if [ ! -d "data" ]; then
    echo "Missing ./data directory in $(pwd)"
    exit 1
fi

cycle_complete() {
    local cycle_id="$1"
    local cycle_key
    cycle_key=$(printf "%02d" "${cycle_id}")
    local cycle_dir="${EXPORT_ROOT}/${DATASET_NAME}/cycle_${cycle_key}"
    local status_file="${cycle_dir}/cycle_status.json"

    # Trained cycle is complete when model + metadata exist.
    if [ -f "${cycle_dir}/model.pt" ] && [ -f "${cycle_dir}/model_meta.json" ]; then
        return 0
    fi

    # Non-trainable cycle is also treated as complete when skip marker exists.
    # This prevents auto-resume from re-submitting known non-trainable cycles forever.
    if [ -f "${status_file}" ] && grep -q '"status"[[:space:]]*:[[:space:]]*"skipped_non_trainable"' "${status_file}"; then
        return 0
    fi

    return 1
}

# Auto-resume: find the first incomplete cycle in the requested range.
# This keeps completed artifacts intact and starts from the first missing model/meta pair.
SUBMIT_FROM="${START_CYCLE}"
if [ "${RESUME_FROM}" = "auto" ]; then
    found_incomplete="false"
    for ((cid=START_CYCLE; cid<=END_CYCLE; cid++)); do
        if ! cycle_complete "${cid}"; then
            SUBMIT_FROM="${cid}"
            found_incomplete="true"
            break
        fi
    done
    if [ "${found_incomplete}" = "false" ]; then
        SUBMIT_FROM=$((END_CYCLE + 1))
        echo "All cycles ${START_CYCLE}-${END_CYCLE} already have model.pt + model_meta.json."
    fi
else
    if ! [[ "${RESUME_FROM}" =~ ^[0-9]+$ ]]; then
        echo "RESUME_FROM must be 'auto' or integer. Received: ${RESUME_FROM}"
        exit 1
    fi
    SUBMIT_FROM="${RESUME_FROM}"
fi

echo "Sequential submission settings:"
echo "  requested_range=${START_CYCLE}-${END_CYCLE}"
echo "  resume_from=${RESUME_FROM}"
echo "  submit_from=${SUBMIT_FROM}"
echo "  export_root=${EXPORT_ROOT}"
echo "  dataset_name=${DATASET_NAME}"

submitted_job_ids=()
prev_job_id=""

# Sequential dependency chain for Experiment A:
# cycle k starts only if cycle k-1 finished successfully (afterok).
for ((cid=SUBMIT_FROM; cid<=END_CYCLE; cid++)); do
    dep_args=()
    dep_info=""
    if [ -n "${prev_job_id}" ]; then
        dep_args+=(--dependency="afterok:${prev_job_id}")
        dep_info=" (depends on afterok:${prev_job_id})"
    fi
    job_id=$(
        sbatch --parsable \
            "${dep_args[@]}" \
            --export="ALL,CYCLE_ID=${cid}" \
            "${TRAIN_SCRIPT}"
    )
    submitted_job_ids+=("${job_id}")
    prev_job_id="${job_id}"
    echo "Submitted cycle ${cid}: ${job_id}${dep_info}"
done

if [ "${#submitted_job_ids[@]}" -gt 0 ]; then
    dependency_chain=$(IFS=:; echo "${submitted_job_ids[*]}")
    manifest_dep="afterany:${dependency_chain}"
    MANIFEST_JOB_ID=$(
        sbatch --parsable \
            --dependency="${manifest_dep}" \
            --export="ALL,CYCLE_SPEC=${START_CYCLE}-${END_CYCLE}" \
            "${MANIFEST_SCRIPT}"
    )
    echo "Submitted training jobs: ${dependency_chain}"
    echo "Submitted manifest job: ${MANIFEST_JOB_ID} (dependency: ${manifest_dep})"
else
    # If everything is already complete, still rebuild manifest to confirm trained/alias/missing statuses.
    MANIFEST_JOB_ID=$(
        sbatch --parsable \
            --export="ALL,CYCLE_SPEC=${START_CYCLE}-${END_CYCLE}" \
            "${MANIFEST_SCRIPT}"
    )
    echo "No cycle jobs submitted. Submitted manifest-only job: ${MANIFEST_JOB_ID}"
fi

echo "Check status with: squeue --me"
