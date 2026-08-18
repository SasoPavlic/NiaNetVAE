#!/bin/bash
#SBATCH --job-name=nianetvae-study
#SBATCH --output=/d/hpc/home/sasop/outputs/nianetvae-%x-%j.out
#SBATCH --error=/d/hpc/home/sasop/outputs/nianetvae-%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --exclude=gwn04,wn211

set -euo pipefail

JOB_MODE="${JOB_MODE:?JOB_MODE is required}"
CONFIG_PATH="${CONFIG_PATH:-configs/metropt_study.yaml}"
WORKFLOW_ID="${WORKFLOW_ID:-}"
CYCLE_ID="${CYCLE_ID:-}"
DONOR_STUDY_ROOT="${DONOR_STUDY_ROOT:-}"
DONOR_SEARCH_RUNTIME_FINGERPRINT="${DONOR_SEARCH_RUNTIME_FINGERPRINT:-}"
IMAGE_PATH="${IMAGE_PATH:-/d/hpc/home/sasop/images/nianet-vaepymoo-current.sif}"
REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(pwd)}"

mkdir -p /d/hpc/home/sasop/outputs "${REPOSITORY_ROOT}/artifacts"

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Missing container image: ${IMAGE_PATH}" >&2
    exit 1
fi
if [ ! -f "${REPOSITORY_ROOT}/${CONFIG_PATH}" ]; then
    echo "Missing study configuration: ${REPOSITORY_ROOT}/${CONFIG_PATH}" >&2
    exit 1
fi

BIND_MOUNTS="${REPOSITORY_ROOT}/data:/app/data:ro,${REPOSITORY_ROOT}/configs:/app/configs:ro,${REPOSITORY_ROOT}/artifacts:/app/artifacts"
CLI=(python -m nianetvae.cli --config "${CONFIG_PATH}")

echo "RUN_START job_id=${SLURM_JOB_ID} mode=${JOB_MODE} workflow=${WORKFLOW_ID:-n/a} cycle=${CYCLE_ID:-n/a}"
srun nvidia-smi || true

case "${JOB_MODE}" in
    prepare|search|compare|validate-study)
        COMMAND=("${CLI[@]}" "${JOB_MODE}")
        ;;
    migrate-search)
        [ -n "${DONOR_STUDY_ROOT}" ] || { echo "DONOR_STUDY_ROOT is required" >&2; exit 1; }
        [ -n "${DONOR_SEARCH_RUNTIME_FINGERPRINT}" ] || {
            echo "DONOR_SEARCH_RUNTIME_FINGERPRINT is required" >&2
            exit 1
        }
        COMMAND=(
            "${CLI[@]}" migrate-search
            --from-study-root "${DONOR_STUDY_ROOT}"
            --donor-search-runtime-fingerprint "${DONOR_SEARCH_RUNTIME_FINGERPRINT}"
        )
        ;;
    workflow)
        [ -n "${WORKFLOW_ID}" ] || { echo "WORKFLOW_ID is required" >&2; exit 1; }
        COMMAND=("${CLI[@]}" run --workflow "${WORKFLOW_ID}")
        ;;
    cycle)
        [ -n "${WORKFLOW_ID}" ] || { echo "WORKFLOW_ID is required" >&2; exit 1; }
        [ -n "${CYCLE_ID}" ] || { echo "CYCLE_ID is required" >&2; exit 1; }
        COMMAND=("${CLI[@]}" run --workflow "${WORKFLOW_ID}" --cycle-id "${CYCLE_ID}")
        ;;
    finalize)
        [ -n "${WORKFLOW_ID}" ] || { echo "WORKFLOW_ID is required" >&2; exit 1; }
        COMMAND=("${CLI[@]}" finalize --workflow "${WORKFLOW_ID}")
        ;;
    *)
        echo "Unsupported JOB_MODE=${JOB_MODE}" >&2
        exit 1
        ;;
esac

singularity exec --nv -e --pwd /app -B "${BIND_MOUNTS}" "${IMAGE_PATH}" "${COMMAND[@]}"
echo "RUN_END job_id=${SLURM_JOB_ID} mode=${JOB_MODE}"
