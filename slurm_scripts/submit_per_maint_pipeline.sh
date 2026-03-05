#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_SCRIPT="${SCRIPT_DIR}/train_per_maint_cycles.sbatch"
MANIFEST_SCRIPT="${SCRIPT_DIR}/build_cycle_manifest.sbatch"
IMAGE_DIR="/d/hpc/home/sasop/images"
IMAGE_REF="docker://spartan300/nianet:vaepymoo"
IMAGE_LATEST="${IMAGE_DIR}/nianet-vaepymoo-latest.sif"
IMAGE_CURRENT="${IMAGE_DIR}/nianet-vaepymoo-current.sif"

if [ ! -f "${ARRAY_SCRIPT}" ]; then
    echo "Missing ${ARRAY_SCRIPT}"
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

ARRAY_JOB_ID=$(sbatch --parsable "${ARRAY_SCRIPT}")
MANIFEST_JOB_ID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} "${MANIFEST_SCRIPT}")

echo "Submitted array job: ${ARRAY_JOB_ID}"
echo "Submitted manifest job: ${MANIFEST_JOB_ID} (dependency: afterany:${ARRAY_JOB_ID})"
echo "Check status with: squeue --me"
