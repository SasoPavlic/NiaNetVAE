#!/bin/bash
#SBATCH --job-name=nianetvae-manifest
#SBATCH --output=/d/hpc/home/sasop/outputs/nianetvae-manifest%j.out
#SBATCH --error=/d/hpc/home/sasop/outputs/nianetvae-manifest%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

OUTPUT_DIR=/d/hpc/home/sasop/outputs
IMAGE_DIR=/d/hpc/home/sasop/images
IMAGE_REF="${IMAGE_REF:-docker://spartan300/nianet:vaepymoo}"
IMAGE_LATEST="${IMAGE_DIR}/nianet-vaepymoo-latest.sif"
IMAGE_CURRENT="${IMAGE_DIR}/nianet-vaepymoo-current.sif"
IMAGE_PATH="${IMAGE_CURRENT}"
CONFIG_PATH="${CONFIG_PATH:-configs/main_config.yaml}"
CYCLE_SPEC="${CYCLE_SPEC:-0-21}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${IMAGE_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Output files will be written to ${OUTPUT_DIR}/"
echo "Image path: ${IMAGE_PATH}"
echo "Image ref: ${IMAGE_REF}"
echo "Config path: ${CONFIG_PATH}"
echo "Cycle spec: ${CYCLE_SPEC}"

# Keep manifest-only runs reproducible with latest pushed container image.
# This mirrors submit_per_maint_pipeline.sh behavior.
if ! command -v singularity >/dev/null 2>&1; then
    echo "singularity command not found in PATH."
    exit 1
fi

echo "Syncing image from ${IMAGE_REF} ..."
singularity pull --force "${IMAGE_LATEST}" "${IMAGE_REF}"
ln -sfn "${IMAGE_LATEST}" "${IMAGE_CURRENT}"
echo "Active image symlink: ${IMAGE_CURRENT} -> $(readlink -f "${IMAGE_CURRENT}")"

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Missing SIF image at ${IMAGE_PATH}"
    exit 1
fi

# Configs are mounted once at /app/configs (no nested /app/configs/configs alias).
BIND_MOUNTS="$(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs"
if [ -f "$(pwd)/.env" ]; then
    BIND_MOUNTS="${BIND_MOUNTS},$(pwd)/.env:/app/.env:ro"
    echo "Detected .env at $(pwd)/.env and mounted it to /app/.env"
else
    echo "No .env found at $(pwd)/.env. Continuing without .env mount."
fi

singularity exec \
    -e \
    --pwd /app \
    -B "${BIND_MOUNTS}" \
    "${IMAGE_PATH}" \
    python -m nianetvae.tools.generate_cycle_manifest --config "${CONFIG_PATH}" --cycles "${CYCLE_SPEC}"
