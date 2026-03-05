#!/bin/bash
#SBATCH --job-name=nianetvae-manifest
#SBATCH --output=/d/hpc/home/sasop/outputs/nianetvae-manifest%j.out
#SBATCH --error=/d/hpc/home/sasop/outputs/nianetvae-manifest%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

OUTPUT_DIR=/d/hpc/home/sasop/outputs
IMAGE_PATH=/d/hpc/home/sasop/images/nianet-vaepymoo-current.sif
mkdir -p "${OUTPUT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Output files will be written to ${OUTPUT_DIR}/"
echo "Image path: ${IMAGE_PATH}"

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Missing SIF image at ${IMAGE_PATH}"
    exit 1
fi

# Note: main_config.yaml may reference dataset.config_file as "configs/<file>.yaml".
# Add a compatibility alias mount so both /app/configs/<file> and
# /app/configs/configs/<file> resolve to the same host directory.
BIND_MOUNTS="$(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs,$(pwd)/configs:/app/configs/configs"
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
    python -m nianetvae.tools.generate_cycle_manifest --config configs/main_config.yaml --cycles 0-21
