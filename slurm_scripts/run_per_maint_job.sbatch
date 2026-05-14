#!/bin/bash
#SBATCH --job-name=nianetvae-metropt
#SBATCH --output=/d/hpc/home/sasop/outputs/nianetvae-%x%j.out
#SBATCH --error=/d/hpc/home/sasop/outputs/nianetvae-%x%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=10
#SBATCH --exclude=wn211

set -euo pipefail

JOB_MODE="${JOB_MODE:-train}"
CONFIG_PATH="${CONFIG_PATH:-configs/main_config.yaml}"
CYCLE_ID="${CYCLE_ID:-${SLURM_ARRAY_TASK_ID:-}}"
CYCLE_SPEC="${CYCLE_SPEC:-0-21}"
OUTPUT_DIR=/d/hpc/home/sasop/outputs
IMAGE_PATH=/d/hpc/home/sasop/images/nianet-vaepymoo-current.sif

mkdir -p "${OUTPUT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Output files will be written to ${OUTPUT_DIR}/"
echo "Job mode: ${JOB_MODE}"
echo "Image path: ${IMAGE_PATH}"
echo "Config path: ${CONFIG_PATH}"
echo "Cycle ID: ${CYCLE_ID:-n/a}"
echo "Cycle spec: ${CYCLE_SPEC}"

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Missing SIF image at ${IMAGE_PATH}"
    exit 1
fi

BIND_MOUNTS="$(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs"
if [ -f "$(pwd)/.env" ]; then
    BIND_MOUNTS="${BIND_MOUNTS},$(pwd)/.env:/app/.env:ro"
    echo "Detected .env at $(pwd)/.env and mounted it to /app/.env"
else
    echo "No .env found at $(pwd)/.env. Postgres runs will fail until .env is provided."
fi

case "${JOB_MODE}" in
    train)
        if [ -z "${CYCLE_ID}" ]; then
            echo "CYCLE_ID is required when JOB_MODE=train."
            exit 1
        fi
        srun nvidia-smi
        singularity exec --nv \
            -e \
            --pwd /app \
            -B "${BIND_MOUNTS}" \
            "${IMAGE_PATH}" \
            python main.py -c "${CONFIG_PATH}" --cycle-id "${CYCLE_ID}"
        ;;
    manifest)
        singularity exec \
            -e \
            --pwd /app \
            -B "${BIND_MOUNTS}" \
            "${IMAGE_PATH}" \
            python -m nianetvae.tools.generate_cycle_manifest --config "${CONFIG_PATH}" --cycles "${CYCLE_SPEC}"
        ;;
    *)
        echo "Unsupported JOB_MODE='${JOB_MODE}'. Expected train or manifest."
        exit 1
        ;;
esac
