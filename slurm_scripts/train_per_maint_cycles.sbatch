#!/bin/bash
#SBATCH --job-name=nianetvae-metropt                             # Job name
#SBATCH --output=/dev/null                                       # Standard output file
#SBATCH --error=/d/hpc/home/sasop/outputs/nianetvae-metropt%j.err # Standard error file
#SBATCH --partition=gpu                                          # GPU partition
#SBATCH --nodes=1                                                # Number of nodes
#SBATCH --ntasks=1                                               # Number of tasks
#SBATCH --gres=gpu:1                                             # Request 1 GPU
#SBATCH --mem-per-gpu=50GB                                       # Memory per GPU
#SBATCH --time=21:00:00                                          # Maximum runtime

set -euo pipefail

# Experiment A requires strict sequential order because cycle k depends on cycle k-1 artifacts.
# This script therefore trains exactly one cycle per job, provided via:
# - CYCLE_ID (preferred, passed by submit_per_maint_pipeline.sh), or
# - SLURM_ARRAY_TASK_ID (fallback for backward compatibility).
CYCLE_ID="${CYCLE_ID:-${SLURM_ARRAY_TASK_ID:-}}"
CONFIG_PATH="${CONFIG_PATH:-configs/main_config.yaml}"
METRIC_NAME="${METRIC_NAME:-SMAPE}"

if [ -z "${CYCLE_ID}" ]; then
    echo "CYCLE_ID is not set. Pass CYCLE_ID via sbatch --export=ALL,CYCLE_ID=<id>."
    exit 1
fi

# === Prepare output directory ===
OUTPUT_DIR=/d/hpc/home/sasop/outputs
IMAGE_PATH=/d/hpc/home/sasop/images/nianet-vaepymoo-current.sif
mkdir -p "${OUTPUT_DIR}"    # ensure the folder exists

# Log environment details
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Output files will be written to ${OUTPUT_DIR}/"
echo "Cycle ID: ${CYCLE_ID}"
echo "Image path: ${IMAGE_PATH}"
echo "Config path: ${CONFIG_PATH}"
echo "Metric: ${METRIC_NAME}"

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Missing SIF image at ${IMAGE_PATH}"
    exit 1
fi

# Check GPU visibility
srun nvidia-smi

# Prepare bind mounts (mount .env if present, so app can load DB credentials inside container).
# Add compatibility alias for nested config references (/app/configs/configs/*).
BIND_MOUNTS="$(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs,$(pwd)/configs:/app/configs/configs"
if [ -f "$(pwd)/.env" ]; then
    BIND_MOUNTS="${BIND_MOUNTS},$(pwd)/.env:/app/.env:ro"
    echo "Detected .env at $(pwd)/.env and mounted it to /app/.env"
else
    echo "No .env found at $(pwd)/.env. Postgres runs will fail until .env is provided."
fi

# Execute one cycle run inside the container.
singularity exec --nv \
    -e \
    --pwd /app \
    -B "${BIND_MOUNTS}" \
    "${IMAGE_PATH}" \
    python main.py -c "${CONFIG_PATH}" -met "${METRIC_NAME}" --cycle-id "${CYCLE_ID}"
