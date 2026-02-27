#!/bin/bash
#SBATCH --job-name=nianetvae-metropt                             # Job name
#SBATCH --output=/dev/null                                       # Standard output file
#SBATCH --error=/d/hpc/home/sasop/outputs/nianetvae-metropt%j.err # Standard error file
#SBATCH --partition=gpu                                          # GPU partition
#SBATCH --nodes=1                                                # Number of nodes
#SBATCH --ntasks=1                                               # Number of tasks
#SBATCH --gres=gpu:1                                             # Request 1 GPU
#SBATCH --mem-per-gpu=50GB                                       # Memory per GPU
#SBATCH --time=96:00:00                                          # Maximum runtime
#SBATCH --array=0-21                                             # One job per cycle (0=pre_W1, 1..21=maintenance cycles)

# === Prepare output directory ===
OUTPUT_DIR=/d/hpc/home/sasop/outputs
mkdir -p "${OUTPUT_DIR}"    # ensure the folder exists

# Log environment details
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Output files will be written to ${OUTPUT_DIR}/"
echo "Cycle ID: ${SLURM_ARRAY_TASK_ID}"

# Check GPU visibility
srun nvidia-smi

# Prepare bind mounts (mount .env if present, so app can load DB credentials inside container)
BIND_MOUNTS="$(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs"
if [ -f "$(pwd)/.env" ]; then
    BIND_MOUNTS="${BIND_MOUNTS},$(pwd)/.env:/app/.env:ro"
    echo "Detected .env at $(pwd)/.env and mounted it to /app/.env"
else
    echo "No .env found at $(pwd)/.env. Postgres runs will fail until .env is provided."
fi

# Execute the Singularity container
singularity exec --nv \
    -e \
    --pwd /app \
    -B "${BIND_MOUNTS}" \
    docker://spartan300/nianet:vaepymoo \
    python main.py -alg particle_swarm -met SMAPE --cycle-id ${SLURM_ARRAY_TASK_ID}
