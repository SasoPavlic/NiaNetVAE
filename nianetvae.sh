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
#SBATCH --array=1-21                                             # One job per cycle

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

# Execute the Singularity container
singularity exec --nv \
    -e \
    --pwd /app \
    -B $(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs \
    docker://spartan300/nianet:vaepymoo \
    python main.py -alg particle_swarm -met SMAPE --cycle-id ${SLURM_ARRAY_TASK_ID}
