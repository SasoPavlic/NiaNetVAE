#!/bin/bash
## Running code on SLURM cluster
##https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html
#SBATCH --job-name=pso-msl               # Job name
#SBATCH --output=pso-msl-%j.out          # Standard output file (%j: job ID)
#SBATCH --error=pso-msl-%j.err           # Standard error file
#SBATCH --partition=gpu                  # GPU partition
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --mem-per-gpu=80GB               # RAM Memory per GPU and not VRAM
#SBATCH --time=96:00:00                  # Maximum runtime

# Log environment details
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Running on GPU node..."

# Check GPU visibility
srun nvidia-smi

# Execute the Singularity container
singularity exec --nv \
    -e \
    --pwd /app \
    -B $(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs \
    docker://spartan300/nianet:vae \
    python main.py -alg particle_swarm