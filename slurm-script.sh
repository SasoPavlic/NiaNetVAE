#!/bin/bash
#SBATCH -J nianet
#SBATCH -o nianet-%j.out
#SBATCH -e nianet-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00

singularity exec -e --pwd /app -B ./logs:/app/logs --nv docker://spartan300/nianet:latest python ./rnn_vae_run.py
