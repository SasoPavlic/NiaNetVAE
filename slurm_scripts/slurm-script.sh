#!/bin/bash
#SBATCH -J nianet
#SBATCH -o nianet-%j.out
#SBATCH -e nianet-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

singularity exec -e --pwd /app -B /ceph/grid/home/sasop/logs:/app/logs --nv docker://spartan300/nianet:latest python ./rnn_vae_run.py