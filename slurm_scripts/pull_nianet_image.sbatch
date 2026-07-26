#!/bin/bash
#SBATCH --job-name=pull-nianet-image
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/d/hpc/home/sasop/outputs/pull-nianet-image-%j.out
#SBATCH --error=/d/hpc/home/sasop/outputs/pull-nianet-image-%j.err

set -euo pipefail

export APPTAINER_CACHEDIR=/d/hpc/home/sasop/apptainer-cache
export SINGULARITY_CACHEDIR=/d/hpc/home/sasop/apptainer-cache
export TMPDIR=/d/hpc/home/sasop/apptainer-tmp

mkdir -p "$APPTAINER_CACHEDIR" "$TMPDIR" /d/hpc/home/sasop/images /d/hpc/home/sasop/outputs

rm -f /d/hpc/home/sasop/images/tmp-copy-*
singularity cache clean --force || true

singularity pull --force \
  /d/hpc/home/sasop/images/nianet-vaepymoo-latest.sif \
  docker://spartan300/nianet:vaepymoo

ln -sfn /d/hpc/home/sasop/images/nianet-vaepymoo-latest.sif \
  /d/hpc/home/sasop/images/nianet-vaepymoo-current.sif

ls -lh /d/hpc/home/sasop/images/nianet-vaepymoo-latest.sif
