#!/bin/bash

#SBATCH -N 1
#SBATCH -p GPU
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:8

module load anaconda3
conda activate opo
module load cuda
cd ordinal-preference-optimization

export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
export NUM_NODES=$SLURM_NNODES
log_file="./logs/score.log"

torchrun --nproc_per_node=$GPUS_PER_NODE score.py > $log_file 2>&1