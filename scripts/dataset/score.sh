#!/bin/bash

#SBATCH -N 1
#SBATCH -p GPU
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:8
#SBATCH -o /jet/home/yzhao15/yangzhao/handbook/logs/train.out

module load anaconda3
conda activate handbook
module load cuda
cd /jet/home/yzhao15/yangzhao/handbook

export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
export NUM_NODES=$SLURM_NNODES
log_file="/jet/home/yzhao15/yangzhao/handbook/logs/score.log"

torchrun --nproc_per_node=$GPUS_PER_NODE score.py > $log_file 2>&1