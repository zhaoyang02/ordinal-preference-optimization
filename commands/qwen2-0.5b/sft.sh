#!/bin/bash

#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 15:00:00
#SBATCH --gpus=v100-32:4

module load anaconda3
conda activate ndcg
module load cuda
cd ndcg-preference-optimization

#configuration
export NUM_NODES=$SLURM_NNODES
export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
export WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(python find_port.py)

export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info

export MODEL="qwen2-0.5b"
export TASK="sft"
log_dir="./logs/${MODEL}/${TASK}"
mkdir -p $log_dir

# Debugging information
COMMAND='
echo "START TIME: $(date)"

env | grep '^SLURM'
echo "NUM_NODES: $NUM_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "HOSTNAME: $(hostname -s)"
# Check node communication
ping -c 3 $MASTER_ADDR
nc -zv $MASTER_ADDR $MASTER_PORT

accelerate launch \
    --config_file config/accelerate_configs/deepspeed_zero3.yaml  \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --max_restarts 1 \
    --tee 3 \
    --role $(hostname -s): \
    scripts/run_$TASK.py config/$MODEL/$TASK/config.yaml

echo "END TIME: $(date)"
'
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

output_file="${log_dir}/${TASK}_%j.out.log"
error_file="${log_dir}/${TASK}_%j.err.log"

srun $SRUN_ARGS --output $output_file --error $error_file bash -c "$COMMAND" 2>&1