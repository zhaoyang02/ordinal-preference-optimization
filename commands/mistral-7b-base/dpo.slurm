#!/bin/bash

#SBATCH -N 2
#SBATCH -p GPU
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:16

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
export NCCL_TIMEOUT=7200
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

export MODEL="mistral-7b-base"
export TASK=$1
log_dir="./logs/${MODEL}/${TASK}"
export CONFIG_FILE="./config/${MODEL}/${TASK}/config.yaml"
mkdir -p $log_dir

BETA=$2
LIST_SIZE=$3
PAIRWISE_TYPE=$4
GRAD_ACCUM=$5
BATCH_SIZE=$(($WORLD_SIZE*$GRAD_ACCUM))

export unique_config_file="./config/${MODEL}/${TASK}/config_${SLURM_JOB_ID}.yaml"
cp ./config/${MODEL}/${TASK}/config.yaml $unique_config_file
RUN_NAME="$MODEL-$TASK-listsize_$LIST_SIZE-tau_$TAU-beta_$BETA-batchsize_$BATCH_SIZE"
OUTPUT_DIR="./$MODEL/$RUN_NAME"

# Modify the unique configuration file
sed -i "s|^\s*loss_type:.*|loss_type: ${TASK}|" $unique_config_file
sed -i "s|^\s*beta:.*|beta: ${BETA}|" $unique_config_file
sed -i "s|^\s*list_size:.*|list_size: ${LIST_SIZE}|" $unique_config_file
sed -i "s|^\s*pairwise_type:.*|pairwise_type: ${PAIRWISE_TYPE}|" $unique_config_file
sed -i "s|^\s*gradient_accumulation_steps:.*|gradient_accumulation_steps: ${GRAD_ACCUM}|" $unique_config_file

sed -i "s|^\s*run_name:.*|run_name: ${RUN_NAME}|" $unique_config_file
sed -i "s|^\s*hub_model_id:.*|hub_model_id: ${RUN_NAME}|" $unique_config_file
sed -i "s|^\s*output_dir:.*|output_dir: ${OUTPUT_DIR}|" $unique_config_file

COMMAND='
echo "$RUN_NAME START TIME: $(date)"
cat $unique_config_file
export NCCL_TIMEOUT=7200
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

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
    scripts/run_list.py $unique_config_file \

echo "$RUN_NAME END TIME: $(date)"
'
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "
output_file="${log_dir}/${RUN_NAME}.out.log"
error_file="${log_dir}/${RUN_NAME}.err.log"

srun $SRUN_ARGS --output $output_file --error $error_file bash -c "$COMMAND" 2>&1

rm $unique_config_file