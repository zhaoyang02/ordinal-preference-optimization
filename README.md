# Ordinal Preference Optimization
![ill](./assets/illustration.png)

# Installation
1. Clone this Repo
```
git clone ...
cd ordinal-preference-optimization
```

2. Install dependent packages:
```
conda env create -f environment.yaml
conda activate opo
```
# Instructions
## Training
```
# SFT Stage
accelerate launch --config_file config/accelerate_configs/deepspeed_zero3.yaml  scripts/run_sft.py mistral-7b-base/sft/config.yaml 

# Ordinal Preference Optimization
accelerate launch --config_file config/accelerate_configs/deepspeed_zero3.yaml  scripts/run_list.py mistral-7b-base/neural_ndcg/config.yaml 
```

For slurm, please refer to [command demo](./commands/mistral-7b-base/commands.sh) and submit the slurm job like:

`sbatch --job-name=all_pairs dpo.slurm dpo 0.1 8 all_pairs 8`

## Evaluation
Please refer to [Evaluation Instructions](./eval/README.md).

## Customize
1. modify the trainer file `./scripts/ndcg_trainer.py` to define your own loss function and the configuration `NDCGConfig` class in `./scripts/run_list.py`.

2. change the config files in `./config/model_name/method_name/config.yaml`. The illustration lies in [NeuralNDCG config](./config/mistral-7b-base/neural_ndcg/config.yaml).

# Dataset
How to get our Listwise Preference Dataset:

```
cd ./scripts/dataset

# Process
python process.py

# Reward Scoring Model
sbatch score.sh
```

# Acknowledge
This Repo is derived from HuggingFace [Alignment-handbook](https://github.com/huggingface/alignment-handbook) and [trl](https://github.com/huggingface/trl). The implementation of NeuralNDCG loss is heavily built on [allRank](https://github.com/allegro/allRank). Thank the authors for their excellent work.