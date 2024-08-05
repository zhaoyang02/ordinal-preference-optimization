## Evaluation
We provide details on the evaluation of the models in this directory. Specifically, we evaluate on [ListUltraFeedback](https://huggingface.co/datasets/yangzhao02/ListUltraFeedback) [AlpacaEval 2](https://github.com/tatsu-lab/alpaca_eval) and [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge). ListUltraFeedback has 1.97k questions from original [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback), AlpacaEval 2 consists of 805 questions from 5 datasets, and MT-Bench covers 8 categories with 80 questions. 

### Proxy Model
1. Responses Generation
```
cd ./eval/proxy
python generation.py mistral-7b-base-neural_ndcg
```

2. Proxy Model Win Rate
change the `baseline` and `run_names` in code, then run:

```
# Pair-Preference
python pairwise_winrate.py

# Scoring
python reward_winrate.py
```

### AlpacaEval 2
Follow the installation instruction of [AlpacaEval 2](https://github.com/tatsu-lab/alpaca_eval)
```
cd ./eval/alpacaeval2

# generate responses for AlpacaEval 2
python generation.py mistal-7b-base-neural_ndcg

# evaluate with GPT-4 Turbo
alpaca_eval --% --model_outputs ".\model_output\mistral-7b-base-neural_ndcg.json" --reference_outputs ".\model_output\zephyr-7b-sft-full.json" --annotators_config "alpaca_eval_gpt4_turbo_fn" --output_path ".\results\all_pairs"
```
### MT-Bench
Follow the installation instruction of [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

Please put the file `./mt-bench/winrate.py` in `FastChat/fastchat/llm_judge/`dictionary and `./mt-bench/gpt-4-turbo.jsonl` in `FastChat/fastchat/llm_judge/data/mt_bench/reference_answer/`.

```
# generate responses for MT-Bench
python gen_model_answer.py --model-path alignment-handbook/zephyr-7b-sft-full --model-id zephyr-7b-sft-full
python gen_model_answer.py --model-path yangzhao02/mistral-7b-base-neural_ndcg --model-id mistral-7b-base-neural_ndcg

# Pair-Preference
python gen_judgment.py --mode pairwise-baseline --baseline-model zephyr-7b-sft-full --judge-model gpt-4-turbo --model-list mistral-7b-base-neural_ndcg
python show_result.py --mode pairwise-baseline --baseline-model zephyr-7b-sft-full --judge-model gpt-4-turbo --model-list mistral-7b-base-neural_ndcg

# Single Scoring
python gen_judgment.py --mode single --judge-model gpt-4-turbo --model-list zephyr-7b-sft-full mistral-7b-base-neural_ndcg
python show_result.py --mode single --judge-model gpt-4-turbo --model-list zephyr-7b-sft-full mistral-7b-base-neural_ndcg

# Calculate win rates for Single Scoring
python winrate.py
```