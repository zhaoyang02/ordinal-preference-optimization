import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.float16
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

attributes = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm', 'code-complexity',
    'code-style', 'code-explanation', 'code-instruction-following', 'code-readability'
]

target_attributes = [
    'ultrafeedback-overall_score',
]
target_indices = [attributes.index(attr) for attr in target_attributes]

data_path = "./model_answer/"
score_path = "./scores/"
baseline = "zephyr-7b-sft-full"
run_names = [
    "zephyr-7b-sft-full",
    "mistral-7b-base-neural_ndcg",
]

for run_name in run_names:
    print(f"calculate scores:{run_name}\n")

    with open(data_path+f'{run_name}.jsonl', 'r', encoding='utf-8') as file:
        lines = [json.loads(line) for line in file]
    
    for line in tqdm(lines):
        prompt_id = line["prompt_id"]
        messages =[
            {"role": "user", "content": line["prompt"]},
            {"role": "assistant", "content": line["response"]}
        ]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(input_ids)
            multi_obj_rewards = output.rewards.cpu().float()
        target_rewards = multi_obj_rewards[0, target_indices]
        overall_score=target_rewards[0].item()

        with open(score_path+f'{run_name}.jsonl', 'a', encoding='utf-8') as score_file:
            score_file.write(json.dumps({"prompt_id": prompt_id,"overall_score": overall_score}) + "\n")

sft_scores = {}
with open(score_path+baseline, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        prompt_id = data["prompt_id"]
        overall_score = data["overall_score"]
        sft_scores[prompt_id]=overall_score

for run_name in run_names:
    print(f"calculate win rate:{run_name}\n")

    win_count = 0
    total_count = 0

    with open(score_path+f'{run_name}.jsonl', 'r', encoding='utf-8') as scores:
        for line in scores:
            line = json.loads(line)
            prompt_id = line["prompt_id"]
            overall_score = line["overall_score"]
            if overall_score > sft_scores[prompt_id]:
                win_count += 1
            elif overall_score == sft_scores[prompt_id]:
                win_count += 0.5
            total_count += 1

    win_rate = win_count / total_count if total_count != 0 else 0

    with open(("./scoring_winrate.jsonl"), 'a', encoding='utf-8') as res:
        res.write(json.dumps({"run_name":f"{run_name}","win_rate": win_rate}) + "\n")