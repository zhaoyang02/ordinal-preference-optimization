from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import json
from tqdm import tqdm

preference_name_or_path = "RLHFlow/pair-preference-model-LLaMA3-8B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(preference_name_or_path,
                                             torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(preference_name_or_path, use_fast=True)
tokenizer_plain = AutoTokenizer.from_pretrained(preference_name_or_path, use_fast=True)
tokenizer_plain.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"


prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
token_id_A = tokenizer.encode("A", add_special_tokens=False)
token_id_B = tokenizer.encode("B", add_special_tokens=False)
assert len(token_id_A) == 1 and len(token_id_B) == 1
token_id_A = token_id_A[0]
token_id_B = token_id_B[0]
temperature = 1.0

model.eval()

data_path = "./model_answer/"
baseline = "zephyr-7b-sft-full"
run_names = [
    "mistral-7b-base-neural_ndcg",
]

with open(data_path+f'{baseline}.jsonl', 'r', encoding='utf-8') as file1:
    sft_lines = [json.loads(line) for line in file1]

for run_name in run_names:
    print(f"calculate scores:{run_name}\n")
    with open(data_path+f'{run_name}.jsonl', 'r', encoding='utf-8') as file2:
        run_lines = [json.loads(line) for line in file2]

    total_count = 0
    correct_count = 0

    for sft_line, run_line in tqdm(zip(sft_lines, run_lines)):
        if sft_line["prompt_id"] != run_line["prompt_id"]:
            print("prompt_id not match")
            continue
        prompt=sft_line["prompt"]
        response_rejected = sft_line["response"]
        response_chosen = run_line["response"]
        instruction = [{"role": "user", "content": prompt}]

        context = tokenizer_plain.apply_chat_template(instruction, tokenize=False)
        responses = [response_chosen, response_rejected]
        probs_chosen = []
            
        for chosen_position in [0, 1]:
            # we swap order to mitigate position bias
            response_A = responses[chosen_position]
            response_B = responses[1 - chosen_position]
            prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
            message = [
                {"role": "user", "content": prompt},
            ]

            input_ids = tokenizer.encode(tokenizer.apply_chat_template(message, tokenize=False).replace(tokenizer.bos_token, ""), return_tensors='pt', add_special_tokens=False).cuda() 

            with torch.no_grad():
                output = model(input_ids)
            logit_A = output.logits[0, -1, token_id_A].item()
            logit_B = output.logits[0, -1, token_id_B].item()
            # take softmax to get the probability; using numpy
            Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
            logit_chosen = [logit_A, logit_B][chosen_position]
            prob_chosen = np.exp(logit_chosen / temperature) / Z
            probs_chosen.append(prob_chosen)

        avg_prob_chosen = np.mean(probs_chosen)
        correct = 0.5 if avg_prob_chosen == 0.5 else float(avg_prob_chosen > 0.5)
        total_count += 1
        correct_count += correct

    win_rate = correct_count / total_count if total_count != 0 else 0

    with open(("./pair_winrate.jsonl"), 'a', encoding='utf-8') as res:
        res.write(json.dumps({"run_name":f"{run_name}","win_rate": win_rate}) + "\n")
