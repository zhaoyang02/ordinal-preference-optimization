from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import random
import os
import json
from tqdm import tqdm
import sys

random.seed(42)
device = "cuda"

ds = load_dataset("OPO-alignment/ListUltraFeedback", split="test")
ds = ds.shuffle(seed=42)

model_name = sys.argv[1]
path = "path/to/your/model"
output_path = "./model_answer/"
full_path = os.path.join(path, model_name)

model = AutoModelForCausalLM.from_pretrained(
    full_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(full_path, use_fast=True)

for example in tqdm(ds):
    prompt = example["prompt"]
    prompt_id = example["prompt_id"]

    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    attention_mask = model_inputs["attention_mask"]

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    with open(output_path+f'{model_name}.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({"prompt_id": prompt_id, "prompt": prompt, "response":response}) + "\n")
        
