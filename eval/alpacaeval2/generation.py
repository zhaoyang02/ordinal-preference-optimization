from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import os
import json
from tqdm import tqdm
import sys

device = "cuda"

model_name = sys.argv[1]

path = "path/to/model"
output_path = "./model_output/"
full_path = os.path.join(path, model_name)

model = AutoModelForCausalLM.from_pretrained(
    full_path,
    torch_dtype=torch.float16
).to(device)
tokenizer = AutoTokenizer.from_pretrained(full_path, use_fast=True)

eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
tokenizer.pad_token_id = tokenizer.eos_token_id

outputs=[]
for i in tqdm(range(len(eval_set["instruction"]))):
    res={}
    prompt = eval_set["instruction"][i]
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        attention_mask=model_inputs["attention_mask"], 
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    res["dataset"]=eval_set["dataset"][i]
    res["instruction"]=prompt
    res["output"]=response
    res["generator"]=f"{model_name}"

    outputs.append(res)

with open(f"{output_path}{model_name}.json", "a") as f:
    json.dump(outputs, f, indent=4)

