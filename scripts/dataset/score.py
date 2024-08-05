import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset,concatenate_datasets
import random
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
import shutil

random.seed(42)

# initialize the process group
torch.distributed.init_process_group(backend='nccl')

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')

path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
model = AutoModelForSequenceClassification.from_pretrained(
    path, 
    trust_remote_code=True, 
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

model = model.to(device)
model = DDP(model, device_ids=[local_rank])

ds_a = load_dataset("yangzhao02/RAW_ListUltraFeedback", split="train")
ds_b = load_dataset("yangzhao02/RAW_ListUltraFeedback", split="test")

attributes = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm', 'code-complexity',
    'code-style', 'code-explanation', 'code-instruction-following', 'code-readability'
]
target_attributes = ['ultrafeedback-overall_score']
target_indices = [attributes.index(attr) for attr in target_attributes]

def get_reward_score(x):
    prompt_id = x["prompt_id"]
    prompt = x["prompt"]
    responses = x["all_responses"]
    overall_scores = []

    for i, response in enumerate(responses):
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(input_ids)
            multi_obj_rewards = output.rewards.cpu().float()

        target_rewards = multi_obj_rewards[0, target_indices]

        overall_scores.append(target_rewards[0].item())

    scores_and_responses = list(zip(overall_scores, responses))
    scores_and_responses.sort(key=lambda x: x[0])
    
    # get the lowest two and highest two scores
    lowest_two = scores_and_responses[:2]
    highest_two = scores_and_responses[-2:]

    # randomly select
    remaining_scores_and_responses = scores_and_responses[2:-2]
    random.shuffle(remaining_scores_and_responses)
    random_four = remaining_scores_and_responses[:4]

    selected_scores_and_responses = lowest_two + highest_two + random_four
    random.shuffle(selected_scores_and_responses)

    selected_scores, selected_responses = zip(*selected_scores_and_responses)

    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "all_responses": list(selected_responses),
        "overall_scores": list(selected_scores),
    }

def process_dataset(dataset,task="train"):
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, collate_fn=lambda x: x)
    local_rank = torch.distributed.get_rank()
    processed_data = []

    for batch in tqdm(dataloader, desc="Processing dataset", leave=False):
        for item in batch:
            processed_item = get_reward_score(item)
            processed_data.append(processed_item)
        torch.cuda.empty_cache() 

    partial_dataset = Dataset.from_list(processed_data)
    save_path = f"./list_ultra/partial_dataset_{task}_{local_rank}.arrow"
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    partial_dataset.save_to_disk(save_path)
    torch.distributed.barrier()
    
    if local_rank == 0:
        all_datasets = [Dataset.load_from_disk(f"./list_ultra/partial_dataset_{task}_{i}.arrow") for i in range(torch.distributed.get_world_size())]
        full_dataset = concatenate_datasets(all_datasets)
        return full_dataset
    else:
        return None

ds_a = process_dataset(ds_a,task="train")
torch.distributed.barrier()

ds_b = process_dataset(ds_b,task="test")
torch.distributed.barrier()

if local_rank == 0:
    all_ds = DatasetDict()
    all_ds["train"] = ds_a
    all_ds["test"] = ds_b

    all_ds.push_to_hub("yangzhao02/ListUltraFeedback")
    print("Pushed to hub! Done!")