import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset,concatenate_datasets
import random
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
#from collections import defaultdict
import shutil

random.seed(42)

# 初始化分布式进程组
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

#ds_a = load_dataset("yangzhao02/RAW_ListUltraFeedback", split="train[:64]")
#ds_b = load_dataset("yangzhao02/RAW_ListUltraFeedback", split="test[:64]")

# 定义奖励目标的属性
attributes = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm', 'code-complexity',
    'code-style', 'code-explanation', 'code-instruction-following', 'code-readability'
]

# 我们感兴趣的属性
target_attributes = [
    'ultrafeedback-overall_score',
    # 'ultrafeedback-instruction_following',
    # 'ultrafeedback-truthfulness',
    # 'ultrafeedback-honesty',
    # 'ultrafeedback-helpfulness'
]

# 找到这些属性在 `attributes` 列表中的索引
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

        # 进行模型推理
        with torch.no_grad():
            output = model(input_ids)
            multi_obj_rewards = output.rewards.cpu().float()

        # 从 `multi_obj_rewards` 中提取对应的奖励值
        target_rewards = multi_obj_rewards[0, target_indices]

        overall_scores.append(target_rewards[0].item())

    # 将 scores 和 responses 对齐并排序
    scores_and_responses = list(zip(overall_scores, responses))
    scores_and_responses.sort(key=lambda x: x[0])
    
    # 获取最低的两个和最高的两个分数对应的 responses
    lowest_two = scores_and_responses[:2]
    highest_two = scores_and_responses[-2:]

    # 获取剩余的分数和 responses
    remaining_scores_and_responses = scores_and_responses[2:-2]
    
    # 随机选择剩余的四个
    random.shuffle(remaining_scores_and_responses)
    random_four = remaining_scores_and_responses[:4]

    # 合并这些分数和 responses
    selected_scores_and_responses = lowest_two + highest_two + random_four
    random.shuffle(selected_scores_and_responses)

    # 分离分数和 responses
    selected_scores, selected_responses = zip(*selected_scores_and_responses)

    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "all_responses": list(selected_responses),
        "overall_scores": list(selected_scores),
    }

# 使用 DataLoader 进行批处理，并显示进度条
def process_dataset(dataset,task="train"):
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, collate_fn=lambda x: x)
    local_rank = torch.distributed.get_rank()
    processed_data = []
    #current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #print(f"Processing rank {local_rank}, {task} dataset at {current_time}...")
    
    # 使用 tqdm 包装 dataloader 以显示进度条
    for batch in tqdm(dataloader, desc="Processing dataset", leave=False):
        for item in batch:
            processed_item = get_reward_score(item)
            processed_data.append(processed_item)
        torch.cuda.empty_cache()  # 清理缓存

    partial_dataset = Dataset.from_list(processed_data)
    save_path = f"/ocean/projects/cis230060p/yzhao15/.cache/yzhao15/list_ultra/partial_dataset_{task}_{local_rank}.arrow"
    
    # 检查目标文件夹是否存在，如果存在则删除
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        #print(f"Deleted {local_rank} previous dataset!")
    
    partial_dataset.save_to_disk(save_path)
    # 等待所有进程完成
    torch.distributed.barrier()
    
    if local_rank == 0:
        # 合并所有分布式进程保存的数据集
        all_datasets = [Dataset.load_from_disk(f"/ocean/projects/cis230060p/yzhao15/.cache/yzhao15/list_ultra/partial_dataset_{task}_{i}.arrow") for i in range(torch.distributed.get_world_size())]
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