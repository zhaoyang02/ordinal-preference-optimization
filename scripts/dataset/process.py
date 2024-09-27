from datasets import load_dataset, DatasetDict, concatenate_datasets,Dataset
import hashlib
import random

random.seed(42)

ds1 = load_dataset("openbmb/UltraFeedback", split="train", revision="40b436560ca83a8dba36114c22ab3c66e43f6d5e")
ds2_a=  load_dataset("princeton-nlp/llama3-ultrafeedback-armorm", split="train")
ds2_b=  load_dataset("princeton-nlp/llama3-ultrafeedback-armorm", split="test")

ds2 = concatenate_datasets([ds2_a, ds2_b])

def format_ds1(x):
    prompt = x["instruction"]
    prompt_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    all_responses = [c["response"] for c in x["completions"]]
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "all_responses":all_responses,
    }

ds1 = ds1.map(format_ds1, num_proc=16, remove_columns=ds1.column_names)

combined_data = {}

for example in ds2:
    prompt_id = example["prompt_id"]
    if prompt_id not in combined_data:
        combined_data[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt": example["prompt"],
            "all_responses": example["all_generated_responses"],
        }

for example in ds1:
    prompt_id = example["prompt_id"]
    if prompt_id in combined_data:
        combined_data[prompt_id]["all_responses"].extend(example["all_responses"])

new_dataset = Dataset.from_dict({
    "prompt_id": [v["prompt_id"] for v in combined_data.values()],
    "prompt": [v["prompt"] for v in combined_data.values()],
    "all_responses": [v["all_responses"] for v in combined_data.values()],
})

train_prompt_ids = set(ds2_a["prompt_id"])
test_prompt_ids = set(ds2_b["prompt_id"])

train_data = new_dataset.filter(lambda x: x["prompt_id"] in train_prompt_ids)
test_data = new_dataset.filter(lambda x: x["prompt_id"] in test_prompt_ids)

all_ds = DatasetDict()
all_ds["train"] = train_data
all_ds["test"] = test_data

for i in range(2):
    print(all_ds["train"][i])

all_ds.push_to_hub("OPO-alignment/RAW_ListUltraFeedback")
print("Done!")