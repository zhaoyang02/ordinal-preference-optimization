import random
import torch
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 4
LIST_SIZE = 3

class TokenHandler:
    def __init__(self, label_pad_token_id, list_size=3):
        self.label_pad_token_id = label_pad_token_id
        self.list_size = list_size

    def process_tokens(self, responses_tokens, prompt_tokens, scores):
        batch = {}

        for i, response_tokens in enumerate(responses_tokens):
            response_sequence_tokens = {
                k: response_tokens[f"prompt_{k}"] + response_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            response_sequence_tokens["labels"] = response_sequence_tokens["input_ids"][:]
            response_sequence_tokens["labels"][: len(response_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(response_tokens["prompt_input_ids"])

            k = f"response_{i}_"
            for type_key, tokens in response_sequence_tokens.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        for type_key, tokens in prompt_tokens.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{type_key}"] = tokens
        
        batch["scores"] = scores

        return batch

    def pad_to_length(self, tensor, length, pad_value):
        """Pads a tensor to the given length."""
        pad_size = length - tensor.size(1)
        if pad_size > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=pad_value)
        return tensor

    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        
        concatenated_batch = {}
        if not is_encoder_decoder:
            max_length = max(batch[f"response_{i}_input_ids"].shape[1] for i in range(self.list_size))

        for k in batch:
            if k.startswith("response") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = re.sub(r'response_\d+', 'concatenated', k)
                if concatenated_key not in concatenated_batch:
                    concatenated_batch[concatenated_key] = self.pad_to_length(batch[k], max_length, pad_value=pad_value)
                else:
                    concatenated_batch[concatenated_key] = torch.cat(
                        (
                            concatenated_batch[concatenated_key],
                            self.pad_to_length(batch[k], max_length, pad_value=pad_value),
                        ),
                        dim=0,
                    )
        for key in concatenated_batch:
            concatenated_batch[key] = concatenated_batch[key].to(device)
        
        return concatenated_batch
    
    def concatenated_forward(
        self, batch: Dict[str, Union[List, torch.LongTensor, torch.FloatTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        #concatenated_batch = self.concatenated_inputs(batch, device='cpu')
        all_logits = torch.rand(BATCH_SIZE * self.list_size,15,20, dtype=torch.float32)
        all_logps = torch.rand(BATCH_SIZE * self.list_size, dtype=torch.float32)
#        scores = torch.rand(BATCH_SIZE , self.list_size, dtype=torch.float32)
        
        return (all_logits, all_logps, batch["scores"])

# Generate multiple samples
def generate_sample(list_size):
    length = random.randint(5, 15)
    prompt_tokens = {
        "prompt_input_ids": [random.randint(0, 100) for _ in range(length)],
        "prompt_attention_mask": [1] * length
    }

    responses_tokens = []
    for _ in range(list_size):
        length1 = random.randint(10, 20)
        response = {
            "input_ids": [random.randint(0, 100) for _ in range(length1)],
            "attention_mask": [1] * length1,
            "prompt_input_ids": prompt_tokens["prompt_input_ids"],
            "prompt_attention_mask": prompt_tokens["prompt_attention_mask"]
        }
        responses_tokens.append(response)

    scores = [random.uniform(0, 1) for _ in range(list_size)]

    return prompt_tokens, responses_tokens, sorted(scores, reverse=True)

samples = [generate_sample(LIST_SIZE) for _ in range(BATCH_SIZE)]
print("Before Processing:")
for j in range(BATCH_SIZE):
    print(f"Sample {j}:")
    print(f"Prompt: {samples[j][0]}")
    for i, response in enumerate(samples[j][1]):
        print(f"Response {i}: {response}")

# Define the collator
@dataclass
class ListDataCollatorWithPadding:
    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if "prompt" in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]
                if k.endswith("_input_ids"):
                    if self.pad_token_id is None:
                        raise ValueError(
                            "Padding is enabled, but the tokenizer is not configured with a padding token."
                            " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                            " before calling the trainer."
                        )
                    padding_value = self.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                padded_batch[k] = torch.tensor([ex[k] for ex in features])

            elif k.endswith("scores"):
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch

# Process samples
token_handler = TokenHandler(label_pad_token_id=-100)
processed_samples = [token_handler.process_tokens(r, p, s) for p, r, s in samples]

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset = CustomDataset(processed_samples)
collate_fn = ListDataCollatorWithPadding()

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Print the batch from DataLoader
for batch in dataloader:
    print("\nBatch from DataLoader:")
    for key, value in batch.items():
        print(f"{key}: {value} (shape: {value.shape if isinstance(value, torch.Tensor) else 'N/A'})")
    concatenated_batch = token_handler.concatenated_inputs(batch, device=torch.device('cpu'))
    # print data after concatenation
    print("\nAfter Concatenation:")
    for key, value in concatenated_batch.items():
        print(f"{key}: {value} (shape: {value.shape})")

    forward_res=token_handler.concatenated_forward(batch)
    keys=["all_logits", "all_logps", "scores"]
    for i,res in enumerate(forward_res):
        print(f"{keys[i]}:{res} (shape: {res.shape})")

    print("\nAfter Matrix transformation:")
    for i,res in enumerate(forward_res):
        if i==0:
            continue
        elif i==1:
            all_logps=res=res.view(LIST_SIZE,BATCH_SIZE).T
        print(f"{keys[i]}:{res} (shape: {res.shape})")
    scores=forward_res[2]

print(f"\nall_logps:{all_logps} (shape: {all_logps.shape})")
print(f"\nscores:{scores} (shape: {scores.shape})")
