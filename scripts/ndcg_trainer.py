import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import re
import json
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from trl import DPOTrainer
from trl.trainer.utils import pad_to_length
from loss_utils import dcg, deterministic_neural_sort, sinkhorn_scaling, stochastic_neural_sort

from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
    
@dataclass
class ListDataCollatorWithPadding:
    r"""DATAPROCESSING STEP 3
    List DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
    """
    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # adapted from https://stackoverflow.com/questions/73256206
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
                # for the prompt, flip back so padding is on left side
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])

            elif k.endswith("scores"):
                padded_batch[k] = torch.tensor([ex[k] for ex in features],dtype=torch.float16)
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


class NDCGTrainer(DPOTrainer):

    _tag_names = ["trl", "ndcg"]

    def __init__(self, **kwargs):
        training_args = kwargs["args"]
        self.is_qwen2 = training_args.is_qwen2
        self.alpha = training_args.alpha
        self.gamma = training_args.gamma
        self.tau = training_args.tau
        self.list_size = training_args.list_size
        self.pairwise_type = training_args.pairwise_type
        self.ablation_type = training_args.ablation_type

        super().__init__(**kwargs)  # Pass all other arguments using **kwargs
        
        
    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        '''DATAPROCESSING STEP 2
        Tokenize a single row from a Listwise preference dataset.
        returns:
        batch={
            'prompt_input_ids':torch.LongTensor,
            'prompt_attention_mask':torch.LongTensor,
            'response_0_input_ids':torch.LongTensor,
            'response_0_attention_mask':torch.LongTensor,
            'response_0_labels':torch.LongTensor,
            'response_1_input_ids':torch.LongTensor,
            'response_1_attention_mask':torch.LongTensor,
            'response_1_labels':torch.LongTensor,
            ...
            ...
            'scores':torch.tensor
        }
        '''
        batch = {}
        prompt = feature["prompt"]
        responses = [feature[f"response_{i}"] for i in range(self.list_size)]
        
        if not self.is_encoder_decoder:
            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            responses_tokens=[]

            for i, response in enumerate(responses):
                if not isinstance(response, str):
                    raise ValueError(f"response_{i} should be an str but got {type(response)}")
                responses_tokens.append(self.build_tokenized_answer(prompt, response))

            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            responses_prompt_len_input_ids = [len(c["prompt_input_ids"]) for c in responses_tokens]
            prompt_len_input_ids = min(responses_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # add BOS token to head of prompt
            if not self.is_qwen2:
                # Qwen2 tokenizers don't have bos_token_id: https://github.com/QwenLM/Qwen2/issues/486
                prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
                for response_tokens in responses_tokens:
                    response_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + response_tokens["prompt_input_ids"]
                    response_tokens["prompt_attention_mask"] = [1] + response_tokens["prompt_attention_mask"]
                    response_tokens["input_ids"].append(self.tokenizer.eos_token_id)
                    response_tokens["attention_mask"].append(1)
            
            longer_response_length = max(len(response_tokens["input_ids"]) for response_tokens in responses_tokens)
            
            # if combined sequence is too long, truncate the prompt
            for answer_tokens in responses_tokens:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in responses_tokens:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels and construct batch data
            for i,response_tokens in enumerate(responses_tokens):
                response_sequence_tokens = {
                    k: response_tokens[f"prompt_{k}"] + response_tokens[k] for k in ["input_ids", "attention_mask"]
                }
                response_sequence_tokens["labels"] = response_sequence_tokens["input_ids"][:]
                response_sequence_tokens["labels"][: len(response_tokens["prompt_input_ids"])] = [
                    self.label_pad_token_id
                ] * len(response_tokens["prompt_input_ids"])

                k=f"response_{i}_"
                for type_key, tokens in response_sequence_tokens.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

            for type_key, tokens in prompt_tokens.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{type_key}"] = tokens
            
            batch["scores"] = feature["scores"]

        return batch
    
    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor, torch.FloatTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        '''DATAPROCESSING STEP 4
        input:
            batch=[2+lise_size*3+1 : [batch_size,seq_len]]
        returns:
            concatenated_batch={3 : [batch_size*lise_size,seq_len]}
            #key: concatenated_input_ids, concatenated_attention_mask, concatenated_labels
        '''
        
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
                    concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
                else:
                    concatenated_batch[concatenated_key] = torch.cat(
                        (
                            concatenated_batch[concatenated_key],
                            pad_to_length(batch[k], max_length, pad_value=pad_value),
                        ),
                        dim=0,
                    )
        for key in concatenated_batch:
            concatenated_batch[key] = concatenated_batch[key].to(device)
        
        return concatenated_batch

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor, torch.FloatTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Returns:
            Tuple([batch_size*list_size,seq_len,vocab_size], [batch_size, list_size])
        '''
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_per_res = batch["response_0_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        average_log_prob_flag = True if self.loss_type in ["simpo"] else False

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=average_log_prob_flag,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        ).view(self.list_size, len_per_res).T

        return (all_logps, all_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor, torch.FloatTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        (
            policy_all_logps,
            policy_all_logits,
        ) = self.concatenated_forward(model, batch)
        
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_all_logps,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_all_logps,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        if self.loss_type == "approx_ndcg":
            losses = self.approxNDCG_loss(
                policy_all_logps,
                reference_all_logps,
                batch["scores"],
            )
        elif self.loss_type in ["dpo","simpo"]:
            losses = self.dpo_loss(
                policy_all_logps,
                reference_all_logps,
                batch["scores"],
            )
        elif self.loss_type == "hinge":
            losses = self.hinge_loss(
                policy_all_logps,
                reference_all_logps,
                batch["scores"],
            )
        elif self.loss_type == "lipo":
            losses = self.lipo_loss(
                policy_all_logps,
                reference_all_logps,
                batch["scores"],
            )
        elif self.loss_type == "neural_ndcg":
            losses = self.neuralNDCG_loss(
                policy_all_logps,
                reference_all_logps,
                batch["scores"],
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss}")

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}logps"] = policy_all_logps.detach().mean().cpu()
        metrics[f"{prefix}logits"] = policy_all_logits.detach().mean().cpu()

        return losses.mean(), metrics

    def dpo_loss(
        self,
        policy_all_logps: torch.FloatTensor,
        reference_all_logps: torch.FloatTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        if self.loss_type == "simpo":
            logratios = policy_all_logps
            logratios = logratios.to(self.accelerator.device)
            gamma=self.gamma

        elif self.loss_type=="dpo":
            if policy_all_logps.shape == reference_all_logps.shape:
                logratios = policy_all_logps - reference_all_logps
                logratios = logratios.to(self.accelerator.device)
            else:
                raise ValueError(f"policy_all_logps shape:{policy_all_logps.shape} and reference_all_logps shape:{reference_all_logps.shape} should be the same!")
            gamma=0.0
        else: 
            raise ValueError(
                    f"Unknown loss type: {self.loss_type}. Should be one of ['dpo', 'simpo']"
                )

        if self.pairwise_type == "all_pairs":
            logratios_diffs = logratios[:, :, None] - logratios[:, None, :]
            indices=torch.triu_indices(row=self.list_size, col=self.list_size, offset=1)
            diag_diffs=logratios_diffs[:,indices[0], indices[1]].reshape(1,-1)
            losses = (
                    -F.logsigmoid(self.beta * diag_diffs - gamma) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * diag_diffs + gamma) * self.label_smoothing
                )
        elif self.pairwise_type == "best_with_others":
            logratios_diffs = logratios[:, :, None] - logratios[:, None, :]
            first_diffs = logratios_diffs[:, 0, 1:].reshape(1,-1)
            losses = (
                    -F.logsigmoid(self.beta * first_diffs - gamma) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * first_diffs + gamma) * self.label_smoothing
                )

        elif self.pairwise_type == "best_with_worst":
            logratios_diffs = logratios[:, :, None] - logratios[:, None, :]
            diffs = logratios_diffs[:, 0, -1]
            losses = (
                    -F.logsigmoid(self.beta * diffs - gamma) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * diffs + gamma) * self.label_smoothing
                )

        elif self.pairwise_type == "list_mle":
            rewards = self.beta * logratios
            exp_rewards = torch.exp(rewards)
            result = torch.zeros(rewards.size(0)).to(self.accelerator.device)
            for i in range(rewards.size(0)):
                prod = 1.0
                for k in range(rewards.size(1)):
                    numerator = exp_rewards[i, k]
                    denominator = torch.sum(exp_rewards[i, k:])
                    prod *= numerator / denominator
                result[i] = torch.log(prod)
            losses = -result

        else:
            raise ValueError(f"pairwise_type should be all_pairs, best_with_others or best_with_worst but got {self.pairwise_type}")
        
        return losses
    
    def hinge_loss(
        self,
        policy_all_logps: torch.FloatTensor,
        reference_all_logps: torch.FloatTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        if policy_all_logps.shape == reference_all_logps.shape:
            logratios = policy_all_logps - reference_all_logps
            logratios = logratios.to(self.accelerator.device)
        else:
            raise ValueError(f"policy_all_logps shape:{policy_all_logps.shape} and reference_all_logps shape:{reference_all_logps.shape} should be the same!")
        
        logratios_diffs = logratios[:, :, None] - logratios[:, None, :]
        indices=torch.triu_indices(row=self.list_size, col=self.list_size, offset=1)
        diag_diffs=logratios_diffs[:,indices[0], indices[1]].reshape(1,-1)

        losses = F.relu(1- self.beta * diag_diffs)        
        return losses
        
    def approxNDCG_loss(
        self,
        policy_all_logps: torch.FloatTensor,
        reference_all_logps: torch.FloatTensor,
        scores: torch.FloatTensor,
    ) ->torch.FloatTensor:
        '''
        The code is based on https://github.com/allegro/allRank/blob/master/allrank/models/losses/approxNDCG.py
        Args:
            policy_all_logps: [batch_size, list_size]
            reference_all_logps: [batch_size, list_size]
            scores: [batch_size, list_size]
        Returns:
            [batch_size,]
        '''
        if policy_all_logps.shape == reference_all_logps.shape:
            logratios = policy_all_logps - reference_all_logps
            y_pred = self.beta * logratios
            y_pred = y_pred.to(self.accelerator.device)
        else:
            raise ValueError(f"policy_all_logps shape:{policy_all_logps.shape} and reference_all_logps shape:{reference_all_logps.shape} should be the same!")
        y_true = scores.to(self.accelerator.device)
        
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(self.accelerator.device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum((torch.pow(2, y_true) - 1) / D, dim=-1)

        G = (torch.pow(2, y_true) - 1) / maxDCGs[:, None]
        scores_diffs = (y_pred[:, :, None] - y_pred[:, None, :])
        approx_pos = 0.5 + torch.sum(torch.sigmoid(-self.alpha * scores_diffs), dim=-1)
        approx_D = torch.log2(1. + approx_pos)
        approx_NDCG = torch.sum((G / approx_D), dim=-1)
        
        return -approx_NDCG
    
    def lipo_loss(
        self,
        policy_all_logps: torch.FloatTensor,
        reference_all_logps: torch.FloatTensor,
        scores: torch.FloatTensor,
        eps=1e-10, padded_value_indicator=-1,sigma=1.,k=None
    ) -> torch.FloatTensor:
        
        device = self.accelerator.device

        if policy_all_logps.shape == reference_all_logps.shape:
            logratios = policy_all_logps - reference_all_logps
            y_pred = self.beta * logratios
            y_pred = y_pred.to(self.accelerator.device)
        else:
            raise ValueError(f"policy_all_logps shape:{policy_all_logps.shape} and reference_all_logps shape:{reference_all_logps.shape} should be the same!")

        y_true = scores.to(self.accelerator.device)

        padded_mask = y_true == padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)

        losses = torch.log(weighted_probas)

        return -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])

    def neuralNDCG_loss(
        self,
        policy_all_logps: torch.FloatTensor,
        reference_all_logps: torch.FloatTensor,
        scores: torch.FloatTensor,
        padded_value_indicator=-1, temperature=1., powered_relevancies=True, k=None,
        )-> torch.FloatTensor:
        '''
        The code is based on https://github.com/allegro/allRank/blob/master/allrank/models/losses/neuralNDCG.py
        Args:
            policy_all_logps: [batch_size, list_size]
            reference_all_logps: [batch_size, list_size]
            scores: [batch_size, list_size]
        Returns:
            [batch_size,]
        '''
        #The code is based on https://github.com/allegro/allRank/blob/master/allrank/models/losses/neuralNDCG.py

        dev=self.accelerator.device
        temperature = self.tau

        if policy_all_logps.shape == reference_all_logps.shape:
            logratios = policy_all_logps - reference_all_logps
            y_pred = self.beta * logratios
            y_pred = y_pred.to(self.accelerator.device)
        else:
            raise ValueError(f"policy_all_logps shape:{policy_all_logps.shape} and reference_all_logps shape:{reference_all_logps.shape} should be the same!")

        y_true = scores.to(self.accelerator.device)

        k = y_true.shape[1]

        mask = (y_true == padded_value_indicator)
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask,dev=dev).unsqueeze(0)

        # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
        P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                                    mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
        P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])

        # Mask P_hat and apply to true labels, ie approximately sort them
        P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.)
        y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1).unsqueeze(0)
        
        if powered_relevancies:
            y_true_masked = torch.pow(2., y_true_masked) - 1.

        ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
        discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
        discounted_gains = ground_truth * discounts

        if powered_relevancies:
            idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
        else:
            idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)

        discounted_gains = discounted_gains[:, :, :k]
        ndcg = discounted_gains.sum(dim=-1) / (idcg + 1e-10)
        idcg_mask = idcg == 0.
        ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)

        assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
        if idcg_mask.all():
            return torch.tensor(0.)

        mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
        return -1. * mean_ndcg

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    
    def get_batch_samples(self, model, batch: Dict[str, Union[List, torch.LongTensor, torch.FloatTensor]]) -> Tuple[str, str]:
        return super().get_batch_samples(model, batch)

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits": metrics["eval_logits"],
        }
        valid_keys = [k for k in logits_dict.keys() if k not in ignore_keys]

        if len(valid_keys) == 1:
            logits = logits_dict[valid_keys[0]].to(self.accelerator.device)
        else:
            logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
            logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)

        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)
