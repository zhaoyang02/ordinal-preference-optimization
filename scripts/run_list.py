#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import maybe_insert_system_message, is_openai_format
from peft import PeftConfig, PeftModel
from dataclasses import dataclass, field
from typing import Optional, Literal
from ndcg_trainer import NDCGTrainer,ListDataCollatorWithPadding

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


@dataclass
class NDCGConfig(DPOConfig):
    alpha: Optional[float] = field(
        default=25.0,
        metadata={"help": "alpha parameter for ApproxNDCG loss."},
    )
    gamma: Optional[float] = field(
        default=0.1,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    tau: Optional[float] = field(
        default=1.0,
        metadata={"help": "The temperature parameter for Sinkhorn scaling."},
    )
    list_size: Optional[int] = field(
        default=8,
        metadata={"help": "The number of responses to consider in the listwise ranking."},
    )
    loss_type: Optional[Literal["dpo","simpo","hinge", "lipo","approx_ndcg","neural_ndcg"]] = field(
        default="neural_ndcg",
        metadata={"help": "The loss function to use for training."},
    )
    pairwise_type: Optional[Literal["all_pairs","best_with_others", "best_with_worst","list_mle"]] = field(
        default="all_pairs",
        metadata={"help": "The type of pairwise comparison to use for training."},
    )
    is_qwen2: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use QWEN2 or not."},
    )

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "simpo","list"],
    auto_insert_empty_system_msg: bool = True,
    change_template = None,
    list_size=8,
):
    """DATAPROCESSING STEP 1
    if task=="list", apply chat template & sort responses by score and select list_size responses
    return:
        ['prompt', 'scores', 'response_0', 'response_1', 'response_2',..., 'response_{list_size-1}']
    
    """
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "simpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            if example["text_chosen"].startswith(tokenizer.bos_token):
                example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            if example["text_rejected"].startswith(tokenizer.bos_token):
                example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "list":
        #Apply chat template & sort responses by score and select list_size responses
        prompt=example["prompt"]
        prompt_message = [{"role": "user", "content": prompt}]
        example["text_prompt"] = tokenizer.apply_chat_template(prompt_message, tokenize=False)

        responses = example["all_responses"]
        overall_scores = example["overall_scores"]

        # Sort responses by score
        scores_and_responses = sorted(zip(overall_scores, responses), key=lambda x: x[0], reverse=True)

        if list_size < 8:
            highest = scores_and_responses[:1]
            lowest = scores_and_responses[-1:]

            if list_size == 2:
                selected_scores, selected_responses = zip(*(highest + lowest))
            else:
            # randomly select list_size-2 samples from remaining responses
                remaining_scores_and_responses = scores_and_responses[1:-1]
                random.shuffle(remaining_scores_and_responses)
                random_select = remaining_scores_and_responses[:list_size - 2]
                
                selected_scores, selected_responses = zip(*(highest + sorted(random_select,key=lambda x: x[0],reverse=True) + lowest))
        else:
            selected_scores, selected_responses = zip(*scores_and_responses)

        example["scores"] = list(selected_scores)

        for i in range(list_size):
            message = [{"role": "assistant", "content": selected_responses[i]}]
            example[f"response_{i}"] = tokenizer.apply_chat_template(message, tokenize=False)

    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, NDCGConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["prompt", "all_responses", "overall_scores"],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    if "mistral" in model_args.model_name_or_path.lower():
        change_template = "mistral"
    else:
        change_template = None

    if "qwen2" in model_args.model_name_or_path.lower():
        training_args.is_qwen2 = True
    #####################
    # Apply chat template & Selecct listsize responses
    #####################
    column_names = list(raw_datasets["train"].features)
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "list",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "change_template": change_template,
            "list_size": training_args.list_size,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Apply chat template & Selecct num=listsize responses",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt"}
        )
    #print(f"raw_dataset example: {raw_datasets['train'][0]}")
    # raw_datasets={'prompt':"...", 'scores':[1,0.9,0.8...], 'response_0':"r0", 'response_1':"r1", ... , 'response_7':"r7"}

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Score sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['scores']}")
        logger.info(f"Best response sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['response_0']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate NDCG trainer
    #########################
    trainer = NDCGTrainer(
        model=model,
        ref_model=ref_model, # pass in to bypass DPO Trainer check for ref model but is not actually used
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
        data_collator = ListDataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id
            ),
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook","ndcg"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
