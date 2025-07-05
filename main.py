#!/usr/bin/env python3
"""
GRPO (Generalized Reinforcement from Preference Optimization) Training Script.
"""

import unsloth
import os
import argparse
import yaml
from typing import List, Dict, Any
from vllm import SamplingParams
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from data_preprocessing import (
    get_quality_questions,
    get_judge_input_message,
)
from data_small import build_debate_hf_datasets, evaluate_judge_response


class Config:
    """Configuration class that allows dot notation access to nested dictionaries."""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """Load configuration from YAML file, merging with base config."""
    # Load base config
    with open("configs/base.yaml", 'r') as file:
        base_config = yaml.safe_load(file)
    
    # Load override config
    with open(config_path, 'r') as file:
        override_config = yaml.safe_load(file)
    
    # Merge configs (override takes precedence)
    def merge_dicts(base, override):
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    config_dict = merge_dicts(base_config, override_config)
    config = Config(config_dict)
    
    # Set environment variables for wandb
    if os.environ.get("WANDB_ENTITY") is None and config.logging.wandb_entity is not None:
        print(f"Setting WANDB_ENTITY to {config.logging.wandb_entity}")
        os.environ["WANDB_ENTITY"] = config.logging.wandb_entity
    if os.environ.get("WANDB_PROJECT") is None and config.logging.wandb_project is not None:
        print(f"Setting WANDB_PROJECT to {config.logging.wandb_project}")
        os.environ["WANDB_PROJECT"] = config.logging.wandb_project
    
    # print config in yaml format
    print(yaml.dump(config_dict, indent=2))
    
    return config


def parse_args():
    """Parse command line arguments for config file path."""
    parser = argparse.ArgumentParser(description="GRPO Training Script for Qwen 2.5 3B")
    parser.add_argument("--config", default="configs/default.yaml",
                       help="Path to configuration YAML file")
    return parser.parse_args()


def test_reward_fn__callback(
    prompts, completions, prompt_llm_frozen, prompt_judge_info, trained_defends, answer, **kwargs
) -> List[float]:
    """
    If the reward function ends with __callback, then the **kwargs dictionary will contain
    a callback function that can be called with `inputs` and `remove_lora` as args, and it
    will return a dictionary with the following keys: 
    - prompt_ids, prompt_mask, completion_ids, completions

    We'd like to have each row of the dataset have:
    - prompt: the prompt for the model to be trained
    - prompt2: the prompt for the opponent model (inference only, frozen)
    - prompt_judge: the prompt for the judge model (inference only)
    """
    # call the frozen opponent model (through the callback)
    callback = kwargs["callback"]
    # print("prompt_llm_frozen \n", prompt_llm_frozen)
    results = callback(prompt_llm_frozen, remove_lora=False)
    opponent_completions = results["completions"]
    print(f"Opponent completions: {len(opponent_completions)=}")
    print(" First:", opponent_completions[0])

    
    # create judge prompts
    debate_text = []
    for completion, opponent_completion, trained_defend in zip(
        completions, opponent_completions, trained_defends
    ):
        if trained_defend == 1:
            debate_text.append(f"""Debater 1 said: {completion[0]['content']}
Debater 2 said: {opponent_completion[0]['content']}""".strip())
        else:
            debate_text.append(f"""Debater 1 said: {opponent_completion[0]['content']}
Debater 2 said: {completion[0]['content']}""".strip())
            
    prompts_judge = [
        get_judge_input_message(
            prompt_judge_info_i, debate_text_i
        ) for prompt_judge_info_i, debate_text_i in zip(prompt_judge_info, debate_text)
    ]
    #  call the judge model (through the callback)
    judge_results = callback(prompts_judge, remove_lora=True)
    parse_judge_response = lambda trained_defend, judge_results: (
        1.0 if str(trained_defend) == judge_results else 0.0
    )
    rewards = [
        parse_judge_response(trained_defend, judge_results["completions"][i][0]['content'])
        for i, trained_defend in enumerate(trained_defends)
    ]
    return rewards


def test_reward_fn_small_dataset__callback(
    prompts, completions, prompt_opponent, judge_prompt_template, topic, **kwargs
) -> List[float]:
    """
    If the reward function ends with __callback, then the **kwargs dictionary will contain
    a callback function that can be called with `inputs` and `remove_lora` as args, and it
    will return a dictionary with the following keys: 
    - prompt_ids, prompt_mask, completion_ids, completions

    We'd like to have each row of the dataset have:
    - prompt: the prompt for the model to be trained
    - prompt2: the prompt for the opponent model (inference only, frozen)
    - prompt_judge: the prompt for the judge model (inference only)
    """
    # call the frozen opponent model (through the callback)
    callback = kwargs["callback"]
    results = callback(prompt_opponent, remove_lora=True)
    opponent_completions = results["completions"]
    print(f"Opponent completions: {len(opponent_completions)=}")

    # create judge prompts
    judge_prompts = []
    for completion_i, opponent_completion_i, topic_i in zip(
        completions, opponent_completions, topic
    ):
        x = judge_prompt_template[0]
        x[-1]["content"] = x[-1]["content"].format(
            topic=topic_i,
            arg1_response=completion_i[0]['content'],
            arg2_response=opponent_completion_i[0]['content']
        )
        judge_prompts.append(x)

    # call the judge model (through the callback)
    judge_results = callback(judge_prompts, remove_lora=True)
    rewards = [
        1.0 if evaluate_judge_response(judge_completion_i[-1]['content']) is True else 0.0
        for judge_completion_i in judge_results["completions"]
    ]
    return rewards


def test_inference(model, tokenizer, prompt: str = "How many r's are in strawberry?"):
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
    ], tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text
    
    print("Output without LoRA:")
    print(output)

    print("Testing model with LoRA...")
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
    ], tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text
    
    print("Output with LoRA:")
    print(output)
    return output


def print_dataset_stats(dataset, tokenizer):
    print(f"Dataset size: {len(dataset)}")
    def get_seqlen(x, col):
        tokens = tokenizer.apply_chat_template(
            x[col], tokenize=False, add_generation_prompt=True,
        )
        return len(tokens)
    if "prompt_judge_without_debate" in dataset.column_names:
        cols = ['prompt', 'prompt_llm_frozen', 'prompt_judge_without_debate']
    else:
        cols = ["prompt"]
    for col in cols:
        dataset = dataset.map(lambda x: {f"{col}_seqlen": get_seqlen(x, col)})
    print(dataset.column_names)
    for col in cols:
        x = dataset[f"{col}_seqlen"]
        print(f"{col}_seqlen: {x}")
        print(f"  min: {min(x)}, max: {max(x)}, mean: {sum(x) / len(x):.1f}")


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Model configuration from config
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        fast_inference=config.model.fast_inference,
        max_lora_rank=config.model.lora_rank,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
    )
    
    print("Setting up PEFT model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        target_modules=config.model.lora_target_modules,  # Remove QKVO if out of memory
        lora_alpha=config.model.lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )

    # Data preparation
    print("Loading dataset...")
    if config.data.dataset_name == "quality_questions":
        dataset = get_quality_questions(config.data.dataset_split)
        print_dataset_stats(dataset, tokenizer)
        reward_funcs = [
            test_reward_fn__callback,
        ]
    elif config.data.dataset_name == "small_debate_dataset":
        dataset, _ = build_debate_hf_datasets()  # ignore test dataset
        print_dataset_stats(dataset, tokenizer)
        reward_funcs = [
            test_reward_fn_small_dataset__callback,
        ]
    else:
        raise ValueError(f"Invalid dataset name: {config.data.dataset_name}")

    # Training configuration
    print("Setting up training configuration...")
    training_args = GRPOConfig(
        use_vllm=config.training.use_vllm,
        max_steps=config.training.max_steps,
        # optimizer
        optim=config.training.optim,
        learning_rate=float(config.training.learning_rate),
        lr_scheduler_type=config.training.lr_scheduler_type,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        # batchsize
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        # group size
        num_generations=config.training.num_generations,
        # sequence length
        max_prompt_length=config.training.max_prompt_length,
        max_completion_length=config.training.max_completion_length,
        # logging
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        output_dir=config.training.output_dir,
        beta=config.training.beta,
        report_to=config.logging.report_to,
        run_name=config.logging.wandb_name,
        log_completions=config.logging.log_completions,
    )

    # Training
    print("Starting training...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Save the trained LoRA
    print("Saving LoRA...")
    model.save_lora("grpo_saved_lora")
    
    # Optional inference testing
    if not config.inference.skip_inference:
        # Inference - test model without LoRA
        print("Testing model without LoRA...")
        test_inference(model, tokenizer, prompt="How many r's are in strawberry?")
    
    # Optional: Save model in different formats
    # Uncomment the following lines if you want to save the model
    
    # Save merged model to 16bit
    # model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    
    # Save merged model to 4bit  
    # model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
    
    # Save just LoRA adapters
    # model.save_pretrained("model")
    # tokenizer.save_pretrained("model")


if __name__ == "__main__":
    main()