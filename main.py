#!/usr/bin/env python3
"""
GRPO (Generalized Reinforcement from Preference Optimization) Training Script
Fine-tunes Qwen 2.5 3B Instruct model using GRPO on GSM8K dataset.
"""

import unsloth
import os
import re
import argparse
import yaml
from typing import List, Optional, Dict, Any
import torch
from datasets import load_dataset, Dataset
from vllm import SamplingParams
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from data_preprocessing import load_quality, questions_to_datasets, get_debater_input_message, get_judge_input_message


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GRPO Training Script for Qwen 2.5 3B")
    
    # Model configuration
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct", 
                       help="Model name to use for training")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--lora-rank", type=int, default=64,
                       help="LoRA rank (higher = smarter but slower)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                       help="GPU memory utilization ratio")
    
    # Training configuration
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=250,
                       help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Per device train batch size")
    parser.add_argument("--num-generations", type=int, default=8,
                       help="Number of generations per step")
    parser.add_argument("--output-dir", default="outputs",
                       help="Output directory for training artifacts")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Beta for the KL divergence loss function")

    # Logging
    parser.add_argument("--wandb-project", default="grpo-debate",
                       help="Wandb project name")
    parser.add_argument("--wandb-entity", default="rug-minds",
                       help="Wandb entity name")
    parser.add_argument("--wandb-name", default="grpo-quality-questions",
                       help="Wandb run name")
    parser.add_argument("--log-completions", action="store_true",
                       help="Log completions to wandb")

    # Reward configuration
    parser.add_argument("--compare-against-reference", action="store_true",
                       help="Compare against reference model")

    # Data configuration
    parser.add_argument("--dataset-split", default="train",
                       help="Dataset split to use")
    
    # Inference configuration
    parser.add_argument("--skip-inference", action="store_true",
                       help="Skip inference testing")
    
    args = parser.parse_args()

    if os.environ.get("WANDB_ENTITY") is None:
        print(f"Setting WANDB_ENTITY to {args.wandb_entity}")
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if os.environ.get("WANDB_PROJECT") is None:
        print(f"Setting WANDB_PROJECT to {args.wandb_project}")
        os.environ["WANDB_PROJECT"] = args.wandb_project

    return args


# Constants and helper functions
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def get_gsm8k_questions(split: str = "train") -> Dataset:
    """Load and prepare GSM8K dataset."""
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
    })
    return data


def get_quality_questions(split: str = "train") -> Dataset:
    """Load and prepare QualityQuestions dataset."""
    train_questions, test_questions = load_quality(n_questions=6)
    dataset_dict = questions_to_datasets(train_questions, test_questions)
    data = []
    for x in dataset_dict[split]:
        for trained_position in [1, 2]:
            frozen_position = 2 if trained_position == 1 else 1
            
            instance = {
                'prompt': get_debater_input_message(
                    x['question'], x['article'], trained_position, x['answer_1'], x['answer_2']
                ),
                'prompt_llm_frozen': get_debater_input_message(
                    x['question'], x['article'], frozen_position, x['answer_1'], x['answer_2']
                ),
                'prompt_judge_info': {
                    'answer_1': x['answer_1'],
                    'answer_2': x['answer_2'],
                    'question': x['question'],
                },
                'answer': x['true_answer'],
                'trained_defends': trained_position
            }
            data.append(instance)

    # Convert back to dataset format if needed
    data = Dataset.from_list(data)
    return data


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


def main():
    args = parse_args()
    # Model configuration from args
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    print("Setting up PEFT model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )

    # Data preparation
    print("Loading dataset...")
    # dataset = get_gsm8k_questions(args.dataset_split)
    dataset = get_quality_questions(args.dataset_split)

    # Training configuration
    print("Setting up training configuration...")
    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference!
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=200,
        max_steps=args.max_steps,
        save_steps=args.max_steps,
        max_grad_norm=0.1,
        output_dir=args.output_dir,
        beta=args.beta,
        report_to="wandb",  # Can use Weights & Biases
        run_name=args.wandb_name,
        log_completions=args.log_completions,
    )

    # Training
    print("Starting training...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            test_reward_fn__callback,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Save the trained LoRA
    print("Saving LoRA...")
    model.save_lora("grpo_saved_lora")
    
    # Optional inference testing
    if not args.skip_inference:
        # Inference - test model without LoRA
        print("Testing model without LoRA...")
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": "How many r's are in strawberry?"},
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How many r's are in strawberry?"},
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

    print("Training and inference completed successfully!")
    
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