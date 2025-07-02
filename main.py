#!/usr/bin/env python3
"""
GRPO (Generalized Reinforcement from Preference Optimization) Training Script
Fine-tunes Qwen 2.5 3B Instruct model using GRPO on GSM8K dataset.
"""

import os
import re
import argparse
from typing import List, Optional
import torch
from datasets import load_dataset, Dataset
from vllm import SamplingParams
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer


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
    
    # Data configuration
    parser.add_argument("--dataset-split", default="train",
                       help="Dataset split to use")
    
    # Inference configuration
    parser.add_argument("--skip-inference", action="store_true",
                       help="Skip inference testing")
    
    return parser.parse_args()


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
    dataset = get_gsm8k_questions(args.dataset_split)

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
        report_to="none",  # Can use Weights & Biases
        output_dir=args.output_dir,
    )

    # Training
    print("Starting training...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
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

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML format."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> Optional[str]:
    """Extract answer from hash format."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split: str = "train") -> Dataset:
    """Load and prepare GSM8K dataset."""
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """Reward function based on correctness of the answer."""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", 
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if answer is a digit."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks strict XML format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks soft XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text: str) -> float:
    """Count XML tags and return reward score."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    """Reward function based on XML tag counting."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


if __name__ == "__main__":
    main()