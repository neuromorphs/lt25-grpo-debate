#!/usr/bin/env python3
"""
GRPO (Generalized Reinforcement from Preference Optimization) Training Script.
"""

import unsloth
import argparse
from typing import List, Any, Callable
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from config import Config, load_config
from dataloaders.quality_questions import (
    get_quality_questions,
    get_judge_input_message,
)
from dataloaders.small_debate import (
    build_debate_hf_datasets,
    evaluate_judge_response,
)
from utils import test_inference, print_dataset_stats


def parse_args():
    """Parse command line arguments for config file path."""
    parser = argparse.ArgumentParser(description="GRPO Training Script for Qwen 2.5 3B")
    parser.add_argument("--config", default="configs/default.yaml",
                       help="Path to configuration YAML file")
    return parser.parse_args()


def make_length_reward_fn(config: Config) -> Callable[[List[str]], List[float]]:
    """
    Make a reward function that rewards the model for generating a response that is shorter than a given length.
    """
    def length_reward_fn(prompts: List[Any], completions: List[Any], **kwargs) -> List[float]:
        maxlen = config.reward.max_completion_length
        lengths = [len(completion[-1]['content']) for completion in completions]
        rewards = [
            1.0 if length <= maxlen else 1.0 - (length - maxlen) / maxlen
            for length in lengths
        ]
        rewards = [r * config.reward.length_reward_weight for r in rewards]
        return rewards
    return length_reward_fn


def make_reward_fn(config: Config) -> Callable[[List[str]], List[float]]:
    """
    Make a reward function for either the quality questions dataset or the small debate dataset.
    """

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
        remove_lora = config.reward.compare_against_reference
        results = callback(prompt_opponent, remove_lora=remove_lora)
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

    if config.data.dataset_name == "quality_questions":
        return test_reward_fn__callback
    elif config.data.dataset_name == "small_debate_dataset":
        return test_reward_fn_small_dataset__callback
    else:
        raise ValueError(f"Invalid dataset name: {config.data.dataset_name}")


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
        reward_fn__callback = make_reward_fn(config)
        reward_funcs = [
            reward_fn__callback,
        ]
        trainer_dataset_kwargs = {
            "train_dataset": dataset,
        }
    elif config.data.dataset_name == "small_debate_dataset":
        train_dataset, test_dataset = build_debate_hf_datasets(test_size=config.data.test_size)
        print_dataset_stats(train_dataset, tokenizer)
        reward_fn__callback = make_reward_fn(config)
        reward_funcs = [
            reward_fn__callback,
            make_length_reward_fn(config),
        ]
        trainer_dataset_kwargs = {
            "train_dataset": train_dataset,
            "eval_dataset": test_dataset,
        }
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
        # eval
        per_device_eval_batch_size=config.eval.batch_size,
        eval_strategy=config.eval.eval_strategy,
        eval_steps=config.eval.eval_steps,
        # logging
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        output_dir=config.training.output_dir,
        beta=config.training.beta,
        report_to=config.logging.report_to,
        run_name=config.logging.wandb_name,
        log_completions=config.logging.log_completions,
        num_completions_to_print=config.logging.num_completions_to_print,
    )

    # Training
    print("Starting training...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        **trainer_dataset_kwargs,
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