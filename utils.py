from vllm import SamplingParams


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