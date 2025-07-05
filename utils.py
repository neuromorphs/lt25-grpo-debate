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
