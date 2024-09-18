import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen2_and_generate():
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    # model_name = "Qwen/Qwen2-0.5B"
    # model_name = "Qwen/Qwen2-1.5B-Instruct"
    # model_name = "Qwen/Qwen2-1.5B"

    device = "cpu"  # the device to load the model onto
    device_map = "cpu"  # "auto"
    cache_dir = "/host/hf_transformers_cache"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    prompt = "Write a story including a software developer, time travel, and a dragon. Include two plot twists."
    max_new_tokens = 512

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids, max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


def main():
    min_sleep_time = 10
    max_sleep_time = 40
    sleep_time = random.randint(min_sleep_time, max_sleep_time)
    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)
    load_qwen2_and_generate()


if __name__ == "__main__":
    load_qwen2_and_generate()
