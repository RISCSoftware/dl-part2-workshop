import logging
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def load_qwen2_and_generate():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "Qwen/Qwen2.5-0.5B"
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # model_name = "Qwen/Qwen2.5-1.5B"

    device = "cpu"  # the device to load the model onto
    device_map = "cpu"  # "auto"
    cache_dir = "/host/hf_transformers_cache"

    logger.info(f"Loading model {model_name} onto {device}")
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
    max_new_tokens = 128

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    logger.info("Tokenizing and generating response")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=max_new_tokens,
    )
    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info(f"Response: {response}")


def main():
    min_sleep_time = 5
    max_sleep_time = 15
    sleep_time = random.randint(min_sleep_time, max_sleep_time)
    logger.info(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)
    load_qwen2_and_generate()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    load_qwen2_and_generate()
