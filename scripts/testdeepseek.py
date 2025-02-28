import logging
import os
import time
from typing import Tuple
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Setup logging
logger = logging.getLogger(__name__)
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv #pip install python-dotenv
import torch

load_dotenv()  # Load environment variables from .env

#token = os.getenv("HUGGING_FACE_TOKEN")

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

def load_model(model_name) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the model and tokenizer with 8-bit quantization configuration to optimize memory usage
    and inference performance.

    The quantization is performed using the `BitsAndBytesConfig`:
    - `load_in_8bit=True` ensures that the model weights are loaded in 8-bit precision (INT8),
        reducing the model's memory requirements.
    - `llm_int8_threshold=6.0` specifies a threshold for applying 8-bit quantization.
    Weights with magnitudes larger than this threshold will be quantized to 8-bit precision,
    while smaller weights may remain in higher precision to retain accuracy.

    Returns:
        Tuple: The tokenizer and model objects.
    """
    logger.info(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the quantization configuration for 8-bit
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  #! Dynamically balancing between CPU and GPU
        quantization_config=quantization_config,  #! Quantization
    )

    logger.info(f"Model ({model_name}) loaded.")
    return tokenizer, model


def generate_chat_response(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_length: int = 2000,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Tuple[str, str]:
    """
        Generate a response from the model based on the input prompt.

        Args:
            - prompt (str): The input prompt.
            - tokenizer (AutoTokenizer): The tokenizer to preprocess the input.
            - model (AutoModelForCausalLM): The model used for generating the response.
            - max_length (int): The maximum length of the generated output.
            - temperature (float): The randomness of the output.
            - top_k (int): The number of top token choices.
            - top_p (float): The cumulative probability threshold for nucleus sampling.

        Returns:
            Tuple[str, str]: The thinking steps and the final answer from the model.

    .   #* About temp, top_k, top_p
        Temperature controls the randomness of the generated text, with higher values
        leading to more creative but less coherent output, and lower values resulting
        in more predictable, deterministic responses.

        Top-k limits token choices to the top k most likely options, reducing irrelevant
        text but potentially limiting creativity.

        Top-p (nucleus sampling) selects tokens dynamically until a cumulative probability
        threshold is met, balancing diversity and coherence, often used in combination
        with top-k.
    """

    # Add a "thinking" instruction to the prompt
    thinking_prompt = f"""
    Question: {prompt}
    <think>
    Please reason through the problem step by step without repeating yourself. \
    Each step should be concise and progress logically toward the final answer:
    """

    # Tokenize the input prompt
    inputs = tokenizer(thinking_prompt, return_tensors="pt")

    # Move input tensors to the same device as the model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Start timing the thinking process
    start_time = time.time()

    # Generate logits and outputs
    with torch.no_grad():
        logits = model(**inputs).logits
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,  # have multi-options (tokens) picks 1 based on prob.
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        logger.info(
            f"Intermediate logits shape: {logits.shape}"
        )  # Debugging: inspect logits

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes):02}:{int(seconds):02}"

    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the "thinking" part and the final answer
    if "<think>" in full_response and "</think>" in full_response:
        thinking_part = full_response.split("<think>")[1].split("</think>")[0].strip()
        final_answer = full_response.split("</think>")[1].strip()
    else:
        thinking_part = "No thinking steps captured."
        final_answer = full_response

    # Log the thinking steps and final answer
    logger.info(f"Thinking time: {time_str}")
    logger.info(f"\nThinking Steps:\n{thinking_part}")
    logger.info(f"\nFinal Answer:\n{final_answer}")

    return thinking_part, final_answer

def main():
    """Orchestrate the loop"""
    print("Chat with DeepSeek R1! Type 'exit' to end the chat.")
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Load the model and tokenizer
    tokenizer, model = load_model(model_name)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # Generate and display the response
        thinking_output, final_output = generate_chat_response(
            prompt=user_input, tokenizer=tokenizer, model=model, max_length=2000
        )
        print(f"DeepSeek (Thinking Steps): {thinking_output}")
        print(f"DeepSeek (Final Answer): {final_output}")
        logger.info(f"Response: {thinking_output} | {final_output}")

def test():
    # Model and tokenizer names (DeepSeek-R1 70B)
    #(py312) [010796032@coe-hpc1 Develop]$ huggingface-cli download "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    #huggingface-cli download "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" #"deepseek-ai/DeepSeek-R1-Distill-Llama-70B" # or "deepseek-ai/deepseek-coder-70b-base-r1" for base model

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name) #trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16) #trust_remote_code=True
    #model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

    # Example prompt (instruct format, if using the instruct model)
    prompt = "Write a Python function to reverse a string."

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate output
    outputs = model.generate(**inputs, max_length=512, do_sample=True, top_p=0.95, top_k=40, temperature=0.8)
    print(len(outputs))
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    batch_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Print the response
    print(response)
    print(batch_response)

    #Example of instruction format if using the instruct model, more detailed prompt.
    prompt = """
    Write a python function that takes a list of integers as input and returns the sum of all even numbers in the list.

    Function signature:
    def sum_even_numbers(numbers: list[int]) -> int:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=512, do_sample=True, top_p=0.95, top_k=40, temperature=0.8)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

#If you have limited GPU memory, consider using bitsandbytes quantization:

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch

# model_name = "deepseek-ai/deepseek-coder-70b-instruct-r1"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto", trust_remote_code=True)

# prompt = "Write a Python function to reverse a string."

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(inputs, max_length=512, do_sample=True, top_p=0.95, top_k=40, temperature=0.8)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)

if __name__ == "__main__":
    main()