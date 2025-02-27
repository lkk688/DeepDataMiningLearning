# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")


# Model and tokenizer names (DeepSeek-R1 70B)
#(py312) [010796032@coe-hpc1 Develop]$ huggingface-cli download "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#huggingface-cli download "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" #"deepseek-ai/DeepSeek-R1-Distill-Llama-70B" # or "deepseek-ai/deepseek-coder-70b-base-r1" for base model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name) #trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16) #trust_remote_code=True

# Example prompt (instruct format, if using the instruct model)
prompt = "Write a Python function to reverse a string."

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
outputs = model.generate(**inputs, max_length=512, do_sample=True, top_p=0.95, top_k=40, temperature=0.8)

# Decode the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the response
print(response)

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