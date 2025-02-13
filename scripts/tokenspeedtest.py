import time
import ollama #pip install ollama
#https://github.com/ollama/ollama-python/tree/main/examples

model = "deepseek-r1:32b"  # or any other model you have
prompt = "Explain me how ollama utilize the GPU in Apple's Silicon in simple terms."

start_time = time.time()
response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
end_time = time.time()

print(response['message']['content'])
tokens = response.get("eval_count", len(response["message"]["content"].split()))
elapsed_time = end_time - start_time
tps = tokens / elapsed_time if elapsed_time > 0 else 0

print(f"Tokens: {tokens}, Time: {elapsed_time:.2f}s, Tokens per second: {tps:.2f}")
#M1Max32GB: Tokens per second: 11.75