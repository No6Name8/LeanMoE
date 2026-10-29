import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924"

def ram_used_gb():
    return psutil.Process().memory_info().rss / (1024**3)

print(f"RAM before load: {ram_used_gb():.2f} GB")
t0 = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    t1 = time.time()
    print(f"Loaded in {t1 - t0:.1f}s")
    print(f"RAM after load: {ram_used_gb():.2f} GB")

    inputs = tokenizer("The capital of France is", return_tensors="pt")
    t2 = time.time()
    out = model.generate(**inputs, max_new_tokens=10)
    t3 = time.time()
    print(f"Generated: {tokenizer.decode(out[0])}")
    print(f"Generation time: {t3 - t2:.1f}s")
    print(f"RAM peak during gen: {ram_used_gb():.2f} GB")

except Exception as e:
    print(f"FAILED: {e}")
