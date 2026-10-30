"""Bounded capability audit: real answers from the Chat model to fixed prompts.

    set LEANMOE_MODELS_ROOT=C:\\path\\to\\models
    python -m eval.capability_audit

Loads the model once (cached source, fixed expert cache, adaptive prefetch,
fused kernel) and answers PROMPTS one after another, as an application would.
Unlike the benchmark scripts, each answer stops at the model's end-of-turn
token or at MAX_NEW_TOKENS. The engine's generate() has no end-of-sequence
stop, so this script runs its own loop over model.forward with the same
greedy (argmax) decoding; it changes nothing in the engine.

For each prompt it records the complete output, why it stopped, prompt and
output token counts, time to first token (the prefill forward pass), decode
tokens/s, private memory and working set before and after the request, and
how many intermediate decoded texts contained a replacement character (what
naive token-by-token streaming would have shown). Correctness is assessed
afterwards against the EXPECTED notes, which were worked out independently.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

# The engine package must be imported before numpy: it sets the OpenBLAS thread default,
# which only takes effect if numpy hasn't been imported yet.
from src.engine import REPO_ROOT  # isort: skip

import numpy as np
import psutil

from src.engine.benchmarking import openblas_threads
from src.engine.runtime.generate import RuntimeConfig, load_model
from src.engine.runtime.qwen_moe import KVCache
from src.engine.runtime.streaming import eos_ids_for
from src.engine.store.chat_template import ChatTemplate
from src.engine.store.tokenizer import load_tokenizer

CONFIG = REPO_ROOT / "configs" / "runtime_cache_chat.json"
MAX_NEW_TOKENS = 320
MAX_SEQ = 640
CACHE_GB = 1.0

PROMPTS = [
    {"id": "arithmetic", "text": "What is 487 × 36? Give the result.",
     "expected": "17532 (487*30 = 14610, 487*6 = 2922)"},
    {"id": "linear", "text": "Solve for x: 3x + 7 = 2x - 5.", "expected": "x = -12"},
    {"id": "quadratic", "text": "Solve x^2 - 5x + 6 = 0.", "expected": "x = 2 or x = 3"},
    {"id": "word_problem",
     "text": "A train leaves at 2:15 pm and travels 180 km at a constant 72 km/h. At what time does it arrive?",
     "expected": "180/72 = 2.5 h, so 4:45 pm"},
    {"id": "latex", "text": r"Compute $\int_0^1 (3x^2 + 2x)\,dx$.", "expected": "[x^3 + x^2] from 0 to 1 = 2"},
    {"id": "code", "text": "Write a Python function fib(n) that returns the n-th Fibonacci number using a loop.",
     "expected": "a loop-based function; checked by running it (F(10) = 55 with F(0) = 0, F(1) = 1)"},
    {"id": "arabic", "text": "ما هي عاصمة اليابان؟ أجب بجملة واحدة.", "expected": "طوكيو (Tokyo)"},
    {"id": "long_context",
     "text": ("Read the passage and answer the question.\n\n"
              "The Riverside public library opened in 1998 with 12,000 books. Over the next decade it grew "
              "steadily: in 2005 the city council funded a new wing, and the library added 8,000 books that year. "
              "Membership also rose, from 1,500 cardholders at opening to 6,200 by 2010. In 2015 the library "
              "donated 3,500 of its oldest books to a school in a neighbouring town, and in 2018 it received a "
              "gift of 2,000 children's books from a local publisher. The library has not added or removed any "
              "other books.\n\nQuestion: How many books did the library have at the end of 2018?"),
     "expected": "12000 + 8000 - 3500 + 2000 = 18500"},
]  # fmt: skip


def mem() -> dict:
    i = psutil.Process().memory_info()
    return {"private_gb": getattr(i, "private", i.rss) / 1e9, "working_set_gb": i.rss / 1e9}


def answer(model, tok, template, text: str) -> dict:
    eos = eos_ids_for(tok)
    ids = tok.encode(template.user_prompt(text)).ids
    if len(ids) + MAX_NEW_TOKENS > MAX_SEQ:
        return {"error": f"prompt of {len(ids)} tokens + {MAX_NEW_TOKENS} new exceeds {MAX_SEQ}"}
    before = mem()
    cache = KVCache(model.c, MAX_SEQ)
    out: list[int] = []
    feed = ids
    times = []
    garbled = 0
    stop = "limit"
    t_start = time.perf_counter()
    for _ in range(MAX_NEW_TOKENS):
        t = time.perf_counter()
        logits, _ = model.forward(np.asarray(feed), cache)
        nxt = int(np.argmax(logits))
        times.append(time.perf_counter() - t)
        if nxt in eos:
            stop = "eos"
            break
        out.append(nxt)
        if "\ufffd" in tok.decode(out):
            garbled += 1
        feed = [nxt]
    total = time.perf_counter() - t_start
    decode_times = times[1:]
    return {
        "prompt_tokens": len(ids),
        "output_tokens": len(out),
        "stop": stop,
        "output": tok.decode(out),
        "time_to_first_token_s": times[0],
        "decode_tokens_per_s": len(decode_times) / sum(decode_times) if decode_times else None,
        "total_seconds": total,
        "intermediate_texts_with_replacement_char": garbled,
        "memory_before": before,
        "memory_after": mem(),
        "kv_cache_mb": KVCache.bytes_for(model.c, MAX_SEQ) / 1e6,
    }


def main() -> None:
    rt = RuntimeConfig.load(CONFIG)
    rt = replace(rt, expert_cache_gb=CACHE_GB, prefetch=True, int8_kernel="fused", max_seq=MAX_SEQ)
    store = rt.store_dir
    if not (store / "store.json").exists():
        sys.exit(f"no model store at {store}")
    tok = load_tokenizer(store)
    template = ChatTemplate.for_store(store)
    t = time.perf_counter()
    model, info = load_model(store, "cached", rt.max_seq, rt.memory_budget_gb, None, rt)
    load_s = time.perf_counter() - t
    print(f"loaded in {load_s:.1f}s, cache {info['cache']['slots']} experts, openblas {openblas_threads()}, {mem()}", flush=True)
    results = []
    for p in PROMPTS:
        r = answer(model, tok, template, p["text"])
        r.update({"id": p["id"], "prompt": p["text"], "expected": p["expected"]})
        results.append(r)
        if "error" in r:
            print(f"--- {p['id']}: ERROR {r['error']}", flush=True)
            continue
        print(f"--- {p['id']}: {r['output_tokens']} tokens, stop={r['stop']}, first token {r['time_to_first_token_s']:.1f}s, "
              f"decode {r['decode_tokens_per_s'] or 0:.2f} tok/s, private {r['memory_before']['private_gb']:.3f} -> "
              f"{r['memory_after']['private_gb']:.3f} GB, garbled partial texts {r['intermediate_texts_with_replacement_char']}",
              flush=True)  # fmt: skip
        print(r["output"], flush=True)
    model.experts.close()
    out = REPO_ROOT / "results" / f"capability_audit_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"settings": {"cache_gb": CACHE_GB, "max_new_tokens": MAX_NEW_TOKENS, "max_seq": MAX_SEQ,
                                            "decoding": "greedy argmax, stop at end-of-turn", "openblas_threads": openblas_threads(),
                                            "cache_slots": info["cache"]["slots"], "load_seconds": load_s},
                               "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")  # fmt: skip
    print(f"Full log: {out}")


if __name__ == "__main__":
    main()
