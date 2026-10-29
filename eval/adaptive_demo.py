"""Adaptive-RAM demo: cut the memory budget mid-generation and keep going.

Part 1 (adaptive): load the real int8 Chat store with an expert cache,
prefetch on and the fused kernel, write a starting budget to a budget file,
and generate. After --cut-at tokens the budget file is rewritten to a smaller
value, as if another program suddenly needed RAM. The engine's LiveBudget
(src/engine/runtime/auto.py) re-reads that file after every token and resizes
the expert cache. Every token is logged: budget, cache size, process RAM,
free system RAM.

Part 2 (oversized fixed cache): start a fresh engine run with the same fixed
cache as part 1 started with, but under the cut budget, with no budget file.
The engine's memory guard refuses it at startup because its estimate exceeds
the budget. This is a startup refusal of an oversized configuration, not an
out-of-memory crash and not a speed baseline; a smaller fixed cache can fit.

How the budget file works (read from auto.py / generate.py, not assumed):
  - it holds one number: the memory budget in GB for the WHOLE engine process,
    not the cache. The cache gets budget - base RAM (everything except the
    cache, ~2 GB for this store), further capped by free RAM + the current
    cache - a safety margin (max(512 MiB, 5% of total RAM));
  - it is re-read after every forward call and applied when its text changes,
    so the starting value is applied right after the first token;
  - shrinking cancels queued prefetches, waits for reads in flight, then drops
    free slots and least-recently-used experts. Below the minimum cache
    (top_k + 8 + 1 experts) it keeps the minimum instead of failing.

The budget file is written from this process at a token boundary. LiveBudget
only ever looks at the file between tokens, so a write from another process
at any moment during token N would take effect at the same point.

    set LEANMOE_MODELS_ROOT=C:\\Users\\h\\Documents\\ExpertRelay\\models
    python -m eval.adaptive_demo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import psutil

from eval.compare_artifacts import compare_runs
from src.engine import REPO_ROOT
from src.engine.cache.expert_cache import CachedExpertSource
from src.engine.manager.profile import measure_memory
from src.engine.runtime.auto import LiveBudget, store_facts
from src.engine.runtime.generate import RuntimeConfig, generate, load_model
from src.engine.store.chat_template import ChatTemplate, is_chat_store
from src.engine.store.tokenizer import load_tokenizer

GB = 1e9
CONFIG = REPO_ROOT / "configs" / "runtime_cache_chat.json"
RESULTS_DIR = REPO_ROOT / "results"
PROMPT = (
    "Explain step by step how a computer moves data between CPU registers, cache, "
    "RAM and an SSD, and why each level is faster but smaller than the next."
)


def process_mb() -> tuple[float, float]:
    """(working set, private bytes) of this process in MB. Private bytes is what
    the process has committed; the working set is what is in physical RAM now."""
    info = psutil.Process().memory_info()
    return info.rss / 1e6, getattr(info, "private", info.rss) / 1e6


SAMPLE_MS = 50


class PhaseSampler(threading.Thread):
    """Samples this process's private memory and working set every SAMPLE_MS,
    keeping the sampled peak per phase (load, prefill, decode, resize). A
    sampled peak can miss allocations shorter than the interval."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.phase = "load"
        self.peaks: dict[str, dict] = {}
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def run(self) -> None:
        while not self._stop.is_set():
            info = self._proc.memory_info()
            p = self.peaks.setdefault(self.phase, {"samples": 0, "peak_private_gb": 0.0, "peak_working_set_gb": 0.0})
            p["samples"] += 1
            p["peak_private_gb"] = max(p["peak_private_gb"], getattr(info, "private", info.rss) / GB)
            p["peak_working_set_gb"] = max(p["peak_working_set_gb"], info.rss / GB)
            self._stop.wait(SAMPLE_MS / 1000)

    def finish(self) -> dict:
        self._stop.set()
        self.join()
        return {"interval_ms": SAMPLE_MS, "phases": self.peaks}


def run_adaptive(args: argparse.Namespace) -> dict:
    rt = RuntimeConfig.load(CONFIG)
    rt = replace(
        rt,
        expert_cache_gb=args.start_cache_gb,
        prefetch=True,
        int8_kernel="fused",
        max_new_tokens=args.tokens,
        memory_budget_gb=args.start_gb,
    )
    store_dir = rt.store_dir
    if not (store_dir / "store.json").exists():
        sys.exit(f"no model store at {store_dir}: set LEANMOE_MODELS_ROOT to the folder that holds it")

    tokenizer = load_tokenizer(store_dir)
    template = ChatTemplate.for_store(store_dir) if is_chat_store(store_dir) else None
    prompt_ids = tokenizer.encode(template.user_prompt(PROMPT) if template else PROMPT).ids
    if len(prompt_ids) + args.tokens > rt.max_seq:
        rt = replace(rt, max_seq=len(prompt_ids) + args.tokens)

    budget_file = RESULTS_DIR / "adaptive_budget.txt"
    budget_file.write_text(f"{args.start_gb}\n")

    free_before = measure_memory().available_bytes
    sampler = PhaseSampler()
    sampler.start()
    t_load = time.perf_counter()
    model, info = load_model(store_dir, "cached", rt.max_seq, rt.memory_budget_gb, None, rt)
    load_s = time.perf_counter() - t_load
    assert isinstance(model.experts, CachedExpertSource)
    facts = store_facts(store_dir, rt, model.backend.name)
    live = LiveBudget(model, facts, measure_memory().total_bytes, budget_file)

    print(f"store {store_dir.name}, loaded in {load_s:.1f}s, {len(prompt_ids)} prompt tokens")
    print(f"base RAM estimate (everything but the cache): {facts.base_ram_bytes / GB:.2f} GB, "
          f"one expert {facts.record_bytes / 1e6:.2f} MB")
    print(f"start: cache {model.experts.num_slots} experts ({model.experts.capacity_bytes / GB:.2f} GB), "
          f"budget file {args.start_gb} GB, free RAM {free_before / GB:.2f} GB")
    print(f"{'tok':>4} {'budget':>7} {'cache':>12} {'proc WS':>8} {'private':>8} {'free':>7} {'step s':>7}")

    log: list[dict] = []
    state = {"budget": args.start_gb, "t": time.perf_counter()}

    def between_tokens() -> None:
        token = len(log) + 1
        if args.cut_at > 0 and token == args.cut_at:
            budget_file.write_text(f"{args.cut_gb}\n")
            state["budget"] = args.cut_gb
        # LiveBudget applies the starting budget after token 1 and the cut after cut_at
        sampler.phase = "resize" if token == 1 or token == args.cut_at else "decode"
        live()
        sampler.phase = "decode"
        now = time.perf_counter()
        ws, private = process_mb()
        row = {
            "token": token,
            "budget_gb": state["budget"],
            "cache_slots": model.experts.num_slots,
            "cache_gb": model.experts.capacity_bytes / GB,
            "process_working_set_mb": ws,
            "process_private_mb": private,
            "free_ram_gb": measure_memory().available_bytes / GB,
            "step_seconds": now - state["t"],
        }
        state["t"] = now
        log.append(row)
        print(f"{token:>4} {row['budget_gb']:>6.2f}G {row['cache_slots']:>4} ({row['cache_gb']:.2f}G) "
              f"{ws:>7.0f}M {private:>7.0f}M {row['free_ram_gb']:>6.2f}G {row['step_seconds']:>7.2f}",
              flush=True)

    error = None
    generated: list[int] = []
    steps = []
    sampler.phase = "prefill"
    try:
        generated, steps = generate(model, prompt_ids, rt.max_new_tokens, rt.max_seq, between_tokens=between_tokens)
    except Exception as e:  # report it, don't hide it
        error = repr(e)
    finally:
        model.experts.close()
        memory_samples = sampler.finish()

    store = json.loads((store_dir / "store.json").read_text(encoding="utf-8"))
    settings = {
        "store": store_dir.name,
        "quantization": store["quantization"],
        "backend": model.backend.name,
        "decoding": "greedy argmax, fixed length (no sampling, so no seed)",
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "chat_template": template is not None,
        "max_new_tokens": rt.max_new_tokens,
        "max_seq": rt.max_seq,
        "start_budget_gb": args.start_gb,
        "start_cache_gb": rt.expert_cache_gb,
        "int8_kernel": rt.int8_kernel,
        "io_threads": rt.io_threads,
        "pin_layer0": rt.pin_layer0,
        "prefetch": rt.prefetch,
        "prefetch_k": rt.prefetch_k,
        "prefetch_min_probability": rt.prefetch_min_probability,
        "prefetch_adaptive": rt.prefetch_adaptive,
        "prefetch_calibration": rt.prefetch_calibration.name if rt.prefetch_calibration else None,
        "prefetch_calibration_kind": rt.prefetch_calibration_kind,
        "prefill_pipelining": rt.prefill_pipelining,
    }
    return {
        "settings": settings,
        "cut_schedule": {"cut_at": args.cut_at, "cut_gb": args.cut_gb if args.cut_at > 0 else None},
        "prompt": PROMPT,
        "start_gb": args.start_gb,
        "cut_gb": args.cut_gb,
        "cut_at": args.cut_at,
        "tokens_requested": args.tokens,
        "tokens_generated": len(log),
        "error": error,
        "generated_ids": generated,
        "generated_text": tokenizer.decode(generated) if generated else "",
        "budget_events": live.events,
        "per_token": log,
        "steps": [s.__dict__ for s in steps],
        "free_ram_before_load_gb": free_before / GB,
        "load_seconds": load_s,
        "base_ram_gb": facts.base_ram_bytes / GB,
        "memory_samples": memory_samples,
    }


def run_fixed(args: argparse.Namespace) -> dict:
    """A fresh run with the same fixed cache, under the cut budget, no budget file."""
    cmd = [
        sys.executable, "-m", "src.engine.runtime.generate",
        "--config", str(CONFIG), "--source", "cached",
        "--cache-gb", str(args.start_cache_gb), "--prefetch", "--int8-kernel", "fused",
        "--memory-budget-gb", str(args.cut_gb),
        "--max-new-tokens", "1", "--prompt", PROMPT,
    ]  # fmt: skip
    free = measure_memory().available_bytes / GB
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().splitlines()
    return {
        "cache_gb": args.start_cache_gb,
        "budget_gb": args.cut_gb,
        "free_ram_gb": free,
        "exit_code": p.returncode,
        "seconds": time.perf_counter() - t0,
        "last_line": out[-1] if out else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-gb", type=float, default=3.5, help="starting process budget (GB)")
    ap.add_argument("--cut-gb", type=float, default=2.5, help="budget after the cut (GB)")
    ap.add_argument("--start-cache-gb", type=float, default=1.3, help="expert cache the run starts with (must fit the starting budget)")
    ap.add_argument("--cut-at", type=int, default=20, help="token after which the budget is cut; 0 = never (uncut reference)")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A_JSON", "B_JSON"), default=None,
                    help="don't run: compare two adaptive_demo results (settings, completion, token ids)")
    ap.add_argument("--tokens", type=int, default=80)
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare_runs(*args.compare))
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=== Part 1: adaptive cache, budget cut mid-generation ===")
    adaptive = run_adaptive(args)
    print("\n=== Part 2: fixed cache, fresh start under the cut budget ===")
    fixed = run_fixed(args)
    print(f"cache {fixed['cache_gb']} GB fixed, budget {fixed['budget_gb']} GB, free RAM {fixed['free_ram_gb']:.2f} GB")
    print(f"exit code {fixed['exit_code']} after {fixed['seconds']:.1f}s: {fixed['last_line']}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"adaptive_demo_{stamp}.json"
    out.write_text(json.dumps({"adaptive": adaptive, "fixed": fixed}, indent=1), encoding="utf-8")

    log = adaptive["per_token"]
    before = next((r for r in log if r["token"] == args.cut_at - 1), None)
    after = next((r for r in log if r["token"] == args.cut_at), None)
    print("\n=== Summary ===")
    if adaptive["error"]:
        print(f"Adaptive run FAILED after {adaptive['tokens_generated']} tokens: {adaptive['error']}")
    elif args.cut_at <= 0:
        print(f"Uncut reference: budget held at {args.start_gb}GB, cache {log[-1]['cache_slots']} experts, "
              f"{adaptive['tokens_generated']}/{args.tokens} tokens generated without error.")
    elif before and after:
        resized = after["cache_slots"] != before["cache_slots"]
        print(
            f"Budget cut from {args.start_gb}GB to {args.cut_gb}GB at token {args.cut_at}: cache "
            f"{before['cache_slots']} -> {after['cache_slots']} experts "
            f"({before['cache_gb']:.2f} -> {after['cache_gb']:.2f} GB), "
            f"{'resized' if resized else 'NOT resized'}; generation continued, "
            f"{adaptive['tokens_generated']}/{args.tokens} tokens generated without error."
        )
    else:
        print(f"Adaptive run ended after {adaptive['tokens_generated']} tokens, before the cut at {args.cut_at}.")
    for e in adaptive["budget_events"]:
        print(f"  budget event: {e}")
    print(f"Sampled peaks every {adaptive['memory_samples']['interval_ms']} ms (can miss shorter allocations):")
    for phase, p in adaptive["memory_samples"]["phases"].items():
        print(f"  {phase:<8} private {p['peak_private_gb']:.3f} GB, working set {p['peak_working_set_gb']:.3f} GB "
              f"({p['samples']} samples)")
    refused = fixed["exit_code"] != 0
    print(f"Oversized fixed cache: {'refused at startup by the memory guard' if refused else 'started'} "
          f"({fixed['last_line'][:200]})")
    print(f"Generated text: {adaptive['generated_text'][:300]!r}")
    print(f"Full log: {out}")


if __name__ == "__main__":
    main()
