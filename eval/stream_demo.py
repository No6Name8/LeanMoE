"""Command-line example of streaming generation with cancellation.

    .\\.venv\\Scripts\\python.exe -m eval.stream_demo
    .\\.venv\\Scripts\\python.exe -m eval.stream_demo --cancel-after-tokens 5
    .\\.venv\\Scripts\\python.exe -m eval.stream_demo --real --prompt "What is 487 x 36?"
    .\\.venv\\Scripts\\python.exe -m eval.stream_demo --real --prompt "Explain why the sky is blue." --cancel-after-seconds 3

Default: a tiny random-weight model built in a temporary folder (no download;
needs the dev requirements because it reuses the test store builder). Its
output is meaningless token ids; it shows the mechanics only. With --real it
uses the Qwen1.5-MoE-A2.7B-Chat store under LEANMOE_MODELS_ROOT and prints text.

Generation runs in a worker thread so this program can react while the model
works. Press Ctrl+C (or use --cancel-after-tokens / --cancel-after-seconds) to
request cancellation: it takes effect after the forward pass that is running,
which during the prompt's first pass (prefill) can take several seconds on the
real model. The summary reports how long the stop took and in which phase.
Because generation runs ahead in its own thread, a token that was already being
computed when the cancel arrived is still shown (e.g. --cancel-after-tokens 5 can show 6).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.engine  # noqa: E402,F401  (before numpy: sets the OpenBLAS thread default)

import argparse  # noqa: E402
import gc  # noqa: E402
import os  # noqa: E402
import queue  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

from src.engine.runtime.streaming import (  # noqa: E402
    CancelToken, ContextTooLong, IncrementalDecoder, eos_ids_for, stream_generate,
)  # fmt: skip

SYNTHETIC_CONFIG = {
    "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4,
    "num_key_value_heads": 4, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24, "norm_topk_prob": False, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
}  # fmt: skip


def synthetic_model(tmp: Path):
    sys.path.insert(0, str(REPO_ROOT / "tests" / "engine"))
    import pytest
    from store_helpers import build_test_store, random_checkpoint

    from src.engine.cache.expert_cache import CachedExpertSource
    from src.engine.predictor.prefetch_policy import PrefetchPolicy
    from src.engine.runtime.generate import model_config
    from src.engine.runtime.qwen_moe import QwenMoe
    from src.engine.runtime.weights import ResidentWeights, UnbufferedExpertSource

    mp = pytest.MonkeyPatch()
    build_test_store(tmp, SYNTHETIC_CONFIG, random_checkpoint(SYNTHETIC_CONFIG, seed=1), mp)
    mp.undo()
    record = UnbufferedExpertSource(tmp).layout.record_size
    experts = CachedExpertSource(tmp, capacity_bytes=6 * record, pinned=[], io_threads=2, min_free_slots=3)
    model = QwenMoe(model_config(tmp), ResidentWeights.load(tmp), experts, prefetcher=PrefetchPolicy(4))
    model.prefetch_enabled = True
    return model


def real_model(args):
    from src.engine.runtime.generate import RuntimeConfig, load_model
    from src.engine.store.chat_template import ChatTemplate
    from src.engine.store.tokenizer import load_tokenizer

    rt = RuntimeConfig.load(REPO_ROOT / "configs" / "runtime_cache_chat.json")
    rt = replace(rt, expert_cache_gb=args.cache_gb, prefetch=True, int8_kernel="fused", max_seq=args.max_seq)
    store = rt.store_dir
    if not (store / "store.json").exists():
        sys.exit(f"No model store at {store}. Set LEANMOE_MODELS_ROOT (see README, 'Full model'), "
                 "or run without --real for the synthetic example.")  # fmt: skip
    tok = load_tokenizer(store)
    template = ChatTemplate.for_store(store)
    ids = tok.encode(template.user_prompt(args.prompt)).ids
    eos = eos_ids_for(tok)
    print(f"loading the model (cache {args.cache_gb} GB, context {args.max_seq} tokens) ...", flush=True)
    model, info = load_model(store, "cached", rt.max_seq, rt.memory_budget_gb, None, rt)
    print(f"loaded in {info['load_seconds']:.1f}s, {info['cache']['slots']} cached experts", flush=True)
    return model, ids, eos, IncrementalDecoder(tok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="use the real Chat model store (LEANMOE_MODELS_ROOT)")
    ap.add_argument("--prompt", default="What is 487 x 36?")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-seq", type=int, default=512, help="context size (prompt + output), real model only")
    ap.add_argument("--cache-gb", type=float, default=1.0, help="expert cache, real model only")
    ap.add_argument("--cancel-after-tokens", type=int, default=None)
    ap.add_argument("--cancel-after-seconds", type=float, default=None)
    args = ap.parse_args()

    # ignore_cleanup_errors: a memory-mapped weight file can outlive the model briefly on Windows
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        if args.real:
            model, prompt_ids, eos, decoder = real_model(args)
            max_seq = args.max_seq
        else:
            if sys.platform != "win32":
                print("The engine needs Windows (Win32 unbuffered reads).")
                return 1
            model = synthetic_model(Path(tmp) / "store")
            prompt_ids, eos, decoder, max_seq = [3, 9, 27, 17, 51], set(), None, 64
            args.max_new_tokens = min(args.max_new_tokens, max_seq - len(prompt_ids))
            print("synthetic model (random weights): token ids only, no meaningful text")

        cancel = CancelToken()
        events: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                for e in stream_generate(model, prompt_ids, args.max_new_tokens, max_seq,
                                         eos_ids=eos, decoder=decoder, cancel=cancel):  # fmt: skip
                    events.put(e)
            except ContextTooLong as e:
                events.put(e)

        if args.cancel_after_seconds is not None:
            threading.Timer(args.cancel_after_seconds, cancel.cancel).start()
        t0 = time.perf_counter()
        threading.Thread(target=worker, daemon=True).start()
        print(f"prompt: {len(prompt_ids)} tokens; generating (Ctrl+C to cancel) ...\n", flush=True)
        done = None
        while done is None:
            try:
                item = events.get(timeout=0.1)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                print("\n[cancel requested; stops after the current forward pass]", flush=True)
                cancel.cancel()
                continue
            if isinstance(item, ContextTooLong):
                print(f"Refused before allocating anything: {item}")
                model.experts.close()
                del model
                gc.collect()
                return 2
            if item.kind == "token":
                print(item.text_delta if decoder else f"{item.token_id} ", end="", flush=True)
                if args.cancel_after_tokens is not None and item.index + 1 >= args.cancel_after_tokens:
                    cancel.cancel()
            else:
                done = item
                print(done.text_delta, end="")
        model.experts.close()
        del model  # release the memory-mapped weights before the temporary folder is removed
        gc.collect()

    print("\n")
    print(f"stop reason        {done.stop_reason}" + (f" ({done.error})" if done.error else ""))
    print(f"prompt / output    {done.prompt_tokens} / {len(done.token_ids)} tokens")
    if done.time_to_first_token_s is not None:
        print(f"first token        {done.time_to_first_token_s:.2f} s (the prefill forward pass)")
    if done.decode_tokens_per_s is not None:
        print(f"decode             {done.decode_tokens_per_s:.2f} tokens/s")
    print(f"total              {done.total_seconds:.2f} s (wall {time.perf_counter() - t0:.2f} s)")
    if done.stop_reason == "cancelled":
        delay = f"{done.cancel_delay_s:.2f} s" if done.cancel_delay_s is not None else "n/a"
        print(f"cancel             took {delay} to take effect, requested during {done.cancel_phase}")
    print(f"OpenBLAS threads   {os.environ.get('OPENBLAS_NUM_THREADS')}")
    return 0 if done.stop_reason != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
