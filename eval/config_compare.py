"""Run matched configurations of the real-model engine and record speed, memory and output.

    set LEANMOE_MODELS_ROOT=C:\\path\\to\\models
    python -m eval.config_compare --stage blas --baseline-blas 12 --prompts explain
    python -m eval.config_compare --stage mmap  --prompts fact code explain --blas 4

Stages:
  blas  Same total memory budget, cached source with adaptive prefetch, BLAS
        threads at the engine default versus 4 (use --baseline-blas 12 to repeat
        the historical 12-versus-4 configuration). The cache size follows the
        normal budget calculation (budget minus the engine's estimate of the
        rest of the process), and that estimate includes the memory committed
        at import, which depends on the BLAS thread count. This compares two
        thread-and-cache configurations together, not either effect alone.
  mmap  OS-managed memory mapping of the whole store (no expert cache, no
        prefetch; Windows pages weights in on demand) versus the cached
        configuration, both with the same BLAS thread setting.

Every run is a fresh child process (thread settings must be in the environment
before numpy is imported), one at a time, with a timeout and a stop if Windows
commit headroom drops below STOP_COMMIT_GB. A fresh process is not a cold
cache: memory-mapped pages can stay in Windows' standby file cache between
runs, while the cached source reads experts with unbuffered I/O.

Timing definitions (from the engine's per-step records):
  load_seconds      load_model: resident weights, cache allocation, kernel setup
  time_to_first_token_s   the prefill forward pass over the whole prompt, which
                    produces the first new token (tokenizing excluded)
  decode_tokens_per_s     (new tokens - 1) / time of the remaining forward passes
Memory: private (committed) and working set, sampled every SAMPLE_MS by the
parent over the child's process tree; sampled peaks can miss shorter spikes.
Disk: expert_bytes is what the engine asked for (logical); device_bytes is the
operating system's physical-disk read counter over each step, system-wide (it
includes any other program's reads in that window).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "runtime_cache_chat.json"
RESULTS_DIR = REPO_ROOT / "results"
STOP_COMMIT_GB = 0.3

PROMPTS = {  # the same three prompts as eval/prefetch_ablation.py
    "fact": "What is the capital of France? Answer in one sentence.",
    "code": "Write a Python function that checks whether a number is prime.",
    "explain": "Explain why the sky is blue in two sentences.",
}


def child(spec_path: Path) -> None:
    """One run, in this process. Writes its record next to the spec."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    from dataclasses import replace

    from src.engine.benchmarking import openblas_threads, process_private_bytes
    from src.engine.runtime import generate as G
    from src.engine.store.chat_template import ChatTemplate, is_chat_store
    from src.engine.store.tokenizer import load_tokenizer

    record: dict = {"import_private_gb": G.IMPORT_PRIVATE_BYTES / 1e9, "openblas_threads": openblas_threads()}
    rt = G.RuntimeConfig.load(CONFIG)
    rt = replace(rt, prefetch=True, int8_kernel="fused", max_new_tokens=spec["tokens"],
                 memory_budget_gb=spec["budget_gb"])  # fmt: skip
    store = rt.store_dir
    tok = load_tokenizer(store)
    template = ChatTemplate.for_store(store) if is_chat_store(store) else None
    prompt_ids = tok.encode(template.user_prompt(spec["prompt"]) if template else spec["prompt"]).ids
    cfg = G.model_config(store)
    if spec["mode"] == "cached":
        base = G.estimate_ram_bytes(store, cfg, rt.max_seq, "cached", "numpy", 0.0, "fused")
        rt = replace(rt, expert_cache_gb=(spec["budget_gb"] * 1e9 - base) / 1e9)
        record["base_estimate_gb"] = base / 1e9
    t = time.perf_counter()
    model, info = G.load_model(store, spec["mode"], rt.max_seq, spec["budget_gb"], None, rt)
    record["load_wall_seconds"] = time.perf_counter() - t
    record["load_seconds"] = info["load_seconds"]
    record["estimated_ram_gb"] = info["estimated_ram_bytes"] / 1e9
    record["private_after_load_gb"] = process_private_bytes() / 1e9
    record["cache_slots"] = info.get("cache", {}).get("slots", 0)
    record["cache_gb"] = info.get("cache", {}).get("bytes", 0) / 1e9
    record["prefetch_active"] = bool(model.prefetch_enabled and model.experts.supports_prefetch)
    import numba

    record["numba_threads"] = numba.get_num_threads()
    generated, steps = G.generate(model, prompt_ids, rt.max_new_tokens, rt.max_seq)
    model.experts.close()
    decode = steps[1:]
    dsec = sum(s.total_seconds for s in decode)
    record.update(
        {
            "prompt_tokens": len(prompt_ids),
            "time_to_first_token_s": steps[0].total_seconds,
            "decode_tokens": len(decode),
            "decode_seconds": dsec,
            "decode_tokens_per_s": len(decode) / dsec if dsec else None,
            "prefill_expert_bytes": steps[0].expert_bytes,
            "prefill_device_bytes": steps[0].device_bytes,
            "decode_expert_bytes_per_token": sum(s.expert_bytes for s in decode) / len(decode),
            "decode_device_bytes_per_token": sum(s.device_bytes for s in decode) / len(decode),
            "decode_expert_read_s_per_token": sum(s.expert_read_seconds for s in decode) / len(decode),
            "decode_compute_s_per_token": sum(s.compute_seconds for s in decode) / len(decode),
            "decode_demand_reads_per_token": sum(s.demand_reads for s in decode) / len(decode),
            "generated_ids": generated,
            "generated_text": tok.decode(generated),
        }
    )
    Path(spec["out"]).write_text(json.dumps(record), encoding="utf-8")


def run_one(mode: str, blas: int | None, prompt_id: str, args: argparse.Namespace, tmp: Path) -> dict:
    from eval.prefetch_ablation import TreeSampler, memory_status

    out = tmp / f"run_{time.time_ns()}.json"
    spec = tmp / f"spec_{time.time_ns()}.json"
    spec.write_text(json.dumps({"mode": mode, "prompt": PROMPTS[prompt_id], "tokens": args.tokens,
                                "budget_gb": args.budget_gb, "out": str(out)}), encoding="utf-8")  # fmt: skip
    env = dict(os.environ)
    env.pop("OPENBLAS_NUM_THREADS", None)
    if blas is not None:
        env["OPENBLAS_NUM_THREADS"] = str(blas)
    before = memory_status()
    record = {"mode": mode, "blas_setting": blas if blas is not None else "default", "prompt_id": prompt_id,
              "memory_before": before}  # fmt: skip
    if before["commit_available_gb"] is not None and before["commit_available_gb"] < args.budget_gb + STOP_COMMIT_GB:
        record["error"] = f"skipped: commit headroom {before['commit_available_gb']:.2f} GB below budget + margin"
        return record
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "eval.config_compare", "--child", str(spec)], cwd=REPO_ROOT,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # fmt: skip
    sampler = TreeSampler(proc.pid)
    sampler.start()
    stop_reason = None
    while proc.poll() is None:
        time.sleep(1)
        if time.perf_counter() - t0 > args.timeout:
            stop_reason = f"timeout after {args.timeout} s"
        status = memory_status()
        if status["commit_available_gb"] is not None and status["commit_available_gb"] < STOP_COMMIT_GB:
            stop_reason = f"stopped: commit headroom fell to {status['commit_available_gb']:.2f} GB"
        if stop_reason:
            for p in [*__import__("psutil").Process(proc.pid).children(recursive=True), proc]:
                try:
                    p.kill()
                except Exception:
                    pass
            break
    output = proc.communicate()[0]
    sampler.stop.set()
    sampler.join()
    record.update({
        "wall_seconds": time.perf_counter() - t0, "exit_code": proc.returncode,
        "sampled_peak_private_gb": sampler.peak_private / 1e9,
        "sampled_peak_working_set_gb": sampler.peak_working_set / 1e9,
        "samples": sampler.samples, "sample_interval_ms": 50,
    })  # fmt: skip
    if stop_reason or proc.returncode != 0 or not out.exists():
        record["error"] = stop_reason or "child failed"
        record["output_tail"] = output.strip().splitlines()[-4:]
        return record
    record.update(json.loads(out.read_text(encoding="utf-8")))
    record["error"] = None
    return record


def configs_for(args: argparse.Namespace) -> list[tuple[str, int | None]]:
    if args.stage == "blas":
        return [("cached", getattr(args, "baseline_blas", None)), ("cached", 4)]
    return [("mmap", args.blas), ("cached", args.blas)]


def summarize(runs: list[dict], configs: list) -> list[str]:
    lines = []
    ok = [r for r in runs if not r.get("error")]
    for pid in dict.fromkeys(r["prompt_id"] for r in runs):
        rs = [r for r in ok if r["prompt_id"] == pid]
        lines.append(f"{pid}: {len(rs)} successful runs, token ids identical across them: "
                     f"{len({tuple(r['generated_ids']) for r in rs}) <= 1}")  # fmt: skip
        for mode, blas in configs:
            sub = [r for r in rs if r["mode"] == mode and r["blas_setting"] == (blas if blas is not None else "default")]
            if not sub:
                continue
            f = lambda k: [round(r[k], 3) for r in sub]  # noqa: E731
            lines.append(
                f"   {mode:<6} blas {str(blas or 'default'):<7} decode tok/s {f('decode_tokens_per_s')} "
                f"first token s {f('time_to_first_token_s')} load s {f('load_seconds')} slots {sorted({r['cache_slots'] for r in sub})} "
                f"peak private {f('sampled_peak_private_gb')} peak WS {f('sampled_peak_working_set_gb')}"
            )
    failed = [r for r in runs if r.get("error")]
    lines.append(f"failed or stopped runs: {len(failed)} {[r['error'] for r in failed]}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stage", choices=["blas", "mmap"])
    ap.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=["explain"])
    ap.add_argument("--baseline-blas", type=int, choices=range(1, 257), default=None, metavar="N",
                    help="blas stage: explicit baseline thread count (default: library default); use 12 for the historical experiment")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--budget-gb", type=float, default=3.0, help="total memory budget for every configuration")
    ap.add_argument("--blas", type=int, default=None, help="mmap stage: BLAS threads for both modes (default: engine default)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    if args.child:
        child(args.child)
        return
    if not args.stage:
        ap.error("give --stage")
    configs = configs_for(args)
    schedule = []
    for rnd in range(args.rounds):
        for i, pid in enumerate(args.prompts):
            order = configs if (rnd + i) % 2 == 0 else configs[::-1]
            schedule += [(mode, blas, pid) for mode, blas in order]
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, (mode, blas, pid) in enumerate(schedule, 1):
            r = run_one(mode, blas, pid, args, Path(tmp))
            r["order"] = n
            runs.append(r)
            head = f"[{n}/{len(schedule)}] {pid:<8} {mode:<6} blas {str(blas or 'default'):<7}"
            if r.get("error"):
                print(f"{head} FAILED: {r['error']} {r.get('output_tail', '')}", flush=True)
            else:
                print(f"{head} decode {r['decode_tokens_per_s']:.3f} tok/s, first token {r['time_to_first_token_s']:.1f}s, "
                      f"load {r['load_seconds']:.1f}s, slots {r['cache_slots']}, openblas {r['openblas_threads']}, "
                      f"import private {r['import_private_gb']:.3f} GB, peak private {r['sampled_peak_private_gb']:.2f} GB, "
                      f"peak WS {r['sampled_peak_working_set_gb']:.2f} GB, commit free before {r['memory_before']['commit_available_gb']:.2f} GB",
                      flush=True)  # fmt: skip
    lines = summarize(runs, configs)
    print("\n".join(lines))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"config_compare_{args.label or args.stage}_{stamp}.json"
    design = {k: v for k, v in vars(args).items() if k != "child"}
    design.update({"prompt_texts": {p: PROMPTS[p] for p in args.prompts}, "config": CONFIG.name})
    out.write_text(json.dumps({"design": design, "runs": runs, "summary": lines}, indent=1, default=str), encoding="utf-8")
    print(f"Full log: {out}")


if __name__ == "__main__":
    main()
