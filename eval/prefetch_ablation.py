"""Prefetch ON versus OFF on the real Chat store, everything else matched.

    set LEANMOE_MODELS_ROOT=C:\\path\\to\\models
    python -m eval.prefetch_ablation

Design, fixed before any run:
  - three short prompts (PROMPTS below), 48 new tokens each, greedy;
  - two configurations that differ only in --prefetch / --no-prefetch:
    same store, fused int8 kernel, fixed expert cache (--cache-gb), I/O
    threads, pipelined prefill and calibration (configs/runtime_cache_chat.json);
    "ON" is the config's default adaptive prefetch;
  - no budget file, so the cache keeps exactly the requested size for the
    whole run (the recorded slot count is checked, not assumed);
  - two runs per configuration per prompt, the order alternating between
    rounds and prompts, each run a fresh process, one process at a time.

For every run this records load time, time to first token (the prefill
step), decode time and tokens/s, the actual cache size, cache and I/O
counters, generated token ids, and memory sampled by this parent process
every SAMPLE_MS milliseconds: private memory (committed) and working set
(in physical RAM), summed over the run's process tree. Sampled peaks can miss
allocations shorter than the interval. Free physical RAM and Windows commit
headroom are recorded before each run.

Expert reads use unbuffered I/O, which bypasses the OS file cache, but the
resident weights and the embedding are read through it; the first run of a
session can therefore load more slowly. Every run, including failures, is
kept in the output file.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "runtime_cache_chat.json"
RESULTS_DIR = REPO_ROOT / "results"
SAMPLE_MS = 50

PROMPTS = [
    {"id": "fact", "text": "What is the capital of France? Answer in one sentence."},
    {"id": "code", "text": "Write a Python function that checks whether a number is prime."},
    {"id": "explain", "text": "Explain why the sky is blue in two sentences."},
]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]  # fmt: skip


def memory_status() -> dict:
    """Free physical RAM and commit headroom (Windows; physical RAM only elsewhere)."""
    vm = psutil.virtual_memory()
    out = {"physical_available_gb": vm.available / 1e9, "commit_available_gb": None, "commit_limit_gb": None}
    if sys.platform == "win32":
        m = _MemoryStatus()
        m.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            out["commit_available_gb"] = m.ullAvailPageFile / 1e9
            out["commit_limit_gb"] = m.ullTotalPageFile / 1e9
    return out


class TreeSampler(threading.Thread):
    """Samples private memory and working set of a process and its children."""

    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.root = psutil.Process(pid)
        self.stop = threading.Event()
        self.samples = 0
        self.peak_private = 0
        self.peak_working_set = 0

    def run(self) -> None:
        while not self.stop.is_set():
            private = ws = 0
            try:
                procs = [self.root, *self.root.children(recursive=True)]
            except psutil.Error:
                procs = []
            for p in procs:
                try:
                    info = p.memory_info()
                except psutil.Error:
                    continue
                private += getattr(info, "private", info.rss)
                ws += info.rss
            self.samples += 1
            self.peak_private = max(self.peak_private, private)
            self.peak_working_set = max(self.peak_working_set, ws)
            self.stop.wait(SAMPLE_MS / 1000)


def run_one(prompt: dict, prefetch: bool, args: argparse.Namespace, workdir: Path) -> dict:
    out_json = workdir / f"{prompt['id']}_{'on' if prefetch else 'off'}_{time.time_ns()}.json"
    cmd = [
        sys.executable, "-m", "src.engine.runtime.generate",
        "--config", str(CONFIG), "--source", "cached",
        "--cache-gb", str(args.cache_gb), "--prefetch" if prefetch else "--no-prefetch",
        "--int8-kernel", "fused", "--max-new-tokens", str(args.tokens),
        "--prompt", prompt["text"], "--json-out", str(out_json),
    ]  # fmt: skip
    before = memory_status()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sampler = TreeSampler(proc.pid)
    sampler.start()
    output, _ = proc.communicate()
    wall = time.perf_counter() - t0
    sampler.stop.set()
    sampler.join()
    record = {
        "prompt_id": prompt["id"],
        "prefetch": prefetch,
        "exit_code": proc.returncode,
        "wall_seconds": wall,
        "memory_before": before,
        "sampled_peak_private_gb": sampler.peak_private / 1e9,
        "sampled_peak_working_set_gb": sampler.peak_working_set / 1e9,
        "samples": sampler.samples,
        "sample_interval_ms": SAMPLE_MS,
        "output_tail": output.strip().splitlines()[-3:],
    }
    if proc.returncode != 0 or not out_json.exists():
        record["error"] = "generation failed"
        return record
    d = json.loads(out_json.read_text(encoding="utf-8"))
    p = d["prompts"][0]
    steps = p["steps"]
    decode = steps[1:]
    dsec = sum(s["total_seconds"] for s in decode)
    sums = {k: sum(s[k] for s in decode) for k in (
        "expert_read_seconds", "compute_seconds", "expert_loads", "demand_reads", "cache_hits",
        "prefetch_hits", "prefetch_waits", "prefetches_issued", "prefetch_reads", "prefetches_wasted",
    )}  # fmt: skip
    record.update(
        {
            "error": None,
            "cache_slots": d["load"]["cache"]["slots"],
            "cache_gb": d["load"]["cache"]["bytes"] / 1e9,
            "prefetch_recorded": d["load"]["cache"]["prefetch"],
            "load_seconds": d["load"]["load_seconds"],
            "prompt_tokens": len(p["prompt_ids"]),
            "time_to_first_token_s": steps[0]["total_seconds"],
            "decode_tokens": len(decode),
            "decode_seconds": dsec,
            "decode_tokens_per_s": len(decode) / dsec if dsec else None,
            "decode_per_token": {k: v / len(decode) for k, v in sums.items()},
            "generated_ids": p["generated_ids"],
            "generated_text": p["generated_text"],
            "engine_peak_working_set_mb": d["peak_rss_mb"],
        }
    )
    return record


def summarize(runs: list[dict]) -> list[str]:
    lines = []
    ok = [r for r in runs if not r.get("error")]
    for prompt in PROMPTS:
        pid = prompt["id"]
        rs = [r for r in ok if r["prompt_id"] == pid]
        ids = {tuple(r["generated_ids"]) for r in rs}
        lines.append(f"{pid}: {len(rs)} successful runs, token ids identical across all of them: {len(ids) == 1}")
        for flag in (True, False):
            sub = [r for r in rs if r["prefetch"] == flag]
            if not sub:
                continue
            tps = [r["decode_tokens_per_s"] for r in sub]
            ttft = [r["time_to_first_token_s"] for r in sub]
            slots = sorted({r["cache_slots"] for r in sub})
            lines.append(
                f"   prefetch {'ON ' if flag else 'OFF'}: decode tok/s {[round(x, 3) for x in tps]} "
                f"mean {statistics.mean(tps):.3f}; first token s {[round(x, 1) for x in ttft]}; "
                f"cache slots {slots}; demand reads/token "
                f"{statistics.mean(r['decode_per_token']['demand_reads'] for r in sub):.1f}"
            )
    on = [r["decode_tokens_per_s"] for r in ok if r["prefetch"]]
    off = [r["decode_tokens_per_s"] for r in ok if not r["prefetch"]]
    if on and off:
        lines.append(f"all prompts: decode tok/s ON mean {statistics.mean(on):.3f} (range {min(on):.3f}-{max(on):.3f}), "
                     f"OFF mean {statistics.mean(off):.3f} (range {min(off):.3f}-{max(off):.3f})")
    failed = [r for r in runs if r.get("error")]
    lines.append(f"failed runs: {len(failed)}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-gb", type=float, default=0.6)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    schedule = []
    for rnd in range(args.rounds):
        for i, prompt in enumerate(PROMPTS):
            first_on = (rnd + i) % 2 == 0
            schedule += [(prompt, first_on), (prompt, not first_on)]

    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, (prompt, prefetch) in enumerate(schedule, 1):
            r = run_one(prompt, prefetch, args, Path(tmp))
            r["order"] = n
            runs.append(r)
            print(
                f"[{n}/{len(schedule)}] {prompt['id']:<8} prefetch {'ON ' if prefetch else 'OFF'} "
                + (f"FAILED exit {r['exit_code']}: {r['output_tail']}" if r.get("error") else
                   f"decode {r['decode_tokens_per_s']:.3f} tok/s, first token {r['time_to_first_token_s']:.1f}s, "
                   f"slots {r['cache_slots']}, peak private {r['sampled_peak_private_gb']:.2f} GB, "
                   f"commit free before {r['memory_before']['commit_available_gb']:.2f} GB"),
                flush=True,
            )

    lines = summarize(runs)
    print("\n".join(lines))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"prefetch_ablation_{stamp}.json"
    out.write_text(
        json.dumps({"design": {"prompts": PROMPTS, "cache_gb": args.cache_gb, "tokens": args.tokens,
                               "rounds": args.rounds, "config": CONFIG.name, "sample_interval_ms": SAMPLE_MS},
                    "runs": runs, "summary": lines}, indent=1),  # fmt: skip
        encoding="utf-8",
    )
    print(f"Full log: {out}")


if __name__ == "__main__":
    main()
