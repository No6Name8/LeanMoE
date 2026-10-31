"""Task-level answer correctness on the real model, in both execution modes.

    set LEANMOE_MODELS_ROOT=C:\\path\\to\\models
    python -m eval.task_benchmark --pilot              # the pre-registered pilot tasks
    python -m eval.task_benchmark                      # all tasks
    python -m eval.task_benchmark --summarize results\\task_benchmark_full_<stamp>.json

The tasks, expected answers, scoring rules, token limit, time limit and run
order are fixed in configs/tasks/task_benchmark_v1.json, written before any
run; each result file records that file's SHA-256. Scoring rules live in
eval/task_scoring.py.

Each (task, mode) is a fresh child process, one at a time: it loads the model
(timed separately), answers the task once ("first request"), then answers the
same prompt again in the same process ("warm repeat"). The two are scored and
timed separately and never averaged together. Modes: "cached" is LeanMoE
(expert cache + adaptive prefetch, cache sized from the memory budget as in
eval/config_compare.py); "mmap" is Windows paging (memory-mapped store, no
expert cache, no prefetch). A fresh process is not a cold cache: Windows can
keep memory-mapped pages in its file cache between runs.

Timing definitions:
  latency_s     one request: start of stream_generate to its final event
                (prefill + decode + end-of-sequence check); model load excluded
  load_s        load_model in the child (resident weights, cache, kernel)
  child_wall_s  the whole child process as seen by the parent (interpreter
                start, imports, tokenizer, load, both requests)
Memory: private (committed) bytes and working set, sampled every 50 ms inside
the child per phase (load, first request, warm repeat), and over the child's
whole life by the parent. Sampled peaks can miss shorter spikes.

These are elapsed times on one laptop; they are not CPU time, energy or cost.
Identical tokens between modes are reported as execution consistency only;
answer correctness comes from the scoring rules, never from token identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.engine  # noqa: E402,F401  (before numpy: sets the OpenBLAS thread default)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

TASK_FILE = REPO_ROOT / "configs" / "tasks" / "task_benchmark_v1.json"
CONFIG = REPO_ROOT / "configs" / "runtime_cache_chat.json"
RESULTS_DIR = REPO_ROOT / "results"
STOP_COMMIT_GB = 0.3
MIN_PHYSICAL_GB = 3.0  # same gate for both modes; avoid a paging storm on the 8 GB laptop
SPEC_LOCK = RESULTS_DIR / "task_benchmark_spec_lock.json"
MODE_NAMES = {"cached": "LeanMoE", "mmap": "Windows paging"}


def load_tasks() -> tuple[dict, str]:
    raw = TASK_FILE.read_bytes()
    # Git on Windows may check out LF text as CRLF; lock the content, not that
    # platform-dependent representation. The pilot's original LF hash matches.
    canonical = raw.replace(b"\r\n", b"\n")
    return json.loads(canonical.decode("utf-8")), hashlib.sha256(canonical).hexdigest()


def memory_guard_reason(before: dict, design: dict) -> str | None:
    if before["commit_available_gb"] is not None and before["commit_available_gb"] < design["memory_budget_gb"] + STOP_COMMIT_GB:
        return (f"skipped: commit headroom {before['commit_available_gb']:.2f} GB below "
                f"budget + {STOP_COMMIT_GB} GB")
    if before["physical_available_gb"] < MIN_PHYSICAL_GB:
        return (f"skipped: available physical RAM {before['physical_available_gb']:.2f} GB "
                f"below {MIN_PHYSICAL_GB:.1f} GB safety gate")
    return None


class PhaseSampler(threading.Thread):
    """Peak private memory and working set of this process, per named phase."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        import psutil

        self.proc = psutil.Process()
        self.phase = "start"
        self.peaks: dict[str, dict] = {}
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            info = self.proc.memory_info()
            p = self.peaks.setdefault(self.phase, {"peak_private_gb": 0.0, "peak_working_set_gb": 0.0, "samples": 0})
            p["peak_private_gb"] = max(p["peak_private_gb"], getattr(info, "private", info.rss) / 1e9)
            p["peak_working_set_gb"] = max(p["peak_working_set_gb"], info.rss / 1e9)
            p["samples"] += 1
            self.stop.wait(0.05)


def child(spec_path: Path) -> None:
    """One (task, mode): load, first request, warm repeat. Writes its record next to the spec."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    from dataclasses import replace

    from src.engine.benchmarking import openblas_threads, process_private_bytes
    from src.engine.runtime import generate as G
    from src.engine.runtime.streaming import CancelToken, IncrementalDecoder, eos_ids_for, stream_generate
    from src.engine.store.chat_template import ChatTemplate
    from src.engine.store.tokenizer import load_tokenizer

    sampler = PhaseSampler()
    sampler.start()
    d = spec["design"]
    import numba

    record: dict = {"openblas_threads": openblas_threads(), "numba_threads": numba.get_num_threads(),
                    "kernel": "fused", "backend": "numpy", "import_private_gb": G.IMPORT_PRIVATE_BYTES / 1e9}
    rt = G.RuntimeConfig.load(CONFIG)
    rt = replace(rt, prefetch=True, int8_kernel="fused", max_seq=d["max_seq"], memory_budget_gb=d["memory_budget_gb"])
    store = rt.store_dir
    store_bytes = (store / "store.json").read_bytes()
    store_source = json.loads(store_bytes)["source"]
    record.update({"store_manifest_sha256": hashlib.sha256(store_bytes).hexdigest(),
                   "store_model": store_source["repo_id"], "store_revision": store_source["revision"]})
    tok = load_tokenizer(store)
    prompt_ids = tok.encode(ChatTemplate.for_store(store).user_prompt(spec["prompt"])).ids
    eos = eos_ids_for(tok)
    cfg = G.model_config(store)
    if spec["mode"] == "cached":
        base = G.estimate_ram_bytes(store, cfg, rt.max_seq, "cached", "numpy", 0.0, "fused")
        rt = replace(rt, expert_cache_gb=(d["memory_budget_gb"] * 1e9 - base) / 1e9)
    sampler.phase = "load"
    t = time.perf_counter()
    model, info = G.load_model(store, spec["mode"], rt.max_seq, d["memory_budget_gb"], None, rt)
    record.update({
        "load_s": time.perf_counter() - t, "estimated_ram_gb": info["estimated_ram_bytes"] / 1e9,
        "cache_slots": info.get("cache", {}).get("slots", 0), "cache_gb": info.get("cache", {}).get("bytes", 0) / 1e9,
        "prefetch_active": bool(model.prefetch_enabled and model.experts.supports_prefetch),
        "private_after_load_gb": process_private_bytes() / 1e9, "prompt_tokens": len(prompt_ids),
    })  # fmt: skip
    requests = []
    out_path = Path(spec["out"])
    out_path.write_text(json.dumps(record), encoding="utf-8")  # load survives an interrupted request
    for kind in ("first_request", "warm_repeat"):
        sampler.phase = kind
        cancel = CancelToken()
        timer = threading.Timer(d["hard_cap_s"], cancel.cancel)
        timer.start()
        stats = {"cache_hits": 0, "demand_reads": 0, "prefetch_hits": 0, "expert_read_seconds": 0.0}
        done = None
        forward_seconds = []
        for e in stream_generate(model, prompt_ids, d["max_new_tokens"], d["max_seq"], eos_ids=eos,
                                 decoder=IncrementalDecoder(tok), cancel=cancel):  # fmt: skip
            if e.kind == "token":
                forward_seconds.append(e.step_seconds)
                for k in stats:
                    stats[k] += e.step_stats.get(k, 0)
            else:
                done = e
        timer.cancel()
        requests.append({
            "kind": kind, "stop_reason": done.stop_reason, "error": done.error, "latency_s": done.total_seconds,
            "time_to_first_token_s": done.time_to_first_token_s, "decode_tokens_per_s": done.decode_tokens_per_s,
            "tokens_generated": len(done.token_ids), "token_ids": done.token_ids, "text": done.text,
            "private_after_gb": process_private_bytes() / 1e9, "expert_stats": stats,
            "prefill_forward_s": done.time_to_first_token_s,
            "decode_forward_s": sum(forward_seconds[1:]),
            "decode_wall_s": (done.total_seconds - done.time_to_first_token_s)
            if done.time_to_first_token_s is not None else None,
        })  # fmt: skip
        record["requests"] = requests
        out_path.write_text(json.dumps(record), encoding="utf-8")
    model.experts.close()
    sampler.stop.set()
    sampler.join()
    for r in requests:
        r["memory"] = sampler.peaks.get(r["kind"], {})
    record["load_memory"] = sampler.peaks.get("load", {})
    record["requests"] = requests
    out_path.write_text(json.dumps(record), encoding="utf-8")


def run_one(task: dict, mode: str, design: dict, tmp: Path, max_wall_s: float | None = None) -> dict:
    import psutil

    from eval.prefetch_ablation import TreeSampler, memory_status
    from eval.task_scoring import score

    out = tmp / f"run_{time.time_ns()}.json"
    spec = tmp / f"spec_{time.time_ns()}.json"
    prompt = task["question"] + design["prompt_suffix"]
    spec.write_text(json.dumps({"mode": mode, "prompt": prompt, "design": design, "out": str(out)}), encoding="utf-8")
    before = memory_status()
    record = {"task_id": task["id"], "category": task["category"], "mode": mode, "memory_before": before}
    guard = memory_guard_reason(before, design)
    if guard:
        record["error"] = guard
        return record
    env = dict(os.environ)
    env.pop("OPENBLAS_NUM_THREADS", None)  # the engine default (4), as pre-registered
    timeout = min(120 + 2 * design["hard_cap_s"], max_wall_s or float("inf"))
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "eval.task_benchmark", "--child", str(spec)], cwd=REPO_ROOT,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # fmt: skip
    sampler = TreeSampler(proc.pid)
    sampler.start()
    stop_reason = None
    while proc.poll() is None:
        time.sleep(1)
        if time.perf_counter() - t0 > timeout:
            stop_reason = f"child timeout after {timeout} s"
        status = memory_status()
        if status["commit_available_gb"] is not None and status["commit_available_gb"] < STOP_COMMIT_GB:
            stop_reason = f"stopped: commit headroom fell to {status['commit_available_gb']:.2f} GB"
        if stop_reason:
            for p in [*psutil.Process(proc.pid).children(recursive=True), proc]:
                try:
                    p.kill()
                except Exception:
                    pass
            break
    output = proc.communicate()[0]
    sampler.stop.set()
    sampler.join()
    record.update({
        "child_wall_s": time.perf_counter() - t0, "exit_code": proc.returncode,
        "child_peak_private_gb": sampler.peak_private / 1e9, "child_peak_working_set_gb": sampler.peak_working_set / 1e9,
    })  # fmt: skip
    if stop_reason or proc.returncode != 0 or not out.exists():
        record["error"] = stop_reason or "child failed"
        record["child_output_sha256"] = hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()
        if out.exists():
            record.update(json.loads(out.read_text(encoding="utf-8")))
            record["partial_requests"] = len(record.get("requests", []))
            for q in record.get("requests", []):
                q.update(score(task, q["text"], q["stop_reason"]))
                q["within_time_limit"] = q["latency_s"] <= design["time_limit_s"]
        return record
    record.update(json.loads(out.read_text(encoding="utf-8")))
    record["error"] = None
    for r in record["requests"]:
        r.update(score(task, r["text"], r["stop_reason"]))
        r["within_time_limit"] = r["latency_s"] <= design["time_limit_s"]
    return record


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(data: dict, manual: dict | None = None) -> dict:
    """Per mode and request kind; manual review verdicts (task_id -> mode -> kind -> verdict) applied if given."""
    design, runs = data["design"], data["runs"]
    out: dict = {"by_mode": {}, "consistency": {}}
    for mode in ("cached", "mmap"):
        mruns = [r for r in runs if r["mode"] == mode]
        skipped = [r for r in mruns if str(r.get("error", "")).startswith("skipped:")]
        failed = [r for r in mruns if r.get("error") and r not in skipped]
        m = {"name": MODE_NAMES[mode], "runs": len(mruns), "failed_runs": len(failed),
             "skipped_before_inference": len(skipped),
             "load_s": [round(r["load_s"], 2) for r in mruns if "load_s" in r]}  # fmt: skip
        for kind in ("first_request", "warm_repeat"):
            reqs = [(r, q) for r in mruns for q in r.get("requests", []) if q.get("kind") == kind and "score" in q]
            verdict = []
            for r, q in reqs:
                v = q["score"]
                if v == "manual_review" and manual:
                    v = manual.get(r["task_id"], {}).get(mode, {}).get(kind, v)
                verdict.append(v)
            correct = [q for (r, q), v in zip(reqs, verdict) if v == "correct" and q["stop_reason"] == "eos"]
            lat = [q["latency_s"] for _, q in reqs]
            with_load = [q["latency_s"] + (r["load_s"] if kind == "first_request" else 0) for r, q in reqs]
            m[kind] = {
                "attempted": len(reqs) + sum(1 for r in failed if kind == "first_request" and not any(q.get("kind") == kind for q in r.get("requests", []))),
                "correct": len(correct),
                "rule_correct_incomplete": sum(1 for (_, q), v in zip(reqs, verdict) if v == "correct" and q["stop_reason"] != "eos"),
                "incorrect": verdict.count("incorrect"),
                "manual_review_open": verdict.count("manual_review"),
                "correct_within_time_limit": sum(1 for q in correct if q["within_time_limit"]),
                "completed_eos": sum(1 for _, q in reqs if q["stop_reason"] == "eos"),
                "truncated_at_token_limit": sum(1 for _, q in reqs if q["stop_reason"] == "limit"),
                "cancelled_at_hard_cap": sum(1 for _, q in reqs if q["stop_reason"] == "cancelled"),
                "total_latency_s": round(sum(lat), 1),
                "latency_s_per_correct_task": round(sum(lat) / len(correct), 1) if correct else None,
                "total_latency_plus_load_s": round(sum(with_load), 1) if kind == "first_request" else None,
                "latency_plus_load_s_per_correct_task": round(sum(with_load) / len(correct), 1)
                if kind == "first_request" and correct else None,
                "mean_latency_s": round(_mean(lat), 1) if lat else None,
                "median_latency_s": round(sorted(lat)[len(lat) // 2], 1) if lat else None,
                "mean_time_to_first_token_s": round(_mean([q["time_to_first_token_s"] for _, q in reqs if q["time_to_first_token_s"]]), 2) if reqs else None,
                "mean_decode_tokens_per_s": round(_mean([q["decode_tokens_per_s"] for _, q in reqs if q["decode_tokens_per_s"]]), 3) if reqs else None,
                "tokens_generated": sum(q["tokens_generated"] for _, q in reqs),
                "peak_private_gb_max": round(max((q["memory"].get("peak_private_gb", 0) for _, q in reqs), default=0), 3),
                "peak_working_set_gb_max": round(max((q["memory"].get("peak_working_set_gb", 0) for _, q in reqs), default=0), 3),
            }  # fmt: skip
        m["child_wall_s_total"] = round(sum(r["child_wall_s"] for r in mruns if "child_wall_s" in r), 1)
        out["by_mode"][mode] = m
    by_task: dict = {}
    for r in runs:
        if not r.get("error"):
            by_task.setdefault(r["task_id"], {})[r["mode"]] = r
    for tid, modes in by_task.items():
        if len(modes) == 2:
            ids = {(mode, q["kind"]): tuple(q["token_ids"]) for mode, r in modes.items() for q in r["requests"]}
            out["consistency"][tid] = {kind: ids.get(("cached", kind)) == ids.get(("mmap", kind))
                                          for kind in ("first_request", "warm_repeat")
                                          if ("cached", kind) in ids and ("mmap", kind) in ids}
    return out


def print_table(data: dict, summary: dict) -> None:
    for r in data["runs"]:
        if r.get("error"):
            print(f"{r['task_id']:<13} {MODE_NAMES[r['mode']]:<15} FAILED: {r['error']} {r.get('output_tail', '')}")
            continue
        for q in r["requests"]:
            print(f"{r['task_id']:<13} {MODE_NAMES[r['mode']]:<15} {q['kind']:<14} {q['score']:<14} "
                  f"{q['stop_reason']:<9} {q['tokens_generated']:>3} tok {q['latency_s']:6.1f} s "
                  f"(limit {'ok' if q['within_time_limit'] else 'OVER'}) field={q['answer_field']!r}")  # fmt: skip
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pilot", action="store_true", help="run only the pre-registered pilot tasks")
    ap.add_argument("--max-wall-s", type=int, default=3600, help="overall elapsed-time ceiling for a full run (default: 3600 seconds)")
    ap.add_argument("--summarize", type=Path, default=None, help="re-summarize a saved result file")
    ap.add_argument("--manual", type=Path, default=None, help="manual review verdicts (JSON) for --summarize")
    args = ap.parse_args()
    if args.child:
        child(args.child)
        return
    if args.summarize:
        data = json.loads(args.summarize.read_text(encoding="utf-8"))
        manual = json.loads(args.manual.read_text(encoding="utf-8"))["verdicts"] if args.manual else None
        print_table(data, summarize(data, manual))
        return
    spec, digest = load_tasks()
    lock = json.loads(SPEC_LOCK.read_text(encoding="utf-8"))
    if digest != lock["task_file_sha256"]:
        raise SystemExit("task specification differs from the frozen SHA-256; refusing inference")
    design = spec["design"]
    tasks = [t for t in spec["tasks"] if not args.pilot or t["id"] in spec["pilot"]]
    schedule = []
    for i, t in enumerate(tasks):  # balanced order: LeanMoE first at even positions of this run's list
        schedule += [(t, m) for m in (("cached", "mmap") if i % 2 == 0 else ("mmap", "cached"))]
    label = "pilot" if args.pilot else "full"
    data = {"design": design, "task_file": TASK_FILE.name, "task_file_sha256": digest, "label": label,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scorer_sha256": hashlib.sha256((REPO_ROOT / "eval" / "task_scoring.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "tasks": tasks, "runs": []}  # fmt: skip
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"task_benchmark_{label}_{stamp}.json"
    t_all = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for n, (task, mode) in enumerate(schedule, 1):
            remaining = args.max_wall_s - (time.perf_counter() - t_all)
            if remaining < 60:
                data["incomplete_reason"] = "overall time ceiling reached before the next fixed-order run"
                data["remaining_schedule"] = [{"task_id": t["id"], "mode": m} for t, m in schedule[n - 1:]]
                break
            r = run_one(task, mode, design, Path(tmp), remaining)
            r["order"] = n
            data["runs"].append(r)
            if r.get("error"):
                print(f"[{n}/{len(schedule)}] {task['id']} {MODE_NAMES[mode]}: FAILED {r['error']} {r.get('output_tail', '')}", flush=True)
            else:
                q1, q2 = r["requests"]
                print(f"[{n}/{len(schedule)}] {task['id']:<13} {MODE_NAMES[mode]:<15} load {r['load_s']:.1f}s | "
                      f"first {q1['score']} {q1['latency_s']:.1f}s {q1['tokens_generated']} tok {q1['stop_reason']} | "
                      f"warm {q2['score']} {q2['latency_s']:.1f}s | child {r['child_wall_s']:.1f}s | "
                      f"field {q1['answer_field']!r}", flush=True)  # fmt: skip
            data["elapsed_s"] = time.perf_counter() - t_all
            out.write_text(json.dumps(data, indent=1), encoding="utf-8")  # after every run: nothing lost on a stop
    data["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data["summary"] = summarize(data)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print_table(data, data["summary"])
    print(f"Total elapsed {data['elapsed_s']:.0f} s. Full log: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
