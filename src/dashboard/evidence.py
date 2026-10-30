"""Recorded benchmark figures for the dashboard, computed from the committed result files.

LeanMoE addition. Nothing here is measured live: every figure is read from a
file in results/ when status is requested, and carries that file's name, so
the page can show it as recorded evidence. A missing or unreadable file is
reported as unavailable, never replaced with a previous figure.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

PRIMARY_UNCUT = "adaptive_demo_20261030-060626.json"
PRIMARY_CUT = "adaptive_demo_20261030-060728.json"
MMAP_FULL = "config_compare_mmap_full_20261030-060448.json"
PREFETCH_11 = "prefetch_ablation_20261029-215951.json"


def _load(results: Path, name: str) -> dict | None:
    try:
        data = json.loads((results / name).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _card(label, files, *, additional=False, methodology="", limitations="", commands=()):
    return {"label": label, "files": files, "additional": additional,
            "value": "unavailable - source artifacts missing or invalid",
            "methodology": methodology, "limitations": limitations, "commands": list(commands)}


PY = r".\.venv\Scripts\python.exe"


def _matched(runs, left, right, field="mode"):
    # Pair the nth occurrence of each mode within the same prompt (alternating order).
    pairs = []
    for prompt in dict.fromkeys(r["prompt_id"] for r in runs):
        group = [r for r in runs if r["prompt_id"] == prompt]
        a = [r for r in group if r[field] == left]
        b = [r for r in group if r[field] == right]
        if not a or len(a) != len(b):
            raise ValueError("incomplete matched experiment")
        pairs.extend(zip(a, b))
    if not pairs or any(r.get("error") or not r["generated_ids"] for p in pairs for r in p):
        raise ValueError("failed or empty run")
    return pairs


def _speed(pairs):
    if any(not math.isfinite(r["decode_tokens_per_s"]) or r["decode_tokens_per_s"] <= 0 for p in pairs for r in p):
        raise ValueError("invalid speed")
    a, b = zip(*pairs)
    ma = statistics.mean(r["decode_tokens_per_s"] for r in a)
    mb = statistics.mean(r["decode_tokens_per_s"] for r in b)
    if not (ma > 0 and mb > 0):
        raise ValueError("invalid speed")
    wins = sum(x["decode_tokens_per_s"] > y["decode_tokens_per_s"] for x, y in pairs)
    same = all(x["generated_ids"] == y["generated_ids"] for x, y in pairs)
    return ma, mb, wins, same


def recorded_evidence(results: Path) -> list[dict]:
    from eval.compare_artifacts import check_pair

    budget = _card("A smaller budget, the same output", [PRIMARY_UNCUT, PRIMARY_CUT],
        methodology="One prompt, greedy decoding, cached + adaptive prefetch, 4 OpenBLAS threads. "
                    "Only the budget-cut schedule differs. Private memory is the per-token sample at the cut. "
                    "The strict saved-artifact checker validates completion, matching settings and tokens, and cache shrink.",
        limitations="One prompt and configuration family. The budget is a planning target, not a hard ceiling. "
                    "Memory sampled every 50 ms can miss spikes. Starting cache sizes differ with available RAM. "
                    "The oversized fixed-cache run is a startup refusal, not an out-of-memory crash or speed baseline.",
        commands=[PY + " eval/compare_artifacts.py results/" + PRIMARY_UNCUT + " results/" + PRIMARY_CUT,
                  "$env:OPENBLAS_NUM_THREADS = '4'", PY + " -m eval.adaptive_demo --cut-at 0", PY + " -m eval.adaptive_demo --cut-at 20"])
    mmap = _card("LeanMoE versus Windows paging", [MMAP_FULL],
        methodology="Three fixed prompts, two runs per mode per prompt, 48 new tokens, 4 OpenBLAS threads; "
                    "same int8 store and fused kernel. Fresh processes in alternating order; cached mode has a 3.0 GB budget. "
                    "Speedup is the ratio of arithmetic mean decode rates, not the mean of pair ratios. "
                    "First-token figures are medians; source timings exclude model loading.",
        limitations="One laptop, six matched pairs. Windows file cache was never cleared, so fresh processes are not cold-cache runs. "
                    "Background load and run order matter; one warm paging run was faster. "
                    "This compares the whole cached configuration and does not isolate prefetch.",
        commands=[PY + " -m eval.config_compare --stage mmap --prompts fact code explain --rounds 2 --tokens 48 --budget-gb 3.0 --blas 4"])
    prefetch = _card("Prefetch on versus off", [PREFETCH_11], additional=True,
        methodology="Three prompts, two runs each, 48 tokens, alternating fresh processes, fixed 1.1 GB cache (126 experts), 12 OpenBLAS threads.",
        limitations="Older thread configuration; not remeasured with the current 4-thread default. "
                    "The smaller 0.6 GB exploratory cache was noisy. ExpertRelay previously found a different ordering on its base store.",
        commands=["$env:OPENBLAS_NUM_THREADS = '12'", PY + " -m eval.prefetch_ablation --cache-gb 1.1 --tokens 48 --rounds 2"])
    blas_files = ["config_compare_blas_explain_20261030-053657.json", "config_compare_blas_confirm_20261030-054506.json"]
    blas = _card("Four versus twelve OpenBLAS threads", blas_files, additional=True,
        methodology="Three prompts, two runs per configuration, 48 tokens, 3.0 GB budget. First-token medians are computed across the six runs of each configuration.",
        limitations="Changes threads and cache capacity together. Decode effects vary by prompt; no isolated threading speedup established. First-token outliers remain included. The explicit baseline thread option is needed because the engine now defaults to 4 threads.",
        commands=[PY + " -m eval.config_compare --stage blas --baseline-blas 12 --prompts fact code explain --rounds 2 --tokens 48 --budget-gb 3.0"])
    try:
        if check_pair(results / PRIMARY_UNCUT, results / PRIMARY_CUT)[0]:
            c = _load(results, PRIMARY_CUT)["adaptive"]
            ev = next(e for e in c["budget_events"] if e["token"] == c["cut_schedule"]["cut_at"])
            row = c["per_token"][ev["token"] - 1]
            budget["value"] = (f"{c['settings']['start_budget_gb']:g} to {ev['budget_gb']:g} GB; "
                               f"{ev['slots_before']} to {ev['slots_after']} cached experts. "
                               f"All {len(c['generated_ids'])} tokens match. "
                               f"Private memory at the cut: {row['process_private_mb'] / 1000:.3f} GB.")
    except (KeyError, TypeError, ValueError, IndexError, StopIteration):
        pass
    for card, name, left, right, field in [(mmap, MMAP_FULL, "cached", "mmap", "mode"),
                                          (prefetch, PREFETCH_11, True, False, "prefetch")]:
        try:
            data = _load(results, name)
            pairs = _matched(data["runs"], left, right, field)
            a, b, wins, same = _speed(pairs)
            card["value"] = (f"{a / b:.2f}x ratio of mean decode rates ({a:.3f} vs {b:.3f} tokens/s). "
                             f"Faster in {wins}/{len(pairs)} pairs. Tokens match in every pair: {'yes' if same else 'no'}.")
            if name == MMAP_FULL:
                ta = statistics.median(x["time_to_first_token_s"] for x, _ in pairs)
                tb = statistics.median(y["time_to_first_token_s"] for _, y in pairs)
                card["methodology"] += f" Recorded first-token medians: {ta:.1f} vs {tb:.1f} s."
                for key, label in [("sampled_peak_private_gb", "Sampled peak private commit"),
                                   ("sampled_peak_working_set_gb", "sampled peak working set")]:
                    a_values = [x[key] for x, _ in pairs]
                    b_values = [y[key] for _, y in pairs]
                    card["methodology"] += (f" {label}: LeanMoE {min(a_values):.2f}-{max(a_values):.2f} GB; "
                                            f"Windows paging {min(b_values):.2f}-{max(b_values):.2f} GB.")
        except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError, statistics.StatisticsError):
            pass
    try:
        runs = [r for name in blas_files for r in _load(results, name)["runs"]]
        pairs = _matched(runs, 4, "default", "blas_setting")
        a = statistics.median(x["time_to_first_token_s"] for x, _ in pairs)
        b = statistics.median(y["time_to_first_token_s"] for _, y in pairs)
        wins = sum(x["time_to_first_token_s"] < y["time_to_first_token_s"] for x, y in pairs)
        blas["value"] = f"Median first token: {a:.1f} vs {b:.1f} s; four threads faster in {wins}/{len(pairs)} pairs. Decode gains were mixed."
    except (KeyError, TypeError, ValueError, AttributeError, statistics.StatisticsError):
        pass
    return [budget, mmap, prefetch, blas]
