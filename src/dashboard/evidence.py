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


ARTIFACT_LABELS = {
    'adaptive_demo_20261030-060626.json': 'Uncut reference',
    'adaptive_demo_20261030-060728.json': 'Adaptive run',
    'adaptive_demo_20261029-220246.json': 'Uncut reference - corrected accounting, 12 threads',
    'adaptive_demo_20261029-220417.json': 'Adaptive run - corrected accounting, 12 threads',
    'adaptive_demo_20261029-220548.json': 'Adaptive run - 2.7 GB budget',
    'adaptive_demo_20261029-205033.json': 'Uncut reference - original accounting',
    'adaptive_demo_20261029-205224.json': 'Adaptive run - original accounting',
    'adaptive_demo_20261029-211301.json': 'Independent review - 2.2 GB budget',
    'adaptive_demo_20261029-200451.json': 'Initial adaptive run - original accounting',
    'prefetch_ablation_20261029-215951.json': 'Prefetch comparison - 1.1 GB cache',
    'prefetch_ablation_20261029-214649.json': 'Exploratory prefetch comparison - 0.6 GB cache',
    'config_compare_blas_explain_20261030-053657.json': 'Thread comparison - explanation prompt',
    'config_compare_blas_confirm_20261030-054506.json': 'Thread comparison - fact and code prompts',
    'config_compare_mmap_pilot_20261030-054758.json': 'Paging comparison - pilot',
    'config_compare_mmap_full_20261030-060448.json': 'Paging comparison',
    'capability_audit_20261030-064849.json': 'Capability audit - 12 threads',
    'dashboard_smoke_20261030-074111.json': 'Dashboard functional check',
}


def _load(results: Path, name: str) -> dict | None:
    try:
        data = json.loads((results / name).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _card(label, files, *, additional=False, limitations="", commands=()):
    return {"label": label, "files": files, "additional": additional,
            "file_labels": [ARTIFACT_LABELS.get(f, "Source artifact") for f in files],
            "value": "unavailable - source artifacts missing or invalid",
            "limitations": limitations, "commands": list(commands),
            "figure": "unavailable", "meaning": "", "settings": []}


def _settings(*, prompts="unavailable", repetitions="unavailable", tokens="unavailable",
              threads="unavailable", budget="unavailable", design="unavailable"):
    return [
        ["Model", "Qwen1.5-MoE-A2.7B-Chat int8; numpy, fused kernel, greedy decoding"],
        ["Hardware", "Historical: Intel Core i5-12450H, 8.30 GB RAM, NVMe SSD, Windows (EVIDENCE.md)"],
        ["Prompts", prompts], ["Repetitions", repetitions], ["Token limit", str(tokens)],
        ["Threads", threads], ["Budget", budget], ["Run design", design],
        ["Timing definitions", "First token: prefill forward pass; excludes model loading and tokenization. "
                               "Decode: (output tokens - 1) / remaining forward-pass seconds."],
    ]


def _experiment_settings(data, runs, *, prefetch=False):
    design = data["design"]
    prompts = list(dict.fromkeys(r["prompt_id"] for r in runs))
    threads = ("12 OpenBLAS (historical, documented in EVIDENCE.md; not recorded per run)" if prefetch else
               " / ".join(str(n) for n in sorted({r["openblas_threads"] for r in runs})) + " OpenBLAS; " +
               " / ".join(str(n) for n in sorted({r["numba_threads"] for r in runs})) + " numba")
    return _settings(prompts=f"{len(prompts)}: " + ", ".join(prompts) + " (texts in source artifacts)",
                     repetitions=f"{design['rounds']} per configuration per prompt", tokens=design["tokens"],
                     threads=threads,
                     budget=(f"Fixed expert cache: {design['cache_gb']:g} GB" if prefetch else
                             f"Cached process budget: {design['budget_gb']:g} GB"),
                     design="Fresh processes, alternating order; system file cache not cleared")


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
        limitations="One prompt and configuration family. The budget is a planning target, not a hard ceiling. "
                    "Memory sampled every 50 ms can miss spikes. Starting cache sizes differ with available RAM. "
                    "The oversized fixed-cache run is a startup refusal, not an out-of-memory crash or speed baseline.",
        commands=[PY + " eval/compare_artifacts.py results/" + PRIMARY_UNCUT + " results/" + PRIMARY_CUT,
                  "$env:OPENBLAS_NUM_THREADS = '4'", PY + " -m eval.adaptive_demo --cut-at 0", PY + " -m eval.adaptive_demo --cut-at 20"])
    mmap = _card("LeanMoE versus Windows paging", [MMAP_FULL],
        limitations="One laptop, six matched pairs. Windows file cache was never cleared, so fresh processes are not cold-cache runs. "
                    "Background load and run order matter; one warm paging run was faster. "
                    "This compares the whole cached configuration and does not isolate prefetch.",
        commands=[PY + " -m eval.config_compare --stage mmap --prompts fact code explain --rounds 2 --tokens 48 --budget-gb 3.0 --blas 4"])
    prefetch = _card("Prefetch on versus off", [PREFETCH_11], additional=True,
        limitations="Older thread configuration; not remeasured with the current 4-thread default. "
                    "The smaller 0.6 GB exploratory cache was noisy. ExpertRelay previously found a different ordering on its base store.",
        commands=["$env:OPENBLAS_NUM_THREADS = '12'", PY + " -m eval.prefetch_ablation --cache-gb 1.1 --tokens 48 --rounds 2"])
    blas_files = ["config_compare_blas_explain_20261030-053657.json", "config_compare_blas_confirm_20261030-054506.json"]
    blas = _card("Four versus twelve OpenBLAS threads", blas_files, additional=True,
        limitations="Changes threads and cache capacity together. Decode effects vary by prompt; no isolated threading speedup established. First-token outliers remain included. The explicit baseline thread option is needed because the engine now defaults to 4 threads.",
        commands=[PY + " -m eval.config_compare --stage blas --baseline-blas 12 --prompts fact code explain --rounds 2 --tokens 48 --budget-gb 3.0"])
    budget["meaning"] = "Matching output tokens in the recorded cut/uncut pair"
    mmap["meaning"] = "Ratio of mean decode throughput versus Windows paging"
    for card in (budget, mmap, prefetch, blas):
        card["settings"] = _settings()
    try:
        if check_pair(results / PRIMARY_UNCUT, results / PRIMARY_CUT)[0]:
            c = _load(results, PRIMARY_CUT)["adaptive"]
            ev = next(e for e in c["budget_events"] if e["token"] == c["cut_schedule"]["cut_at"])
            row = c["per_token"][ev["token"] - 1]
            u = _load(results, PRIMARY_UNCUT)["adaptive"]
            matched = sum(a == b for a, b in zip(u["generated_ids"], c["generated_ids"]))
            budget["figure"] = f"{matched}/{len(c['generated_ids'])}"
            settings = c["settings"]
            budget["settings"] = _settings(prompts="1: " + settings["prompt"],
                repetitions="1 uncut + 1 cut run", tokens=settings["max_new_tokens"],
                threads=f"{settings['openblas_threads']} OpenBLAS; numba count not recorded",
                budget=f"{settings['start_budget_gb']:g} to {ev['budget_gb']:g} GB after token {ev['token']}; starting cache {settings['start_cache_gb']:g} GB",
                design="Same settings except cut schedule; cache + adaptive prefetch. Strict saved-pair checker passes.")
            budget["value"] = (f"{c['settings']['start_budget_gb']:g} to {ev['budget_gb']:g} GB; "
                               f"{ev['slots_before']} to {ev['slots_after']} cached experts. "
                               f"All {len(c['generated_ids'])} tokens match. "
                               f"Private memory at the cut: {row['process_private_mb'] / 1000:.3f} GB.")
            budget["context"] = (f"{settings['start_budget_gb']:g} to {ev['budget_gb']:g} GB; cache "
                                 f"{ev['slots_before']} to {ev['slots_after']} experts; "
                                 f"{row['process_private_mb'] / 1000:.3f} GB private memory at the cut.")
    except (KeyError, TypeError, ValueError, IndexError, StopIteration):
        pass
    for card, name, left, right, field in [(mmap, MMAP_FULL, "cached", "mmap", "mode"),
                                          (prefetch, PREFETCH_11, True, False, "prefetch")]:
        try:
            data = _load(results, name)
            pairs = _matched(data["runs"], left, right, field)
            a, b, wins, same = _speed(pairs)
            card["settings"] = _experiment_settings(data, data["runs"], prefetch=name == PREFETCH_11)
            card["value"] = (f"{a / b:.2f}x ratio of mean decode rates ({a:.3f} vs {b:.3f} tokens/s). "
                             f"Faster in {wins}/{len(pairs)} pairs. Tokens match in every pair: {'yes' if same else 'no'}.")
            if name == MMAP_FULL:
                card["figure"] = f"{a / b:.2f}\u00d7"
                card["context"] = (f"Faster in {wins} of {len(pairs)} matched comparisons. "
                                     f"Mean decode: {a:.3f} vs {b:.3f} tokens/s. "
                                     f"Tokens match in every pair: {'yes' if same else 'no'}.")
                card["settings"].append(["Throughput ratio", "Ratio of arithmetic means, not the mean of pair ratios"])
                ta = statistics.median(x["time_to_first_token_s"] for x, _ in pairs)
                tb = statistics.median(y["time_to_first_token_s"] for _, y in pairs)
                card["settings"].append(["Recorded first-token medians", f"LeanMoE {ta:.1f} s; Windows paging {tb:.1f} s"])
                for key, label in [("sampled_peak_private_gb", "Sampled peak private commit"),
                                   ("sampled_peak_working_set_gb", "sampled peak working set")]:
                    a_values = [x[key] for x, _ in pairs]
                    b_values = [y[key] for _, y in pairs]
                    card["settings"].append([label, f"LeanMoE {min(a_values):.2f}-{max(a_values):.2f} GB; "
                                            f"Windows paging {min(b_values):.2f}-{max(b_values):.2f} GB"])
        except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError, statistics.StatisticsError):
            pass
    try:
        runs = [r for name in blas_files for r in _load(results, name)["runs"]]
        pairs = _matched(runs, 4, "default", "blas_setting")
        blas["settings"] = _experiment_settings(_load(results, blas_files[0]), runs)
        blas["settings"].append(["First-token summary", "Median across all six runs per configuration; thread and cache sizes change together"])
        a = statistics.median(x["time_to_first_token_s"] for x, _ in pairs)
        b = statistics.median(y["time_to_first_token_s"] for _, y in pairs)
        wins = sum(x["time_to_first_token_s"] < y["time_to_first_token_s"] for x, y in pairs)
        blas["value"] = f"Median first token: {a:.1f} vs {b:.1f} s; four threads faster in {wins}/{len(pairs)} pairs. Decode gains were mixed."
    except (KeyError, TypeError, ValueError, AttributeError, statistics.StatisticsError):
        pass
    return [budget, mmap, prefetch, blas]
