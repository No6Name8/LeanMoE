"""Recorded benchmark figures for the dashboard, computed from the committed result files.

LeanMoE addition. Nothing here is measured live: every figure is read from a
file in results/ when the dashboard starts, and carries that file's name, so
the page can show it as recorded evidence. A missing or unreadable file is
skipped, not replaced.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

PRIMARY_UNCUT = "adaptive_demo_20261030-060626.json"
PRIMARY_CUT = "adaptive_demo_20261030-060728.json"
MMAP_FULL = "config_compare_mmap_full_20261030-060448.json"
PREFETCH_11 = "prefetch_ablation_20261029-215951.json"


def _load(results: Path, name: str) -> dict | None:
    try:
        return json.loads((results / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def recorded_evidence(results: Path) -> list[dict]:
    items: list[dict] = []
    uncut, cut = _load(results, PRIMARY_UNCUT), _load(results, PRIMARY_CUT)
    if uncut and cut:
        try:
            u, c = uncut["adaptive"], cut["adaptive"]
            ev = next(e for e in c["budget_events"] if e["token"] == c["cut_schedule"]["cut_at"])
            row = c["per_token"][ev["token"] - 1]
            items.append({
                "label": "Budget cut mid-generation (one prompt, 80 tokens)",
                "value": (f"{c['settings']['start_budget_gb']} GB to {ev['budget_gb']} GB after token {ev['token']}: "
                          f"cache {ev['slots_before']} to {ev['slots_after']} experts, private memory "
                          f"{row['process_private_mb'] / 1000:.3f} GB after the cut; token ids identical to the "
                          f"uncut run: {u['generated_ids'] == c['generated_ids']}"),
                "files": [PRIMARY_UNCUT, PRIMARY_CUT],
            })  # fmt: skip
        except (KeyError, StopIteration, IndexError, TypeError):
            pass
    mm = _load(results, MMAP_FULL)
    if mm:
        try:
            runs = [r for r in mm["runs"] if not r.get("error")]
            dm = [r["decode_tokens_per_s"] for r in runs if r["mode"] == "mmap"]
            dc = [r["decode_tokens_per_s"] for r in runs if r["mode"] == "cached"]
            tm = [r["time_to_first_token_s"] for r in runs if r["mode"] == "mmap"]
            tc = [r["time_to_first_token_s"] for r in runs if r["mode"] == "cached"]
            items.append({
                "label": "Cached with prefetch versus memory-mapped (3 prompts x 2 runs, 48 tokens)",
                "value": (f"decode mean {statistics.mean(dc):.3f} versus {statistics.mean(dm):.3f} tokens/s "
                          f"(ratio of means {statistics.mean(dc) / statistics.mean(dm):.2f}); median first token "
                          f"{statistics.median(tc):.1f} versus {statistics.median(tm):.1f} s"),
                "files": [MMAP_FULL],
            })  # fmt: skip
        except (KeyError, ZeroDivisionError, statistics.StatisticsError, TypeError):
            pass
    pf = _load(results, PREFETCH_11)
    if pf:
        try:
            on = [r["decode_tokens_per_s"] for r in pf["runs"] if r.get("prefetch") and not r.get("error")]
            off = [r["decode_tokens_per_s"] for r in pf["runs"] if not r.get("prefetch") and not r.get("error")]
            items.append({
                "label": "Prefetch on versus off, 1.1 GB cache (12 OpenBLAS threads, before the 4-thread default)",
                "value": f"decode mean {statistics.mean(on):.3f} versus {statistics.mean(off):.3f} tokens/s",
                "files": [PREFETCH_11],
            })  # fmt: skip
        except (KeyError, statistics.StatisticsError, TypeError):
            pass
    return items
