"""Compare two adaptive_demo result files. Python standard library only.

    python eval/compare_artifacts.py results/adaptive_demo_20261029-205033.json results/adaptive_demo_20261029-205224.json

PASS only if both runs recorded their settings, the settings are identical
(only the budget-cut schedule may differ), both runs generated every
requested token without error, and the full generated token-id lists match.

This reads saved JSON. It does not load model weights or rerun inference, so
it checks the recorded evidence, not that the result can be reproduced. A
PASS means identical generated token ids for that prompt and configuration,
not a general correctness guarantee.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _cut_figures(run: dict) -> str | None:
    """Cache, memory and per-token time on either side of the cut, from the per-token log."""
    cut_at = run.get("cut_schedule", {}).get("cut_at", run.get("cut_at", 0))
    log, steps = run["per_token"], run["steps"]
    if not cut_at or cut_at < 2 or len(log) < cut_at:
        return None
    before, after = log[cut_at - 2], log[cut_at - 1]
    return (
        f"cut after token {cut_at}: cache {before['cache_slots']} -> {after['cache_slots']} experts "
        f"({before['cache_gb']:.2f} -> {after['cache_gb']:.2f} GB), process private memory "
        f"{before['process_private_mb']:.0f} -> {after['process_private_mb']:.0f} MB"
    )


def compare_runs(path_a: Path, path_b: Path) -> int:
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))["adaptive"]
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))["adaptive"]
    problems = []
    sa, sb = a.get("settings"), b.get("settings")
    if sa is None or sb is None:
        problems.append("settings not recorded in " + ("A" if sa is None else "B") + ": comparison not admissible")
    else:
        diff = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
        if diff:
            problems.append(f"settings differ: {diff}")
    for name, r in (("A", a), ("B", b)):
        if r["error"]:
            problems.append(f"{name} failed: {r['error']}")
        if r["tokens_generated"] != r["tokens_requested"]:
            problems.append(f"{name} generated {r['tokens_generated']}/{r['tokens_requested']} tokens")
    ia, ib = a["generated_ids"], b["generated_ids"]
    first_diff = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
    if first_diff is not None:
        problems.append(f"token ids differ, first at token {first_diff + 1}: {ia[first_diff]} vs {ib[first_diff]}")
    elif len(ia) != len(ib):
        problems.append(
            f"one list is a prefix of the other: lengths {len(ia)} vs {len(ib)}, "
            f"first missing token at position {min(len(ia), len(ib)) + 1}"
        )

    for name, path, r in (("A", path_a, a), ("B", path_b, b)):
        schedule = r.get("cut_schedule", {"cut_at": r.get("cut_at")})
        events = [(e["token"], e["slots_before"], e["slots_after"], e["reason"]) for e in r["budget_events"]]
        print(f"{name}: {path}")
        print(f"   cut schedule {schedule}, {r['tokens_generated']}/{r['tokens_requested']} tokens, error {r['error']}")
        print(f"   budget events (token, experts before, after, reason): {events}")
        figures = _cut_figures(r)
        if figures:
            print(f"   {figures}")
    if len(a["steps"]) >= 80 and len(b["steps"]) >= 80:
        ta = _mean([s["total_seconds"] for s in a["steps"][20:80]])
        tb = _mean([s["total_seconds"] for s in b["steps"][20:80]])
        print(f"mean seconds per token, tokens 21-80: A {ta:.3f}, B {tb:.3f} ({(tb / ta - 1) * 100:+.1f}% B vs A)")

    if problems:
        print("FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"PASS: identical generated token ids ({len(ia)} tokens) for this tested prompt and configuration.")
    print("(Checked from saved artifacts only; this does not rerun inference.)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(compare_runs(Path(sys.argv[1]), Path(sys.argv[2])))
