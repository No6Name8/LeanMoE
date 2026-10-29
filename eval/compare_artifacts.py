"""Check an uncut and a cut adaptive_demo run against each other. Python standard library only.

    python eval/compare_artifacts.py results/adaptive_demo_20261029-205033.json results/adaptive_demo_20261029-205224.json

The two files can be given in either order. PASS requires all of these:
  - two distinct, readable result files (not the same path, not identical bytes);
  - each run: a positive requested token count, no recorded error, and the
    generated token-id list, reported generated count, per-token log and step
    log all as long as the requested count, with valid integer token ids;
  - each run: every required setting present with a valid value, and the
    settings identical between the runs;
  - exactly one uncut reference (no scheduled cut, the budget never changes)
    and exactly one cut run whose cut falls inside the generated sequence
    and lowers the budget;
  - in the cut run, a budget event at the scheduled token with the cut budget
    where the cache actually shrank, and a per-token log that agrees;
  - identical generated token-id lists.

Cache adjustments that LiveBudget makes at the starting budget (capped by free
RAM, usually after token 1) are allowed in both runs: they are not the cut.

This reads saved JSON. It does not load model weights or rerun inference, so it
checks the recorded evidence, not that the result can be reproduced. A PASS
means identical generated token ids for that prompt and configuration, not a
general correctness guarantee.

Exit code 0 on PASS, 1 on FAIL (including unreadable or malformed files).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

MAX_TOKEN_ID = 2**31 - 1


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: object) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)) and math.isfinite(x)


def _nonempty_str(x: object) -> bool:
    return isinstance(x, str) and x.strip() != ""


# Every setting adaptive_demo records, with the check its value must pass.
REQUIRED_SETTINGS = {
    "store": _nonempty_str,
    "quantization": lambda v: isinstance(v, dict) and len(v) > 0,
    "backend": _nonempty_str,
    "decoding": _nonempty_str,
    "prompt": _nonempty_str,
    "prompt_ids": lambda v: isinstance(v, list) and len(v) > 0 and all(_is_int(t) and 0 <= t <= MAX_TOKEN_ID for t in v),
    "chat_template": lambda v: isinstance(v, bool),
    "max_new_tokens": lambda v: _is_int(v) and v > 0,
    "max_seq": lambda v: _is_int(v) and v > 0,
    "start_budget_gb": lambda v: _is_num(v) and v > 0,
    "start_cache_gb": lambda v: _is_num(v) and v > 0,
    "int8_kernel": _nonempty_str,
    "io_threads": lambda v: _is_int(v) and v >= 1,
    "pin_layer0": lambda v: isinstance(v, bool),
    "prefetch": lambda v: isinstance(v, bool),
    "prefetch_k": lambda v: _is_int(v) and v >= 0,
    "prefetch_min_probability": lambda v: _is_num(v) and 0 <= v <= 1,
    "prefetch_adaptive": lambda v: isinstance(v, bool),
    "prefetch_calibration": lambda v: v is None or _nonempty_str(v),
    "prefetch_calibration_kind": _nonempty_str,
    "prefill_pipelining": lambda v: isinstance(v, bool),
}


class Malformed(Exception):
    """An artifact is missing data or has the wrong shape."""


def _get(d: object, key: str, where: str) -> object:
    if not isinstance(d, dict) or key not in d:
        raise Malformed(f"{where}: missing '{key}'")
    return d[key]


def _load(path: Path, name: str) -> tuple[dict, bytes]:
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise Malformed(f"{name}: cannot read {path}: {e.strerror or e}") from None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Malformed(f"{name}: {path} is not valid JSON ({e})") from None
    run = _get(doc, "adaptive", name)
    if not isinstance(run, dict):
        raise Malformed(f"{name}: 'adaptive' is not an object")
    return run, raw


def _check_run(run: dict, name: str) -> tuple[list[str], dict | None, int | None, float | None]:
    """Problems with one run on its own, plus its settings, cut token and cut budget."""
    problems: list[str] = []
    settings = _get(run, "settings", name)
    if not isinstance(settings, dict):
        raise Malformed(f"{name}: 'settings' is not an object")
    missing = [k for k in REQUIRED_SETTINGS if k not in settings]
    if missing:
        problems.append(f"{name}: settings missing {missing}")
    invalid = [k for k, ok in REQUIRED_SETTINGS.items() if k in settings and not ok(settings[k])]
    if invalid:
        problems.append(f"{name}: settings with invalid values {invalid}")

    requested = _get(run, "tokens_requested", name)
    reported = _get(run, "tokens_generated", name)
    ids = _get(run, "generated_ids", name)
    per_token = _get(run, "per_token", name)
    steps = _get(run, "steps", name)
    error = _get(run, "error", name)
    events = _get(run, "budget_events", name)
    for key, value, kind in (("generated_ids", ids, list), ("per_token", per_token, list),
                             ("steps", steps, list), ("budget_events", events, list)):  # fmt: skip
        if not isinstance(value, kind):
            raise Malformed(f"{name}: '{key}' is not a list")

    if not (_is_int(requested) and requested > 0):
        problems.append(f"{name}: requested token count must be a positive integer, got {requested!r}")
    elif "max_new_tokens" in settings and settings["max_new_tokens"] != requested:
        problems.append(f"{name}: requested {requested} tokens but settings say max_new_tokens={settings['max_new_tokens']}")
    if error is not None:
        problems.append(f"{name}: run recorded an error: {error}")
    if _is_int(requested) and requested > 0:
        lengths = {"generated ids": len(ids), "reported generated": reported,
                   "per-token log": len(per_token), "step log": len(steps)}  # fmt: skip
        short = {k: v for k, v in lengths.items() if v != requested}
        if short:
            problems.append(f"{name}: incomplete run, requested {requested} but {short}")
    bad_ids = [i + 1 for i, t in enumerate(ids) if not (_is_int(t) and 0 <= t <= MAX_TOKEN_ID)]
    if bad_ids:
        problems.append(f"{name}: invalid token ids at positions {bad_ids[:5]}")
    for i, row in enumerate(per_token):
        if _get(row, "token", f"{name} per_token[{i}]") != i + 1:
            problems.append(f"{name}: per-token log out of order at entry {i + 1}")
            break

    schedule = run.get("cut_schedule")
    if not isinstance(schedule, dict) or not _is_int(schedule.get("cut_at")):
        problems.append(f"{name}: no valid cut schedule recorded")
        return problems, settings, None, None
    cut_at = schedule["cut_at"]
    cut_gb = schedule.get("cut_gb")
    return problems, settings, cut_at, cut_gb


def _row(per_token: list, token: int) -> dict | None:
    return per_token[token - 1] if 1 <= token <= len(per_token) else None


def _check_uncut(run: dict, start: float, name: str) -> list[str]:
    problems = []
    for e in run["budget_events"]:
        if _get(e, "budget_gb", f"{name} budget event") != start:
            problems.append(f"{name}: uncut reference changed its budget to {e['budget_gb']} at token {e.get('token')}")
    if any(_get(r, "budget_gb", f"{name} per_token") != start for r in run["per_token"]):
        problems.append(f"{name}: uncut reference's per-token budget is not always {start}")
    return problems


def _check_cut(run: dict, cut_at: int, cut_gb: object, start: float, name: str) -> list[str]:
    problems = []
    n = len(run["generated_ids"])
    if not 1 <= cut_at < n:
        problems.append(f"{name}: scheduled cut at token {cut_at} is not inside the generated sequence (1..{n - 1})")
        return problems
    if not (_is_num(cut_gb) and 0 < cut_gb < start):
        problems.append(f"{name}: cut budget {cut_gb!r} is not below the starting budget {start}")
        return problems
    events = [e for e in run["budget_events"] if _get(e, "token", f"{name} budget event") == cut_at]
    cut_events = [e for e in events if e.get("budget_gb") == cut_gb]
    if not cut_events:
        problems.append(f"{name}: no budget event at token {cut_at} with the cut budget {cut_gb} GB")
    else:
        e = cut_events[0]
        before, after = _get(e, "slots_before", f"{name} cut event"), _get(e, "slots_after", f"{name} cut event")
        if not (_is_int(before) and _is_int(after) and after < before):
            problems.append(f"{name}: cache did not shrink at the cut (experts {before!r} -> {after!r})")
    other = [e for e in run["budget_events"] if e not in cut_events and e.get("budget_gb") != start]
    if other:
        problems.append(f"{name}: unexpected budget changes besides the scheduled cut: {other}")
    prev, at = _row(run["per_token"], cut_at - 1), _row(run["per_token"], cut_at)
    if at is None or at.get("budget_gb") != cut_gb:
        problems.append(f"{name}: per-token log does not show the cut budget at token {cut_at}")
    elif prev is not None and not at.get("cache_slots", 0) < prev.get("cache_slots", 0):
        problems.append(f"{name}: per-token log does not show the cache shrinking at token {cut_at}")
    return problems


def check_pair(path_a: Path, path_b: Path) -> tuple[bool, list[str]]:
    """(passed, lines to print). Never raises for bad input: malformed data is a FAIL."""
    lines: list[str] = []
    try:
        if Path(path_a).resolve() == Path(path_b).resolve():
            return False, ["FAIL", "  - the same file was given twice: self-comparison proves nothing"]
        a, raw_a = _load(path_a, "A")
        b, raw_b = _load(path_b, "B")
        if raw_a == raw_b:
            return False, ["FAIL", "  - the two files have identical contents: self-comparison proves nothing"]
        pa, sa, cut_a, gb_a = _check_run(a, "A")
        pb, sb, cut_b, gb_b = _check_run(b, "B")
        problems = pa + pb
        if sa is not None and sb is not None:
            diff = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
            if diff:
                problems.append(f"settings differ between the runs: {diff}")
        roles = {"A": cut_a, "B": cut_b}
        uncut = [k for k, c in roles.items() if c == 0]
        cut = [k for k, c in roles.items() if _is_int(c) and c > 0]
        if len(uncut) != 1 or len(cut) != 1:
            problems.append(f"need exactly one uncut reference and one cut run; cut schedules are {roles}")
        elif _is_num(sa.get("start_budget_gb")):
            start = sa["start_budget_gb"]
            runs = {"A": (a, gb_a), "B": (b, gb_b)}
            problems += _check_uncut(runs[uncut[0]][0], start, f"{uncut[0]} (uncut)")
            run, gb = runs[cut[0]]
            problems += _check_cut(run, roles[cut[0]], gb, start, f"{cut[0]} (cut)")
            ev = [(e.get("token"), e.get("slots_before"), e.get("slots_after"), e.get("budget_gb"))
                  for e in run["budget_events"]]  # fmt: skip
            lines.append(f"uncut reference: {path_a if uncut[0] == 'A' else path_b}")
            lines.append(f"cut run: {path_a if cut[0] == 'A' else path_b}, cut at token {roles[cut[0]]} to {gb} GB")
            lines.append(f"   cut run budget events (token, experts before, after, budget GB): {ev}")

        ia, ib = a["generated_ids"], b["generated_ids"]
        first_diff = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
        if first_diff is not None:
            problems.append(f"token ids differ, first at token {first_diff + 1}: {ia[first_diff]!r} vs {ib[first_diff]!r}")
        elif len(ia) != len(ib):
            problems.append(
                f"one token list is a prefix of the other: lengths {len(ia)} vs {len(ib)}, "
                f"first missing token at position {min(len(ia), len(ib)) + 1}"
            )
    except Malformed as e:
        return False, lines + ["FAIL", f"  - malformed artifact: {e}"]
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
        return False, lines + ["FAIL", f"  - malformed artifact: {type(e).__name__}: {e}"]

    if problems:
        return False, lines + ["FAIL"] + [f"  - {p}" for p in problems]
    return True, lines + [
        f"PASS: identical generated token ids ({len(ia)} tokens) for this tested prompt and configuration; "
        "the scheduled cut shrank the cache and the run completed.",
        "(Checked from saved artifacts only; this does not rerun inference.)",
    ]


def compare_runs(path_a: Path, path_b: Path) -> int:
    passed, lines = check_pair(Path(path_a), Path(path_b))
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    sys.exit(compare_runs(Path(sys.argv[1]), Path(sys.argv[2])))
