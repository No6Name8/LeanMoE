"""Scoring rules for the pre-registered task benchmark (configs/tasks/task_benchmark_v1.json).

Standard library only. Each rule turns a model reply into "correct", "incorrect"
or "manual_review", with a short reason; the rules are the ones written in the
task file before any run. Nothing here judges execution consistency (identical
tokens across modes); that is a separate check.
"""

from __future__ import annotations

import re
from fractions import Fraction

CORRECT, INCORRECT, MANUAL = "correct", "incorrect", "manual_review"

_ANSWER = re.compile(r"^\s*(?:\*\*)?answer(?:\*\*)?\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_NUMBER = re.compile(r"-?\d+(?:,\d{3})+(?:\.\d+)?(?![\d,])|-?\d+/\d+|-?\d+(?:\.\d+)?")
_X_EQUALS = re.compile(r"\bx\s*=\s*(" + _NUMBER.pattern + ")")
TOL = 1e-9


def answer_field(text: str) -> str | None:
    """Text after the last standalone 'Answer:' line, markdown emphasis stripped."""
    matches = list(_ANSWER.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).splitlines()[0].strip().strip("*_ ").strip()


def numbers(field: str) -> list[float]:
    field = field.replace("−", "-")
    out = []
    for m in _NUMBER.finditer(field):
        s = m.group(0).replace(",", "")
        out.append(float(Fraction(s)) if "/" in s else float(s))
    return out


def _eq(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def _in(v: float, values) -> bool:
    return any(_eq(v, w) for w in values)


def _same_set(values, expected) -> bool:
    return bool(values) and all(_in(v, expected) for v in values) and all(_in(e, values) for e in expected)


def last_number(field: str, expected: float) -> tuple[str, str]:
    nums = numbers(field)
    if nums and _eq(nums[-1], expected):
        return CORRECT, f"last number {nums[-1]:g} equals {expected:g}"
    if _in(expected, nums):
        return MANUAL, f"{expected:g} appears but the last number is {nums[-1]:g}"
    return INCORRECT, f"numbers {[f'{n:g}' for n in nums]} do not give {expected:g}"


def root_set(field: str, expected: list[float]) -> tuple[str, str]:
    norm = field.replace("−", "-")
    x_values = [numbers(m.group(1))[0] for m in _X_EQUALS.finditer(norm)]
    nums = numbers(norm)
    if _same_set(x_values, expected) or _same_set(nums, expected):
        return CORRECT, f"solutions {sorted({round(v, 9) for v in (x_values or nums)})} match"
    if all(_in(e, nums) for e in expected):
        return MANUAL, f"expected values appear among {[f'{n:g}' for n in nums]} with others"
    return INCORRECT, f"x values {[f'{v:g}' for v in x_values]}, numbers {[f'{n:g}' for n in nums]}"


def choice(field: str, expected: str, options: list[str], case_sensitive: bool = False) -> tuple[str, str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    named = [o for o in options if re.search(r"(?<![A-Za-z])" + re.escape(o) + r"(?![A-Za-z])", field, flags)]
    wrong = [o for o in named if o != expected]
    if expected in named and not wrong:
        return CORRECT, f"names {expected}"
    if expected in named:
        return MANUAL, f"names {expected} and also {wrong}"
    return INCORRECT, f"names {wrong or 'none of the options'}"


def score(task: dict, text: str, stop_reason: str) -> dict:
    """Apply the task's pre-registered rule to one reply."""
    field = answer_field(text)
    if field is None:
        if stop_reason == "eos":
            return {"score": MANUAL, "answer_field": None, "reason": "no 'Answer:' line in a reply that ended normally"}
        return {"score": INCORRECT, "answer_field": None, "reason": f"no 'Answer:' line; reply ended by {stop_reason}"}
    rule = task["rule"]
    if rule == "last_number":
        s, why = last_number(field, float(task["expected"]))
    elif rule == "root_set":
        s, why = root_set(field, [float(v) for v in task["expected"]])
    elif rule == "choice":
        s, why = choice(field, task["expected"], task["options"], task.get("case_sensitive", False))
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return {"score": s, "answer_field": field, "reason": why}
