"""A tiny random-weight model for the dashboard's synthetic engine demonstration.

LeanMoE addition. Builds the same kind of store the engine tests use (with the
test helper in tests/engine/store_helpers.py) in a temporary folder. Its
weights are random, so its output is meaningless token ids: it exercises the
real engine code (store reads, expert cache, prefetch, budget, cancellation)
without the 14 GB model, and is always labelled as synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4,
    "num_key_value_heads": 4, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24, "norm_topk_prob": False, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
}  # fmt: skip
CACHE_SLOTS = 6  # fewer than the 3 x 8 experts, so the cache has to evict
MIN_FREE_SLOTS = 3


class _Patch:
    """The one monkeypatch method build_test_store needs, without pytest."""

    def __init__(self) -> None:
        self._undo: list[tuple[object, str, object]] = []

    def setattr(self, obj, name, value) -> None:
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self) -> None:
        while self._undo:
            obj, name, value = self._undo.pop()
            setattr(obj, name, value)


def build_store(path: Path) -> Path:
    sys.path.insert(0, str(REPO_ROOT / "tests" / "engine"))
    from store_helpers import build_test_store, random_checkpoint

    patch = _Patch()
    try:
        build_test_store(path, CONFIG, random_checkpoint(CONFIG, seed=1), patch)
    finally:
        patch.undo()
    return path


def prompt_ids(text: str) -> list[int]:
    """Map prompt text to synthetic token ids (UTF-8 bytes modulo the tiny vocabulary).
    Deterministic, so the same prompt gives the same ids, and meaningless by design."""
    data = text.encode("utf-8") or b" "
    return [b % CONFIG["vocab_size"] for b in data]
