"""
Synthetic MoE fallback for LeanMoE.

A tiny fake mixture-of-experts: N experts, each a single Linear(dim -> dim)
layer saved to its own file on disk. Nothing is loaded at startup. Experts
are pulled from disk on first use, which is where the real I/O cost lives.
This lets us measure caching and prefetching without a 14 GB checkpoint.

Quick start:
    from src.synthetic_moe import build_experts, run_token, generate_expert_sequence, new_hidden_state
    build_experts()                                  # writes experts/expert_0.pt .. expert_7.pt
    seq = generate_expert_sequence(length=100, num_experts=8)
    cache = {}
    h = new_hidden_state()
    for eid in seq:
        h = run_token(eid, h, cache)
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERTS_DIR = REPO_ROOT / "experts"
NUM_EXPERTS = 8
DIM = 256


def expert_path(expert_id: int, experts_dir: Path = EXPERTS_DIR) -> Path:
    return Path(experts_dir) / f"expert_{expert_id}.pt"


def build_experts(
    num_experts: int = NUM_EXPERTS,
    dim: int = DIM,
    experts_dir: Path = EXPERTS_DIR,
    seed: int = 0,
    overwrite: bool = False,
) -> list[Path]:
    """Create one Linear(dim, dim) per expert and save each to its own file.

    Weights are freed right after saving, so nothing stays in RAM.
    Raise `dim` (e.g. 2048 -> ~16 MB per expert) to make disk loads more costly.
    """
    experts_dir = Path(experts_dir)
    experts_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    paths = []
    for i in range(num_experts):
        p = expert_path(i, experts_dir)
        if overwrite or not p.exists():
            layer = nn.Linear(dim, dim)
            # Small weights so repeated application doesn't blow up the hidden state.
            nn.init.normal_(layer.weight, std=1.0 / dim**0.5)
            torch.save(layer.state_dict(), p)
            del layer
        paths.append(p)
    return paths


def load_expert(
    expert_id: int,
    experts_dir: Path = EXPERTS_DIR,
    simulated_latency_s: float = 0.0,
) -> nn.Linear:
    """Read one expert from disk into a fresh Linear layer (the 'disk I/O' step).

    simulated_latency_s adds an artificial delay. Use it when the OS file cache
    makes real reads too fast to show a difference, and say so in results.
    """
    state = torch.load(expert_path(expert_id, experts_dir), map_location="cpu", weights_only=True)
    out_dim, in_dim = state["weight"].shape
    layer = nn.Linear(in_dim, out_dim)
    layer.load_state_dict(state)
    layer.eval()
    if simulated_latency_s > 0:
        time.sleep(simulated_latency_s)
    return layer


def run_token(
    expert_id: int,
    hidden_state: torch.Tensor,
    loaded_experts: dict,
    experts_dir: Path = EXPERTS_DIR,
    simulated_latency_s: float = 0.0,
) -> torch.Tensor:
    """Run one token's hidden state through one expert.

    Cache hit: expert_id is already in loaded_experts, used directly.
    Cache miss: expert is loaded from disk into loaded_experts first.

    Eviction is not done here. The caller (the engine) owns the RAM budget.
    """
    expert = loaded_experts.get(expert_id)
    if expert is None:
        expert = load_expert(expert_id, experts_dir, simulated_latency_s)
        loaded_experts[expert_id] = expert
    with torch.no_grad():
        # Residual connection plus tanh keeps values bounded across many tokens.
        return torch.tanh(hidden_state + expert(hidden_state))


def new_hidden_state(dim: int = DIM, seed: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(1, dim, generator=g)


def generate_expert_sequence(
    length: int,
    num_experts: int = NUM_EXPERTS,
    seed: int | None = 0,
    p_repeat: float = 0.35,
    p_recent: float = 0.25,
    recent_window: int = 4,
) -> list[int]:
    """Produce a routing sequence with learnable structure.

    At each step:
      - with p_repeat, reuse the previous expert (short-term stickiness);
      - with p_recent, pick one of the last `recent_window` distinct experts,
        weighted toward the most recent (temporal locality);
      - otherwise, follow a fixed sparse transition table where each expert
        has 2 favourite successors (structure a learned predictor can pick up),
        with a small chance of a uniformly random jump.

    The transition table depends only on `num_experts` and `seed`, so the same
    seed gives the same routing patterns for both naive and predictive runs.
    """
    if length <= 0:
        return []
    rng = random.Random(seed)
    table_rng = random.Random(None if seed is None else seed + 1)

    # Each expert prefers two successors (70% / 20%), 10% uniform jump.
    successors = {}
    for e in range(num_experts):
        others = [x for x in range(num_experts) if x != e] or [e]
        successors[e] = table_rng.sample(others, k=min(2, len(others)))

    seq = [rng.randrange(num_experts)]
    recent: list[int] = [seq[0]]
    while len(seq) < length:
        prev = seq[-1]
        r = rng.random()
        if r < p_repeat:
            nxt = prev
        elif r < p_repeat + p_recent and len(recent) > 1:
            # Most recent gets the highest weight.
            weights = list(range(len(recent), 0, -1))
            nxt = rng.choices(recent, weights=weights, k=1)[0]
        else:
            u = rng.random()
            favs = successors[prev]
            if u < 0.7:
                nxt = favs[0]
            elif u < 0.9 and len(favs) > 1:
                nxt = favs[1]
            else:
                nxt = rng.randrange(num_experts)
        seq.append(nxt)
        if nxt in recent:
            recent.remove(nxt)
        recent.insert(0, nxt)
        del recent[recent_window:]
    return seq


def _demo() -> None:
    """Run 200 tokens twice: cold cache vs warm cache, and report timing."""
    build_experts()
    seq = generate_expert_sequence(200, NUM_EXPERTS, seed=0)

    def run(cache: dict) -> tuple[float, int, int]:
        hits = misses = 0
        h = new_hidden_state(seed=0)
        t0 = time.perf_counter()
        for eid in seq:
            if eid in cache:
                hits += 1
            else:
                misses += 1
            h = run_token(eid, h, cache)
        return time.perf_counter() - t0, hits, misses

    # Worst case: no cache at all, every token reloads from disk.
    t_nocache = 0.0
    h = new_hidden_state(seed=0)
    t0 = time.perf_counter()
    for eid in seq:
        h = run_token(eid, h, {})
    t_nocache = time.perf_counter() - t0

    cold, hits_c, miss_c = run({})
    warm_cache = {i: load_expert(i) for i in range(NUM_EXPERTS)}
    warm, hits_w, miss_w = run(warm_cache)

    repeats = sum(a == b for a, b in zip(seq, seq[1:]))
    print(f"sequence: first 30 = {seq[:30]}")
    print(f"same-as-previous rate: {repeats / (len(seq) - 1):.0%}")
    print(f"{'run':<26}{'time (ms)':>10}{'hits':>6}{'misses':>8}")
    print(f"{'reload every token':<26}{t_nocache * 1000:>10.1f}{0:>6}{len(seq):>8}")
    print(f"{'unbounded cache, cold':<26}{cold * 1000:>10.1f}{hits_c:>6}{miss_c:>8}")
    print(f"{'all experts preloaded':<26}{warm * 1000:>10.1f}{hits_w:>6}{miss_w:>8}")


if __name__ == "__main__":
    _demo()
