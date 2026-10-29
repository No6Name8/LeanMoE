# LeanMoE — Build Contract

Single-machine MoE expert prefetching: predict which expert is needed next
and load it from disk into RAM *before* it's needed, overlapping disk I/O
with compute instead of stalling on every cache miss.

## Ownership

- src/ — Claude (chat session). Model loading, expert cache, predictor, prefetch scheduler.
- eval/ — GPT Plus (relayed by Abdullah). Test prompts, timing/RAM measurement, baseline-vs-predictive comparison script, robustness prompt set.
- Nobody edits outside their own folder without flagging it in this file first.

## Interfaces (the contract between src/ and eval/)

### src/engine.py exposes:

```python
class LeanMoEEngine:
    def __init__(self, model_name: str, mode: str, ram_budget_experts: int):
        """
        mode: "naive" (load expert from disk only on demand, blocking)
              or "predictive" (prefetch likely-next expert in background)
        ram_budget_experts: max experts held in RAM at once
        """

    def generate(self, prompt: str, max_new_tokens: int = 20) -> GenResult:
        """Runs one prompt through the model. Returns timing + text."""


@dataclass
class GenResult:
    text: str
    total_time_s: float
    stall_time_s: float       # time spent blocked waiting on disk load
    experts_loaded: int       # total disk loads during this generation
    cache_hits: int           # times the needed expert was already in RAM
    cache_misses: int         # times it had to be loaded (blocking or prefetched)
    peak_ram_mb: float
```

### eval/ calls src/engine.py only through LeanMoEEngine and GenResult.
It never reaches into internals (cache dict, scheduler thread, etc).

### eval/ provides back to src/ / run.py:

- eval/test_cases.py — TEST_PROMPTS: list[str] (~30-50 prompts, varied topics so different experts activate)
- eval/robustness_cases.py — ROBUSTNESS_PROMPTS: list[str] (10-15 edge cases: very long prompts, rapid topic switching, repeated prompts)
- eval/metrics.py — run_comparison(engine_naive, engine_predictive, prompts) -> ComparisonResult, producing the results table (avg stall time, tokens/sec, peak RAM, cache hit rate) for both modes
- eval/log_experiment(config: dict, result: ComparisonResult) — appends one row to results/log.csv

## Data flow

```
run.py
  → loads eval/test_cases.py prompts
  → builds two LeanMoEEngine instances (naive, predictive) from src/engine.py
  → eval/metrics.run_comparison(...) drives both engines, times them
  → prints results table to terminal (demo-ready)
  → eval/log_experiment(...) appends to results/log.csv
```

## What "predictive" must do differently from "naive"

- Naive: on a cache miss, block, load expert from disk, then compute.
- Predictive: while computing with the current expert, look at routing signal (or a simple heuristic/learned predictor) to guess the next likely expert, and start loading it on a background thread now, so it's hopefully ready by the time it's needed.

## Model

Start with allenai/OLMoE-1B-7B-0924 if it loads on 8GB tonight. Fallback: a
small synthetic MoE (hand-built, 4-8 tiny "expert" weight files) if the real
model doesn't fit — proves the mechanism even if not a production model.
This gets decided after tonight's load test, logged in DECISIONS.md.
