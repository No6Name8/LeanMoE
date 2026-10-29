# LeanMoE

Run a mixture-of-experts (MoE) model on an 8 GB laptop by keeping only some experts in RAM, loading the rest from disk on demand, and prefetching the experts the router is likely to need next. This submission's new piece is an **adaptive-RAM demo**: the memory budget is cut while the model is generating, the expert cache shrinks between tokens, and generation carries on.

See [DECISIONS.md](DECISIONS.md) for the decisions log, [CONTRACT.md](CONTRACT.md) for repo rules, and [results/EVIDENCE.md](results/EVIDENCE.md) for the evidence files.

## Prior work: ExpertRelay

The inference engine in `src/engine/` is **existing work**, ported from ExpertRelay, a private research project by Abdullah Ali Alanazi (2026). Concept note: [doi:10.5281/zenodo.22980838](https://doi.org/10.5281/zenodo.22980838). Every ported file starts with `# Adapted from ExpertRelay (private research project, 2026)`.

Ported, not written for this submission:

- **Engine:** the numpy Qwen-MoE runtime, int8 and int4 kernels and compute backends (`src/engine/runtime/`).
- **Expert store format and readers:** `store.json`, `experts.bin` and `resident.bin`, with unbuffered disk reads (`src/engine/store/`).
- **Expert cache:** slot cache with LRU eviction and resizing (`src/engine/cache/expert_cache.py`).
- **Predictor:** the router-probability prefetch policy and its fitted calibrations (`src/engine/predictor/`, `configs/calibration/`).
- **Manager:** machine profiling and automatic configuration (`src/engine/manager/`).
- **LiveBudget:** re-reads a budget file between tokens and resizes the cache (`src/engine/runtime/auto.py`). The adaptive demo drives this existing mechanism; it did not add it.
- **Memory guard:** refuses at startup a configuration whose estimated RAM exceeds the configured budget (`src/engine/memory_budget.py`).

## New in this submission

- **The port:** imports moved to `src.engine`, model stores located through `LEANMOE_MODELS_ROOT`, and MindSpore made optional (the machine profile required it). Seven extra ExpertRelay files came along because the requested ones import them.
- **Adaptive-RAM demo** (`eval/adaptive_demo.py`): cuts the budget mid-generation, logs cache size and RAM per token, records every setting, and runs the fixed-cache comparison.
- **Artifact comparison** (`eval/compare_artifacts.py`): standard-library-only check of two saved runs.
- **Synthetic MoE fallback** (`src/synthetic_moe.py`): 8 toy experts on disk. **It only illustrates disk-load versus RAM-hit cost on toy weights. It is not a model, it doesn't use LiveBudget, and it validates nothing about real-model performance.**
- **Fresh measurements**, below.

## Fresh measurements (this submission, 2026-10-29)

One 8 GB Windows laptop, Qwen1.5-MoE-A2.7B-Chat int8 store, numpy backend, fused int8 kernel, prefetch on, 80 tokens, 3.5 GB starting budget, 1.5 GB starting cache. Two runs made back to back from a fresh virtual environment, with all settings recorded and identical apart from the cut schedule.

| | Uncut reference | Cut to 2.5 GB after token 20 |
|---|---|---|
| Evidence file | `results/adaptive_demo_20261029-205033.json` | `results/adaptive_demo_20261029-205224.json` |
| Cache after the first token | 120 experts (1.04 GB) | 112 experts (0.97 GB) |
| Cache after token 20 | 120 experts | 59 experts (0.51 GB), 47% fewer |
| Process private memory, token 19 to 20 | 3,189 MB, unchanged | 3,120 MB to 2,659 MB, 14.8% less |
| Mean seconds per token, tokens 21 to 80 | 0.556 | 0.637, 14.7% slower |
| Tokens, errors | 80 of 80, none | 80 of 80, none |

- **Token ids matched** for this tested prompt and configuration: all 80 generated token ids are identical between the two runs. This is one prompt and one configuration, not a general correctness guarantee.
- **Starting cache sizes differed** because available system RAM differed. Both runs were configured for 1.5 GB, and LiveBudget capped each by the RAM free when it applied the starting budget after the first token.
- **Timing is from one pair of runs on a busy laptop.** The slowdown figure is indicative only.
- **Fixed-cache comparison:** a fresh run with a fixed 1.5 GB cache under the 2.5 GB budget was **refused at startup** by the memory guard (estimated 3.48 GB over the configured 2.50 GB). It never generated and did not go through the same live cut. It is a refusal of the smaller configured budget, not an out-of-memory crash: 2.66 to 2.91 GB of RAM was actually free when it ran (once after each primary run).
- **Port check:** the ported engine generated the same 32 token ids as the original ExpertRelay code on the same store, prompt and settings.

An earlier cut run (`results/adaptive_demo_20261029-200451.json`) is supplementary only: it predates settings recording. Its figures (165 to 59 experts, 64% fewer; about 5% slower after the cut than before it within that run) are not the fresh pair's figures.

## Historical ExpertRelay measurement

One run of the **original ExpertRelay code**, earlier the same evening, on the same store with a 1.5 GB cache: first token 22.1 s, decode 0.78 tokens/s, peak RAM 3,094 MB. ExpertRelay's own older benchmark results are not reproduced or relied on here.

## Reproducing

Install (tested in a fresh virtual environment on Windows with Python 3.13):

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Windows, use a short directory path or enable long paths: torch contains deeply nested files, and an install into a long path failed with a path-length error during testing.

**1. Check the saved evidence, no model needed.** Standard library only. It reads the committed JSON files; it does not rerun inference, so it is not an independent reproduction.

```
python eval/compare_artifacts.py results/adaptive_demo_20261029-205033.json results/adaptive_demo_20261029-205224.json
```

**2. Synthetic fallback, no model needed.** Toy weights only; see the warning above.

```
python -m src.synthetic_moe
```

**3. Real-model adaptive demo, needs the external store.** Real-model inference needs the Qwen1.5-MoE-A2.7B-Chat int8 store, about 14 GB, which is not in this repo. It was built by ExpertRelay's store builder, which is not included here and needs the 27 GB original checkpoint. With the store in a folder named `qwen1.5-moe-a2.7b-chat-int8` inside some directory:

```
set LEANMOE_MODELS_ROOT=C:\path\to\that\directory
python -m eval.adaptive_demo --cut-at 0
python -m eval.adaptive_demo --cut-at 20
python -m eval.adaptive_demo --compare results\<uncut run>.json results\<cut run>.json
```

Each run takes about two to three minutes on this laptop and writes a new `results/adaptive_demo_<timestamp>.json`. If `LEANMOE_MODELS_ROOT` is unset, the engine looks in `models/` at the repo root, which is gitignored.
