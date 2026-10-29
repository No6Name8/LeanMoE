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

**Main result:** a scheduled budget reduction after token 20 shrank the expert cache, generation completed all 80 tokens without an error, and the generated token ids matched the uncut reference for this tested prompt and configuration.

| | Uncut reference | Cut to 2.5 GB after token 20 |
|---|---|---|
| Evidence file | `results/adaptive_demo_20261029-205033.json` | `results/adaptive_demo_20261029-205224.json` |
| Cache after the first token | 120 experts (1.041 GB) | 112 experts (0.971 GB) |
| Cache after token 20 | 120 experts (1.041 GB) | 59 experts (0.512 GB), 47% fewer |
| Process private memory, token 19 to 20 | 3.189 GB, unchanged | 3.120 GB to 2.659 GB, 0.461 GB (14.8%) less |
| Tokens, errors | 80 of 80, none | 80 of 80, none |
| Generated token ids | identical in both runs (80 of 80) | |

- **Token ids matched** for this tested prompt and configuration. It is one prompt, one configuration and greedy decoding, not a general correctness guarantee.
- **Starting cache sizes differed** because available system RAM differed. Both runs were configured for a 1.5 GB cache. LiveBudget applies the starting budget after the first token and also caps the cache by the RAM free at that moment, which varied between the runs. This is not the scheduled cut.
- **Cache reduction is not total memory saving.** The cache fell 47%; the whole process's private memory fell 14.8%, because the rest of the process (about 2 GB) doesn't shrink.
- **Timing:** tokens 21 to 80 averaged 0.556 s per token uncut and 0.637 s cut. That is one noisy paired observation, not an established resizing penalty: the runs started with different cache sizes, and time per token varied by about 0.12 s (standard deviation) within the uncut run alone.
- **Fixed-cache example:** a fresh run with an *oversized* fixed cache (1.5 GB) under the 2.5 GB budget was refused at startup by the memory guard, whose estimate was 3.48 GB. It never generated and went through no live cut. It is not an out-of-memory crash (2.66 to 2.91 GB of RAM was free when it ran), not a speed baseline, and not evidence that fixed caching can't work: by the same estimate, a fixed cache of up to about 0.51 GB fits under 2.5 GB. That smaller fixed cache was not run.
- **Port check:** the ported engine generated the same 32 token ids as the original ExpertRelay code on the same store, prompt and settings.

Supplementary runs, not part of the primary pair: an earlier cut run (`results/adaptive_demo_20261029-200451.json`, recorded before settings were saved; 165 to 59 experts) and a reduced-budget run made during review (`results/adaptive_demo_20261029-211301.json`, 2.6 GB cut to 2.2 GB, 0.6 GB starting cache; 57 to 24 experts, 80 of 80 tokens, the same 80 token ids). See [results/EVIDENCE.md](results/EVIDENCE.md).

## Memory: what the budget does and doesn't guarantee

The budget is a **target enforced against the engine's own estimate**, not a hard ceiling on measured process memory. At startup the memory guard refuses a configuration whose estimated RAM exceeds the budget. During a run, LiveBudget sizes the cache as the budget minus the estimated rest of the process (1.987 GB for this store and settings), further capped by free RAM. Nothing measures the process and stops it at the budget.

Two measured quantities, both in decimal GB (1 GB = 10^9 bytes), sampled after every token:

- **Private memory:** what the process has committed, whether or not it is in physical RAM. Windows charges it against the commit limit (RAM plus page file).
- **Working set:** the part currently in physical RAM. Windows can trim it.

| Cut run | Budget | Engine estimate | Measured private memory | Measured working set |
|---|---|---|---|---|
| Tokens 1 to 19, after the token-1 resize | 3.5 GB | 2.958 GB | up to 3.126 GB | up to 2.577 GB |
| Tokens 20 to 80 | 2.5 GB | 2.499 GB | 2.659 to 2.660 GB | 2.118 to 2.143 GB |

After the cut the estimate met the 2.5 GB target, but measured private memory was 2.659 GB, 0.159 GB over it. The working set stayed under it. The reduced-budget supplementary run shows the same pattern: 2.2 GB target, 2.354 GB private memory after its cut.

**Observed memory use at the default demo settings** (3.5 GB starting budget, 1.5 GB starting cache): private memory up to 3.19 GB and working set up to 2.67 GB in the uncut run. These are per-token samples; the peak during the prompt's first pass isn't captured.

**Headroom guidance, not a guarantee.** Two different limits apply on Windows. Commit headroom (commit limit minus memory already committed) must cover the process's private memory, or allocations fail. Available physical RAM decides how large LiveBudget lets the cache grow and how much the system pages. Check both before a run, and leave margin above the figures above; other programs' memory use changes during a run. If headroom is short, lower `--start-gb`, `--start-cache-gb` and `--cut-gb` (the reduced-budget run above used 2.6 GB, 0.6 GB and 2.2 GB).

```
powershell -Command "Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit','\Memory\Available MBytes'"
```

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

PASS requires two distinct files: exactly one uncut reference and one run with a scheduled cut inside the sequence that lowered the budget, with a recorded cache shrink at that token. Both runs must have every requested token (token-id list, reported count and logs all complete), valid token ids, no recorded error, all required settings with valid values, identical settings, and identical token ids. Anything else, including unreadable or malformed files, is a FAIL with exit code 1. The regression tests for these rules use only the standard library:

```
python -m unittest discover -s tests -v
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

Each run takes about two to three minutes on this laptop and writes a new `results/adaptive_demo_<timestamp>.json`. Check commit headroom and free RAM first (see the memory section above); at these defaults the process committed up to 3.19 GB. If `LEANMOE_MODELS_ROOT` is unset, the engine looks in `models/` at the repo root, which is gitignored.
