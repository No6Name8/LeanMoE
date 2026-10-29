# Evidence files

Every file here is raw output of a script in `eval/`, written on one Windows laptop (Intel Core i5-12450H, 8.30 GB RAM, NVMe SSD) on 2026-10-29, with the Qwen1.5-MoE-A2.7B-Chat int8 store, the numpy backend and the fused int8 kernel. None has been edited after it was written. Every LeanMoE measurement in the README comes from a file listed here; figures from ExpertRelay are labeled as historical. "Private memory" is what the process has committed (charged against Windows' commit limit); "working set" is the part in physical RAM. GB means 10^9 bytes.

## Budget-cut demo (`eval/adaptive_demo.py`)

Same prompt (52 tokens with the chat template), 80 new tokens, greedy, prefetch on.

| File | Role | Engine accounting | Start budget, start cache | Cut | Cache at the cut | Private memory after the cut |
|---|---|---|---|---|---|---|
| `adaptive_demo_20261029-220246.json` | **Primary**: uncut reference | corrected | 3.5 GB, 1.3 GB | none | 142 experts throughout | 3.381 GB |
| `adaptive_demo_20261029-220417.json` | **Primary**: cut run | corrected | 3.5 GB, 1.3 GB | 2.5 GB after token 20 | 127 to 37 | 2.467 to 2.468 GB |
| `adaptive_demo_20261029-220548.json` | Second budget setting | corrected | 3.0 GB, 0.8 GB | 2.7 GB after token 20 | 87 to 60 | 2.668 GB |
| `adaptive_demo_20261029-205033.json` | Earlier pair: uncut | before correction | 3.5 GB, 1.5 GB | none | 120 throughout | 3.189 GB |
| `adaptive_demo_20261029-205224.json` | Earlier pair: cut | before correction | 3.5 GB, 1.5 GB | 2.5 GB after token 20 | 112 to 59 | 2.659 to 2.660 GB |
| `adaptive_demo_20261029-211301.json` | Supplementary: reduced budget, independent review | before correction | 2.6 GB, 0.6 GB | 2.2 GB after token 20 | 57 to 24 | 2.354 GB |
| `adaptive_demo_20261029-200451.json` | Supplementary: first cut run | before correction | 3.5 GB, 1.5 GB | 2.5 GB after token 20 | 165 to 59 | 2.659 GB |

All seven runs generated 80 of 80 tokens without an error, and all seven have the same 80 generated token ids. That is one prompt and one configuration family, not a general guarantee.

- **Strict pairs.** Both pairs pass `eval/compare_artifacts.py` (distinct files, one uncut and one cut run, a recorded shrink at the scheduled token, complete runs, identical settings and token ids). The first cut run predates settings recording, and the review and second-budget runs have no uncut partner with matching settings, so the checker can't pair them.
- **"Before correction" and "corrected".** The earlier files were made with the engine as ported (commits 25aba31 to 65817d7), whose estimate used a fixed 200 MiB interpreter baseline. The corrected files use the baseline measured at import (2.18 GB estimated for everything but the cache, against 1.99 GB before). See DECISIONS.md, "Memory accounting".
- **Cache before the cut** differs between runs with the same settings because LiveBudget also caps the cache by free RAM when it applies the starting budget after token 1.
- **Memory sampling.** Every file has a per-token log (private memory and working set after each token). The corrected files also have `memory_samples`: sampled peaks per phase (load, prefill, decode, resize) every 50 ms. Sampled peaks can miss shorter allocations, and the decode phase covers tokens before and after the cut.
- **The "fixed" section** in each file is a fresh run with an oversized fixed cache (that run's starting cache) under its cut budget. The memory guard refused it at startup on its estimate. It never generated: it is not an out-of-memory crash and not a speed baseline, and a fixed cache sized for the smaller budget would start.

Check a strict pair from the saved files (standard library only; this does not rerun inference):

```
python eval/compare_artifacts.py results/adaptive_demo_20261029-220246.json results/adaptive_demo_20261029-220417.json
python eval/compare_artifacts.py results/adaptive_demo_20261029-205033.json results/adaptive_demo_20261029-205224.json
```

## Prefetch on versus off (`eval/prefetch_ablation.py`)

Prompts fixed before running (`fact`, `code`, `explain`; texts in the script and the files), 48 new tokens, greedy, fixed cache with no budget file, adaptive prefetch on versus off, everything else from `configs/runtime_cache_chat.json`. Two runs per configuration per prompt, order alternating, each a fresh process, one at a time. Memory sampled every 50 ms from the parent process. Engine before the accounting correction (it doesn't affect a run without a budget file).

| File | Cache | Prefetch on: tokens/s | Prefetch off: tokens/s | On faster in |
|---|---|---|---|---|
| `prefetch_ablation_20261029-215951.json` | 1.1 GB, 126 experts | fact 1.506, 1.473; code 1.715, 1.748; explain 1.803, 1.696 | fact 1.414, 1.440; code 1.572, 1.611; explain 1.637, 1.625 | 6 of 6 pairs |
| `prefetch_ablation_20261029-214649.json` (exploratory) | 0.6 GB, 69 experts | fact 1.259, 1.220; code 1.656, 1.375; explain 0.803, 1.370 | fact 1.437, 0.618; code 0.425, 1.316; explain 0.630, 1.248 | 5 of 6 pairs |

- **1.1 GB:** compute time per token stayed between 0.12 and 0.17 s and commit headroom between 4.45 and 4.56 GB, so conditions were steady. Prefetch on was 4.3% (fact), 8.9% (code) and 7.3% (explain) faster on average, and for each prompt the slower prefetch run beat the faster no-prefetch run. Blocking disk reads per token were 83 to 90 with prefetch off and 54 to 68 with it on. Time to first token was not consistently different (20 to 32 s, noisy).
- **0.6 GB:** compute time per token ranged from 0.13 to 0.97 s and commit headroom from 3.55 to 7.17 GB, so other programs were competing for the machine. The cache was smaller than the 96 experts a token uses, so with prefetch off it got no hits at all. Treat it as exploratory.
- Token ids were identical across all four runs of each prompt in both files. Sampled peak private memory was 2.76 GB in every 0.6 GB-cache run and 3.25 to 3.26 GB in every 1.1 GB-cache run.
- ExpertRelay's earlier benchmark (base store, 1.25 GB cache) found the cache alone faster than adaptive prefetch with this kernel. That did not reproduce here; the stores, prompts and code versions differ.

## Not measured here

No comparison against a non-caching baseline (for example the engine's memory-mapped source) was run for LeanMoE, so there is no speedup claim against one. ExpertRelay's historical results live in that private repository and are not reproduced here.
