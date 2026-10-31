# Evidence files

Every file here is raw output of a script in `eval/`, written on one Windows laptop (Intel Core i5-12450H, 8.30 GB RAM, NVMe SSD), with the Qwen1.5-MoE-A2.7B-Chat int8 store, the numpy backend and the fused int8 kernel. None has been edited after it was written. Every LeanMoE measurement in the README comes from a file listed here; figures from ExpertRelay are labeled as historical. "Private memory" is what the process has committed (charged against Windows' commit limit); "working set" is the part in physical RAM. GB means 10^9 bytes.

## Task-success evaluation (`eval/task_benchmark.py`)

The [fixed 12-task specification](../configs/tasks/task_benchmark_v1.json), [specification lock](task_benchmark_spec_lock.json), [runner](../eval/task_benchmark.py), [scorer](../eval/task_scoring.py) and [method/reproduction guide](../docs/TASK_BENCHMARK.md) are committed. The [three-task pilot](task_benchmark_pilot_20261030-134509.json) checked the harness: first request and warm repeat in each mode, all three pilot tasks rule-correct and EOS-completed in both modes. A [separate manual audit](task_benchmark_pilot_manual_review.json) checked each of its twelve generated texts against the fixed answers and found no disagreement. It is **not** a twelve-task quality result. The pilot records the task-file hash used before it ran; the first Git lock was made after that pilot. The numerical/choice rules are unchanged, but the `Answer:` detector was tightened after the pilot to reject non-answer labels.

The task study grades answers and 120-second post-load task latency, while the throughput and 80-token records below measure different things. A `correct` rule score must be read with stop reason and manual audit; a timeout is neither a completion nor a precise latency. No generated code is executed by the scorer.

## Budget-cut demo (`eval/adaptive_demo.py`)

Same prompt (52 tokens with the chat template), 80 new tokens, greedy, prefetch on.

| File | Role | Engine accounting | OpenBLAS threads | Start budget, start cache | Cut | Cache at the cut | Private memory after the cut |
|---|---|---|---|---|---|---|---|
| [Uncut reference](adaptive_demo_20261030-060626.json) | **Primary**: uncut reference | corrected | 4 (recorded) | 3.5 GB, 1.3 GB | none | 109 experts throughout | 2.825 GB |
| [Adaptive run](adaptive_demo_20261030-060728.json) | **Primary**: cut run | corrected | 4 (recorded) | 3.5 GB, 1.3 GB | 2.5 GB after token 20 | 124 to 59 | 2.390 GB |
| [Uncut reference - corrected accounting, 12 threads](adaptive_demo_20261029-220246.json) | Corrected accounting, 12 threads: uncut | corrected | 12 | 3.5 GB, 1.3 GB | none | 142 experts throughout | 3.381 GB |
| [Adaptive run - corrected accounting, 12 threads](adaptive_demo_20261029-220417.json) | Corrected accounting, 12 threads: cut | corrected | 12 | 3.5 GB, 1.3 GB | 2.5 GB after token 20 | 127 to 37 | 2.467 to 2.468 GB |
| [Adaptive run - 2.7 GB budget](adaptive_demo_20261029-220548.json) | 2.7 GB budget | corrected | 12 | 3.0 GB, 0.8 GB | 2.7 GB after token 20 | 87 to 60 | 2.668 GB |
| [Uncut reference - original accounting](adaptive_demo_20261029-205033.json) | Original accounting: uncut | before correction | 12 | 3.5 GB, 1.5 GB | none | 120 throughout | 3.189 GB |
| [Adaptive run - original accounting](adaptive_demo_20261029-205224.json) | Original accounting: cut | before correction | 12 | 3.5 GB, 1.5 GB | 2.5 GB after token 20 | 112 to 59 | 2.659 to 2.660 GB |
| [Independent review - 2.2 GB budget](adaptive_demo_20261029-211301.json) | Supplementary: reduced budget, independent review | before correction | 12 | 2.6 GB, 0.6 GB | 2.2 GB after token 20 | 57 to 24 | 2.354 GB |
| [Initial adaptive run - original accounting](adaptive_demo_20261029-200451.json) | Supplementary: first cut run | before correction | 12 | 3.5 GB, 1.5 GB | 2.5 GB after token 20 | 165 to 59 | 2.659 GB |

All nine runs generated 80 of 80 tokens without an error, and all nine have the same 80 generated token ids. That is one prompt and one configuration family, not a general guarantee.

- **Strict pairs.** All three pairs pass `eval/compare_artifacts.py` (distinct files, one uncut and one cut run, a recorded shrink at the scheduled token, complete runs, identical settings and token ids). The first cut run predates settings recording, and the review and second-budget runs have no uncut partner with matching settings, so the checker can't pair them.
- **"Before correction" and "corrected".** The earlier files were made with the engine as ported (commits 25aba31 to 65817d7), whose estimate used a fixed 200 MiB interpreter baseline: 1.99 GB for everything but the cache. With the baseline measured at import, that estimate is 2.18 GB with 12 OpenBLAS threads and 1.99 GB with 4. See DECISIONS.md, "Memory accounting".
- **OpenBLAS threads.** Only the primary pair records the effective count in its settings. The earlier runs used numpy's default of one thread per logical core (12 here); their import-time private memory (0.40 GB, against 0.13 GB with 4) is consistent with that.
- **Time to first token** was 9.5 and 9.9 s in the primary pair, against 27.1 and 28.1 s in the corrected-accounting pair (12 threads).
- **Cache before the cut** differs between runs with the same settings because LiveBudget also caps the cache by free RAM when it applies the starting budget after token 1.
- **Memory sampling.** Every file has a per-token log (private memory and working set after each token). The corrected files also have `memory_samples`: sampled peaks per phase (load, prefill, decode, resize) every 50 ms. Sampled peaks can miss shorter allocations, and the decode phase covers tokens before and after the cut.
- **The "fixed" section** in each file is a fresh run with an oversized fixed cache (that run's starting cache) under its cut budget. The memory guard refused it at startup on its estimate. It never generated: it is not an out-of-memory crash and not a speed baseline, and a fixed cache sized for the smaller budget would start.

Check a strict pair from the saved files (standard library only; this does not rerun inference):

```
python eval/compare_artifacts.py results/adaptive_demo_20261030-060626.json results/adaptive_demo_20261030-060728.json
python eval/compare_artifacts.py results/adaptive_demo_20261029-220246.json results/adaptive_demo_20261029-220417.json
python eval/compare_artifacts.py results/adaptive_demo_20261029-205033.json results/adaptive_demo_20261029-205224.json
```

## Prefetch on versus off (`eval/prefetch_ablation.py`)

Prompts fixed before running (`fact`, `code`, `explain`; texts in the script and the files), 48 new tokens, greedy, fixed cache with no budget file, adaptive prefetch on versus off, everything else from `configs/runtime_cache_chat.json`. Two runs per configuration per prompt, order alternating, each a fresh process, one at a time. Memory sampled every 50 ms from the parent process. Engine before the accounting correction (it doesn't affect a run without a budget file), 12 OpenBLAS threads.

| File | Cache | Prefetch on: tokens/s | Prefetch off: tokens/s | On faster in |
|---|---|---|---|---|
| [Prefetch comparison - 1.1 GB cache](prefetch_ablation_20261029-215951.json) | 1.1 GB, 126 experts | fact 1.506, 1.473; code 1.715, 1.748; explain 1.803, 1.696 | fact 1.414, 1.440; code 1.572, 1.611; explain 1.637, 1.625 | 6 of 6 pairs |
| [Exploratory prefetch comparison - 0.6 GB cache](prefetch_ablation_20261029-214649.json) (exploratory) | 0.6 GB, 69 experts | fact 1.259, 1.220; code 1.656, 1.375; explain 0.803, 1.370 | fact 1.437, 0.618; code 0.425, 1.316; explain 0.630, 1.248 | 5 of 6 pairs |

- **1.1 GB:** compute time per token stayed between 0.12 and 0.17 s and commit headroom between 4.45 and 4.56 GB, so conditions were steady. Prefetch on was 4.3% (fact), 8.9% (code) and 7.3% (explain) faster on average, and for each prompt the slower prefetch run beat the faster no-prefetch run. Blocking disk reads per token were 83 to 90 with prefetch off and 54 to 68 with it on. Time to first token was not consistently different (20 to 32 s, noisy).
- **0.6 GB:** compute time per token ranged from 0.13 to 0.97 s and commit headroom from 3.55 to 7.17 GB, so other programs were competing for the machine. The cache was smaller than the 96 experts a token uses, so with prefetch off it got no hits at all. Treat it as exploratory.
- Token ids were identical across all four runs of each prompt in both files. Sampled peak private memory was 2.76 GB in every 0.6 GB-cache run and 3.25 to 3.26 GB in every 1.1 GB-cache run.
- ExpertRelay's earlier benchmark (base store, 1.25 GB cache) found the cache alone faster than adaptive prefetch with this kernel. That did not reproduce here; the stores, prompts and code versions differ.

## BLAS threads and memory-mapping (`eval/config_compare.py`)

Same three prompts as the prefetch experiment, 48 new tokens, greedy, adaptive prefetch on, fused kernel. Each run is a fresh child process with its thread settings in the environment, one at a time, with a 600 s timeout and a stop if commit headroom fell below 0.3 GB (no run was stopped). Memory sampled every 50 ms from the parent over the child's process tree. Timing: load is `load_model`; time to first token is the prefill forward pass; decode tokens/s is (tokens - 1) / time of the other forward passes. Expert bytes are what the engine requested; device bytes are the operating system's physical-read counter over each step, system-wide.

**BLAS threads, default (12) versus 4**, at a 3.0 GB total budget. The cache follows the normal calculation (budget minus the estimate of the rest of the process), so the two configurations differ in threads and cache size together.

| File | Prompt | 12 threads: first token, decode | 4 threads: first token, decode |
|---|---|---|---|
| [Thread comparison - explanation prompt](config_compare_blas_explain_20261030-053657.json) | explain | 19.6 s, 0.538 tok/s; 21.6 s, 0.823 tok/s | 8.7 s, 1.080 tok/s; 28.8 s, 1.314 tok/s |
| [Thread comparison - fact and code prompts](config_compare_blas_confirm_20261030-054506.json) | fact | 32.3 s, 1.443; 35.0 s, 1.333 | 7.5 s, 1.025; 7.1 s, 1.250 |
| (same file) | code | 24.6 s, 1.793; 23.1 s, 1.672 | 7.9 s, 1.973; 7.6 s, 2.005 |

- Effective OpenBLAS threads were read from the library in every run (12 or 4); numba stayed at 12 in both.
- Committed memory at import 0.401 against 0.131 GB; cache 95 against 117 experts; sampled peak private memory 2.98 against 2.90 GB; sampled peak working set 2.43 to 2.56 against 2.63 to 2.76 GB.
- Token ids identical within each prompt.
- The first file's 12-thread runs started with 0.89 and 3.28 GB of free physical RAM and spent 1.0 to 1.6 s per token reading experts; their decode figures are affected by paging.

**Memory-mapped (no expert cache, no prefetch) versus cached**, both with 4 OpenBLAS threads, at a 3.0 GB budget for the cached mode (117 experts).

| File | Prompt | Memory-mapped: first token, decode | Cached: first token, decode |
|---|---|---|---|
| [Paging comparison - pilot](config_compare_mmap_pilot_20261030-054758.json) (pilot) | explain | 19.5 s, 0.777 tok/s | 9.2 s, 1.454 tok/s |
| [Paging comparison](config_compare_mmap_full_20261030-060448.json) | fact | 26.0 s, 0.505; 19.0 s, 2.056 | 38.5 s, 1.477; 6.7 s, 1.883 |
| (same file) | code | 19.4 s, 0.513; 19.2 s, 1.085 | 9.8 s, 1.574; 7.3 s, 2.079 |
| (same file) | explain | 18.9 s, 0.187; 18.5 s, 0.997 | 7.3 s, 2.098; 6.8 s, 1.943 |

- Full run: decode mean 0.891 (memory-mapped) against 1.842 tok/s (cached), ratio of means 2.07; matched-pair ratios 2.92, 3.07, 11.2, 0.92, 1.92, 1.95 (median 2.44). Time to first token median 19.1 against 7.3 s; lower with the cache in 5 of 6 pairs (by 50 to 64%), higher in one (38.5 against 26.0 s).
- Sampled peak private memory 0.33 to 0.35 GB (memory-mapped; mapped file pages are not private) against 2.90 GB; sampled peak working set 2.88 to 4.74 GB against 2.62 to 2.76 GB.
- Memory-mapped runs read 0.97 to 1.24 GB per decode token from disk in the first round and 0.29 to 0.70 GB in the second, as the Windows file cache warmed between runs; the one memory-mapped run faster than its cached partner (fact, 2.056 tok/s) read 0.29 GB per token. Processes were fresh, but the system file cache was never cleared.
- Token ids identical across both modes for every prompt.

## Capability audit (`eval/capability_audit.py`)

[Capability audit - 12 threads](capability_audit_20261030-064849.json): eight fixed prompts answered by the real Chat model in one process, one after another, greedy decoding, stopping at the model's end-of-turn token or after 320 new tokens (the script's own loop; the engine's `generate()` has no end-of-sequence stop). 1.0 GB fixed cache, prefetch on. **This run used 12 OpenBLAS threads**: the script imported numpy before the engine package, so the engine's 4-thread default did not apply (fixed in the script afterwards; the file records the effective count). Expected answers were worked out independently.

| Prompt | Result |
|---|---|
| 487 x 36 | Correct (17532) |
| 3x + 7 = 2x - 5 | Correct (x = -12) |
| x^2 - 5x + 6 = 0 | **Wrong**: factored with -6 and 1, answered x = 6 or x = -1 (correct: 2 and 3) |
| Train, 180 km at 72 km/h from 2:15 pm | Correct (4:45 pm) |
| Integral of 3x^2 + 2x from 0 to 1, LaTeX input | Correct value (2), sloppy "+ C" steps, cut off at the 320-token limit |
| Python Fibonacci with a loop | Loop works for its own indexing (F(1) = 0); returns a string for invalid input; its example claims fib(20) = 6765, wrong for that indexing (4181); cut off at the limit |
| Arabic: capital of Japan | Right fact, broken language ("الcapital of Japan is Tokyo.") |
| Passage arithmetic (books remaining) | **Wrong**: said it can't be determined (correct: 18,500) |

Time to first token 10.6 to 24.7 s and decode 1.72 to 1.95 tokens/s (12 threads). Private memory returned to 3.094 to 3.095 GB before each request and peaked at 3.346 to 3.347 GB during each: no growth over the eight sequential requests. These are model answers on a handful of prompts, not a benchmark of the model's quality.

## Dashboard smoke test (`eval/dashboard_smoke.py`)

[Dashboard functional check](dashboard_smoke_20261030-074111.json): the dashboard server driven over its HTTP interface with the real Chat model, 4 OpenBLAS threads, 3.5 GB budget, 1.2 GB starting cache, prompt "What is 487 × 36?". A functional check, not a benchmark.

- Two cached runs in a row: both stopped at end-of-sequence after 20 tokens. The second reused the loaded model with the cache reset to its starting size (138 experts), but those experts still held weights from the first run, which partly explains its faster first token (5.1 against 8.6 s).
- A live budget change to 2.5 GB, sent for the second run's request id after its first token, was accepted and shrank the cache from 138 to 34 experts at token 2.
- One comparison, cached then memory-mapped, each freshly loaded, 16 tokens: identical token ids; first token 5.9 against 19.1 s; sampled peak private memory 3.24 against 0.50 GB and working set 2.86 against 4.96 GB. One run each, so no speed conclusion.
- After the comparison nothing was loaded and the process's private memory was back to 0.26 GB.
- An earlier run of this test (not kept) recorded the local store path in its file; the script now records only whether a store was found, and the test was rerun. Its results matched this one in every functional respect.
- The model's answer was **wrong** in every run: "The product of 487 and 36 is 17,052" (correct: 17,532). The capability audit's prompt "What is 487 × 36? Give the result." was answered correctly; the wording changed the answer.

## Not measured here

No comparison with a fully cold file cache (the system cache was never cleared), with other prompts or lengths, or on another machine. ExpertRelay's historical results live in that private repository and are not reproduced here.
