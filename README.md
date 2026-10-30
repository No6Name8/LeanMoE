# LeanMoE

A mixture-of-experts (MoE) model only uses a few of its experts for each token, so it doesn't need all its weights in RAM at once. LeanMoE runs Qwen1.5-MoE-A2.7B-Chat, whose int8 weights take 14.36 GB on disk, on a laptop with 8.30 GB of RAM: shared weights stay in memory, routed experts are read from the SSD when needed, a RAM cache keeps recently used experts, and a predictor starts reading the next layer's likely experts early.

This submission evaluates that design on the real model, including an **adaptive memory budget**: the budget is cut while the model is generating, the expert cache shrinks between tokens, and generation continues with the same output.

## Results at a glance

Measured on one 8 GB Windows laptop; details and caveats below and in [results/EVIDENCE.md](results/EVIDENCE.md).

- **About 2x the decode throughput of Windows memory-mapping.** Against the same store memory-mapped and paged in by Windows, the cached engine with prefetch averaged 1.842 against 0.891 tokens/s (ratio of means 2.07) and was faster in 5 of 6 matched comparisons; individual pairs ranged from 0.92 to 11.2 because memory-mapped speed depends on what Windows already has cached. Three prompts, two runs each, identical output. [`config_compare_mmap_full_20261030-060448.json`](results/config_compare_mmap_full_20261030-060448.json)
- **Budget cut mid-generation, same output.** The budget dropped from 3.5 to 2.5 GB after token 20; the expert cache shrank from 124 to 59 experts, measured private memory fell to 2.390 GB, and all 80 generated tokens match the uncut reference run. One prompt. [`adaptive_demo_20261030-060626.json`](results/adaptive_demo_20261030-060626.json) (uncut), [`adaptive_demo_20261030-060728.json`](results/adaptive_demo_20261030-060728.json) (cut)
- **First token in 9.5 and 9.9 s** in that demo pair. Of 13 other cached runs with the same thread setting, 11 took 6.7 to 9.8 s and two were slow outliers (28.8 and 38.5 s).

## Prior work: ExpertRelay

The inference engine in `src/engine/` is **existing work**, ported from ExpertRelay, a private research project by Abdullah Ali Alanazi (2026). Concept note: [doi:10.5281/zenodo.22980838](https://doi.org/10.5281/zenodo.22980838). Every ported file starts with `# Adapted from ExpertRelay (private research project, 2026)`.

From ExpertRelay: the int8 expert store format and readers, the numpy runtime with its fused int8 kernel, the expert cache, the prefetch predictor and its calibrations, the Manager, **LiveBudget** (which re-reads a budget file between tokens and resizes the cache) and the startup memory guard. Docstrings in `src/engine/` sometimes cite ExpertRelay's own docs, tests and benchmark files; those are not part of this repository.

New in LeanMoE:

- **The port:** imports, store location through `LEANMOE_MODELS_ROOT`, optional MindSpore.
- **Real-model evaluation of LiveBudget:** the budget-cut demo (`eval/adaptive_demo.py`), a strict checker for its result files (`eval/compare_artifacts.py`) and regression tests (`tests/`). ExpertRelay's own notes record LiveBudget as tested for identical tokens but not yet measured on the real model.
- **Memory-accounting correction:** the engine's estimate now counts memory committed before loading (result 2).
- **Thread tuning:** OpenBLAS defaults to 4 threads (result 3).
- **Comparison harnesses:** prefetch on/off (`eval/prefetch_ablation.py`) and BLAS threads and memory-mapped baseline (`eval/config_compare.py`).
- **The measurements** in this README.

## Quick start

Tested on Windows with Python 3.13. Use a short directory path (or enable long paths): torch has deeply nested files and failed to install into a long path.

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

**Check the saved evidence (no model needed, standard library only).** This reads the committed result files; it does not rerun inference.

```
python eval/compare_artifacts.py results/adaptive_demo_20261030-060626.json results/adaptive_demo_20261030-060728.json
python -m unittest discover -s tests -v
```

**Engine tests (no model needed, Windows).** These are ExpertRelay's own engine tests, ported with the store builder they use. Each test builds a tiny random-weight model and store in a temporary folder and runs the real engine code on it: cache resizing, the live budget changing mid-generation (shrinking to the minimum and growing back), and bit-identical output across expert sources, cache sizes and prefetch settings. They show the engine's mechanisms behave as specified on that synthetic store; they say nothing about the real model's output quality or speed. They need Windows (the unbuffered reader uses Win32) and are skipped elsewhere; a skipped test is not a passed one.

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

**Toy fallback (no model needed).** `src/synthetic_moe.py` writes 8 tiny experts to disk and times disk loads against cache hits. It is not a model, doesn't use the engine or LiveBudget, and says nothing about real-model performance.

```
python -m src.synthetic_moe
```

**Real model (needs the external store).** The Qwen1.5-MoE-A2.7B-Chat int8 store (14.36 GB) is not in this repository. The ExpertRelay store builder is included only because the engine tests use it (`src/engine/store/build_store.py`); building the real store with it hasn't been tried here and would need the original 28.6 GB bf16 checkpoint. With the store in a folder named `qwen1.5-moe-a2.7b-chat-int8` inside some directory:

```
set LEANMOE_MODELS_ROOT=C:\path\to\that\directory
python -m eval.adaptive_demo --cut-at 0
python -m eval.adaptive_demo --cut-at 20
python -m eval.adaptive_demo --compare results\<uncut run>.json results\<cut run>.json
python -m eval.prefetch_ablation --cache-gb 1.1
python -m eval.config_compare --stage blas --prompts fact code explain
python -m eval.config_compare --stage mmap --prompts fact code explain --blas 4
```

Each demo run takes about two minutes here; the prefetch experiment about 20 minutes and each `config_compare` stage about 15. At the demo's defaults the process's sampled private memory peaked at 3.18 GB. Check Windows commit headroom and free physical RAM before running (see [Limitations](#limitations)); lower `--start-gb`, `--start-cache-gb` and `--cut-gb` if headroom is short.

## Verified results

All on one laptop (Intel Core i5-12450H, 8.30 GB RAM, NVMe SSD), int8 Chat store, numpy backend, fused int8 kernel, greedy decoding, 2026-10-29 and 30. Details, raw files and the earlier runs are in [results/EVIDENCE.md](results/EVIDENCE.md).

**1. The budget can be cut mid-generation.** Primary pair, made with the current engine (corrected memory accounting, 4 OpenBLAS threads); only the cut schedule differs.

| | Uncut | Cut to 2.5 GB after token 20 |
|---|---|---|
| Evidence file | `adaptive_demo_20261030-060626.json` | `adaptive_demo_20261030-060728.json` |
| Expert cache before token 20 | 109 experts (0.95 GB) | 124 experts (1.08 GB) |
| Expert cache after token 20 | 109 experts | 59 experts (0.51 GB) |
| Measured private memory after token 20 | 2.825 GB | 2.390 GB |
| Time to first token | 9.5 s | 9.9 s |
| Tokens, errors | 80 of 80, none | 80 of 80, none |

The 80 generated token ids are identical between the two runs, for this one prompt and configuration, and identical to every earlier demo run. The cache sizes before the cut differ because LiveBudget also caps the cache by free RAM when it applies the starting budget, and free RAM differed between the runs. Earlier pairs, made with 12 OpenBLAS threads, are listed in [results/EVIDENCE.md](results/EVIDENCE.md).

**2. The memory estimate now covers what the budget is compared against.** The engine sizes the cache from an estimate of the rest of the process. That estimate used a fixed 200 MiB interpreter baseline, but importing numpy commits about 34 MB per OpenBLAS thread (381 MB with 12 threads here), almost none of it in physical RAM. The baseline is now the larger of the constant and the private memory measured when the engine is imported (401 MB here with 12 OpenBLAS threads, 132 MB with the current default of 4).

Before and after, matched: the same cut from 3.5 to 2.5 GB after token 20, 12 OpenBLAS threads (the starting cache was 1.5 GB before and 1.3 GB after, so the corrected estimate fit the starting budget; the cache after the cut is set by the budget).

| Cut run | Measured private memory after the cut | Cache after the cut |
|---|---|---|
| Before the correction | 2.660 GB (0.160 over the 2.5 GB budget) | 59 experts |
| After the correction | 2.468 GB (0.032 under) | 37 experts |

Single runs with other settings, not before/after pairs:

| Run | Budget after the cut | Measured private memory after the cut | Cache after the cut |
|---|---|---|---|
| Before the correction, 12 threads, review run | 2.2 GB | 2.354 GB (0.154 over) | 24 experts |
| After the correction, 12 threads | 2.7 GB | 2.668 GB (0.032 under) | 60 experts |
| After the correction, 4 threads (primary demo pair) | 2.5 GB | 2.390 GB (0.110 under) | 59 experts |

With 12 threads the correction cost cache space at the same budget (37 instead of 59 experts at 2.5 GB); with 4 threads the cache after the same cut was 59 experts again (result 3). Decode time for tokens 21 to 80 was 0.626 s per token after the correction and 0.637 before, a difference smaller than the per-token spread (standard deviation about 0.06 s). Memory was sampled once per token in the per-token log and every 50 ms by a background sampler; sampled peaks can miss shorter allocations. The budget is still a target the engine plans against, not a hard ceiling: nothing stops the process if it goes over.

**3. Four OpenBLAS threads instead of twelve: much faster first token, a larger cache, mixed decode speed.** Same 3.0 GB budget, cache sized by the normal budget calculation, prefetch on, 48 tokens, three prompts, two runs per configuration in alternating order. This compares two thread-and-cache configurations together, not either effect alone. Decoding runs on numba's own 12 threads in both.

| Prompt | First token, 12 threads | First token, 4 threads | Decode, 12 threads (tok/s) | Decode, 4 threads (tok/s) |
|---|---|---|---|---|
| fact | 32.3, 35.0 s | 7.5, 7.1 s | 1.443, 1.333 | 1.025, 1.250 |
| code | 24.6, 23.1 s | 7.9, 7.6 s | 1.793, 1.672 | 1.973, 2.005 |
| explain | 19.6, 21.6 s | 8.7, 28.8 s | 0.538, 0.823 | 1.080, 1.314 |

Time to first token fell to 7 to 9 s in 5 of 6 matched pairs. Decode speed went down for one prompt and up for two. With 4 threads numpy commits 0.131 GB at import instead of 0.401 GB, so the same budget leaves room for 117 cached experts instead of 95, and sampled peak private memory was slightly lower (2.90 against 2.98 GB); the sampled peak working set was higher (2.63 to 2.76 against 2.43 to 2.56 GB), because more of the cache sits in physical RAM. Why prefill got faster wasn't isolated (a likely cause is 12 OpenBLAS threads competing with numba's threads and the I/O threads). The engine now defaults to 4 OpenBLAS threads; an `OPENBLAS_NUM_THREADS` set by the user still wins, and result files record the effective count.

**4. Adaptive prefetching helps decode on the Chat store when the cache holds a token's experts.** Three short prompts chosen in advance, 48 tokens, prefetch on or off with everything else matched, two runs each in alternating order, one process at a time (12 OpenBLAS threads, before result 3).

| Cache | Prefetch on, tokens/s | Prefetch off, tokens/s | Matched pairs where on was faster |
|---|---|---|---|
| 1.1 GB (126 experts) | mean 1.657, range 1.473 to 1.803 | mean 1.550, range 1.414 to 1.637 | 6 of 6 |
| 0.6 GB (69 experts), exploratory | mean 1.281, range 0.803 to 1.656 | mean 0.946, range 0.425 to 1.437 | 5 of 6 |

At 1.1 GB, prefetch was faster by 4.3%, 8.9% and 7.3% on the three prompts, and for each prompt both prefetch runs beat both no-prefetch runs. Token ids were identical across all runs of each prompt. The 0.6 GB run was noisy: compute time per token swung from 0.13 to 0.97 s while other programs used the machine, and a 69-expert cache is smaller than the 96 experts one token uses, so the cache alone got no hits. The default (adaptive prefetch on) is unchanged. An earlier ExpertRelay benchmark on the base store, where the cache alone was fastest, did not reproduce here. **This isolated prefetch gain was measured with 12 OpenBLAS threads and has not been re-measured under the current 4-thread default.**

**5. Against Windows memory-mapping.** The engine's memory-mapped source maps the whole store and lets Windows page weights in on demand, with no expert cache and no prefetch. Same store, kernel, 4 OpenBLAS threads, three prompts and 48 tokens; two runs per mode per prompt, order rotating, one process at a time.

| | Memory-mapped | Cached with prefetch |
|---|---|---|
| Decode, tokens/s | mean 0.891, range 0.187 to 2.056 | mean 1.842, range 1.477 to 2.098 |
| Time to first token | median 19.1 s, range 18.5 to 26.0 | median 7.3 s, range 6.7 to 38.5 |
| Sampled peak private memory | 0.33 to 0.35 GB | 2.90 GB |
| Sampled peak working set | 2.88 to 4.74 GB | 2.62 to 2.76 GB |

Decode was faster with the cache in 5 of 6 matched pairs: about 2 times on average (ratio of means 2.07; median matched pair 2.44; individual pairs ranged from 0.92 to 11.2). Time to first token was 50 to 64% lower in 5 of 6 pairs; the exception was one cached run at 38.5 s. Token ids were identical across both modes for every prompt. Memory-mapping commits little private memory because the weights stay in Windows' file cache, but it took more physical RAM (up to 4.74 GB). Its speed depended on how much of the store Windows had already cached from earlier runs: one warm run reached 2.056 tok/s, faster than its cached partner. This compares the whole cached configuration (expert cache, unbuffered reads and prefetch together) with memory-mapping; it does not separate prefetch's share, which result 4 measured under the older 12-thread setting. The design itself is ExpertRelay's prior work.

## Limitations

- **Small samples on one machine.** One demo prompt and three experiment prompts, two runs per configuration, greedy decoding, one 8 GB Windows laptop with other programs running. Output equality holds for those prompts only; output quality against the original model wasn't evaluated here.
- **Timings are noisy.** Background load and Windows' file cache affect them; fresh processes are not cold-cache runs, and slow first-token outliers occurred.
- **The budget is a planning target.** The engine sizes the cache from an estimate and doesn't enforce the budget; measured private memory stayed under it in the per-token samples after each cut since the correction, but short spikes between samples aren't seen.
- **Memory headroom.** Windows commit headroom must cover the process's private memory (sampled peak 3.18 GB at the demo defaults) and free physical RAM limits the cache. Check both before running:

  ```
  powershell -Command "Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit','\Memory\Available MBytes'"
  ```

- **The fixed-cache example is a startup refusal.** Each demo run also starts a run whose fixed cache is too large for the cut budget; the memory guard refuses it on its estimate. It is not an out-of-memory crash or a speed baseline; a fixed cache sized for the smaller budget would start.
- **External, Windows-only model.** The 14.36 GB store isn't in this repository, and rebuilding it here is untested. The real-model path and the engine tests use Win32 unbuffered reads. The engine tests run on a tiny synthetic store, not the real model.

## Repository layout

- `src/engine/` — engine ported from ExpertRelay (see above). LeanMoE's changes are marked `LeanMoE change` in the code.
- `src/synthetic_moe.py` — toy fallback.
- `eval/adaptive_demo.py` — budget-cut demo and the oversized fixed-cache example.
- `eval/compare_artifacts.py` — standard-library checker for an uncut and a cut run.
- `eval/prefetch_ablation.py` — prefetch on/off experiment.
- `eval/config_compare.py` — BLAS-thread and memory-mapping comparisons.
- `tests/` — standard-library tests for the evidence checker and memory accounting; `tests/engine/` — engine tests ported from ExpertRelay (pytest, Windows), run with `python -m pytest`.
- `configs/` — engine configs, prompt set and prefetch calibrations. `runtime.json` (the engine's default config) and `runtime_cache.json` point at the base store, which these demos don't use.
- `results/` — raw result files, indexed in [results/EVIDENCE.md](results/EVIDENCE.md).
- [DECISIONS.md](DECISIONS.md) — what was decided, why, and what was rejected.
