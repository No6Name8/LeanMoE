# LeanMoE

A mixture-of-experts (MoE) model only uses a few of its experts for each token, so it doesn't need all its weights in RAM at once. LeanMoE runs Qwen1.5-MoE-A2.7B-Chat, whose int8 weights take 14.36 GB on disk, on a laptop with 8.30 GB of RAM: shared weights stay in memory, routed experts are read from the SSD when needed, a RAM cache keeps recently used experts, and a predictor starts reading the next layer's likely experts early.

This submission's addition is an **adaptive memory budget test on the real model**: the budget is cut while the model is generating, the expert cache shrinks between tokens, and generation continues with the same output.

## Prior work: ExpertRelay

The inference engine in `src/engine/` is **existing work**, ported from ExpertRelay, a private research project by Abdullah Ali Alanazi (2026). Concept note: [doi:10.5281/zenodo.22980838](https://doi.org/10.5281/zenodo.22980838). Every ported file starts with `# Adapted from ExpertRelay (private research project, 2026)`.

From ExpertRelay: the int8 expert store format and readers, the numpy runtime with its fused int8 kernel, the expert cache, the prefetch predictor and its calibrations, the Manager, **LiveBudget** (which re-reads a budget file between tokens and resizes the cache) and the startup memory guard. Docstrings in `src/engine/` sometimes cite ExpertRelay's own docs, tests and benchmark files; those are not part of this repository.

New in LeanMoE:

- the port (imports, `LEANMOE_MODELS_ROOT`, optional MindSpore);
- the adaptive-budget demo, its strict evidence checker and regression tests;
- a prefetch on/off experiment on the Chat store;
- a memory-accounting correction in the engine's estimate (see below);
- the measurements in this README.

## Quick start

Tested on Windows with Python 3.13. Use a short directory path (or enable long paths): torch has deeply nested files and failed to install into a long path.

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

**Check the saved evidence (no model needed, standard library only).** This reads the committed result files; it does not rerun inference.

```
python eval/compare_artifacts.py results/adaptive_demo_20261029-220246.json results/adaptive_demo_20261029-220417.json
python -m unittest discover -s tests -v
```

**Toy fallback (no model needed).** `src/synthetic_moe.py` writes 8 tiny experts to disk and times disk loads against cache hits. It is not a model, doesn't use the engine or LiveBudget, and says nothing about real-model performance.

```
python -m src.synthetic_moe
```

**Real model (needs the external store).** The Qwen1.5-MoE-A2.7B-Chat int8 store (14.36 GB) is not in this repository, and the ExpertRelay store builder that produced it is not included. With the store in a folder named `qwen1.5-moe-a2.7b-chat-int8` inside some directory:

```
set LEANMOE_MODELS_ROOT=C:\path\to\that\directory
python -m eval.adaptive_demo --cut-at 0
python -m eval.adaptive_demo --cut-at 20
python -m eval.adaptive_demo --compare results\<uncut run>.json results\<cut run>.json
python -m eval.prefetch_ablation --cache-gb 1.1
```

Each demo run takes two to three minutes here; the prefetch experiment takes about 20 minutes. At the demo's defaults the process's sampled private memory peaked at 3.45 GB. Check Windows commit headroom and free physical RAM before running (see [Limitations](#limitations)); lower `--start-gb`, `--start-cache-gb` and `--cut-gb` if headroom is short.

## Verified results

All on one laptop (Intel Core i5-12450H, 8.30 GB RAM, NVMe SSD), int8 Chat store, numpy backend, fused int8 kernel, greedy decoding, 2026-10-29. Details, raw files and the earlier runs are in [results/EVIDENCE.md](results/EVIDENCE.md).

**1. The budget can be cut mid-generation.** Primary pair, made after the accounting correction; only the cut schedule differs.

| | Uncut | Cut to 2.5 GB after token 20 |
|---|---|---|
| Evidence file | `adaptive_demo_20261029-220246.json` | `adaptive_demo_20261029-220417.json` |
| Expert cache before token 20 | 142 experts (1.23 GB) | 127 experts (1.10 GB) |
| Expert cache after token 20 | 142 experts | 37 experts (0.32 GB) |
| Measured private memory after token 20 | 3.381 GB | 2.467 to 2.468 GB |
| Tokens, errors | 80 of 80, none | 80 of 80, none |

The 80 generated token ids are identical between the two runs, for this one prompt and configuration. The cache sizes before the cut differ because LiveBudget also caps the cache by free RAM when it applies the starting budget, and free RAM differed between the runs. A second cut, 3.0 GB to 2.7 GB, shrank the cache from 87 to 60 experts with the same 80 token ids.

**2. The memory estimate now covers what the budget is compared against.** The engine sizes the cache from an estimate of the rest of the process. That estimate used a fixed 200 MiB interpreter baseline, but importing numpy commits about 34 MB per OpenBLAS thread (381 MB with 12 threads here), almost none of it in physical RAM. The baseline is now the larger of the constant and the private memory measured when the engine is imported (401 MB here).

| Cut run | Budget after cut | Measured private memory after cut | Cache after cut |
|---|---|---|---|
| Before the correction | 2.5 GB | 2.660 GB (0.160 over) | 59 experts |
| After | 2.5 GB | 2.468 GB (0.032 under) | 37 experts |
| Before the correction | 2.2 GB | 2.354 GB (0.154 over) | 24 experts |
| After | 2.7 GB | 2.668 GB (0.032 under) | 60 experts |

The price is a smaller cache at the same budget (37 instead of 59 experts at 2.5 GB). Decode time for tokens 21 to 80 was 0.626 s per token after the correction and 0.637 before, a difference smaller than the per-token spread (standard deviation about 0.06 s). Memory was sampled once per token in the per-token log and every 50 ms by a background sampler; sampled peaks can miss shorter allocations. The budget is still a target the engine plans against, not a hard ceiling: nothing stops the process if it goes over.

**3. Adaptive prefetching helps decode on the Chat store when the cache holds a token's experts.** Three short prompts chosen in advance, 48 tokens, prefetch on or off with everything else matched, two runs each in alternating order, one process at a time.

| Cache | Prefetch on, tokens/s | Prefetch off, tokens/s | Matched pairs where on was faster |
|---|---|---|---|
| 1.1 GB (126 experts) | mean 1.657, range 1.473 to 1.803 | mean 1.550, range 1.414 to 1.637 | 6 of 6 |
| 0.6 GB (69 experts), exploratory | mean 1.281, range 0.803 to 1.656 | mean 0.946, range 0.425 to 1.437 | 5 of 6 |

At 1.1 GB, prefetch was faster by 4.3%, 8.9% and 7.3% on the three prompts, and for each prompt both prefetch runs beat both no-prefetch runs. Token ids were identical across all runs of each prompt. The 0.6 GB run was noisy: compute time per token swung from 0.13 to 0.97 s while other programs used the machine, and a 69-expert cache is smaller than the 96 experts one token uses, so the cache alone got no hits. The default (adaptive prefetch on) is unchanged. An earlier ExpertRelay benchmark on the base store, where the cache alone was fastest, did not reproduce here.

**Not shown:** no speedup against a non-caching baseline was measured for this submission. ExpertRelay's own historical comparisons (for example against memory-mapping) used the base store and older code and are not repeated here.

## Limitations

- **One machine, few prompts.** Output equality was tested on one demo prompt (and three prefetch prompts) with greedy decoding; it is not a general correctness guarantee. Timings vary with background load.
- **The budget is a planning target.** LiveBudget and the startup guard work from an estimate. After the correction, measured private memory stayed under the cut budget in every per-token sample after each cut, and the 50 ms sampler's loading and prefill peaks stayed under the starting budget, but nothing enforces it and short spikes between samples aren't seen.
- **Two Windows memory limits.** Commit headroom (commit limit minus memory already committed) must cover the process's private memory, or allocations fail. Free physical RAM decides how large LiveBudget lets the cache grow and how much the system pages. Check both:

  ```
  powershell -Command "Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit','\Memory\Available MBytes'"
  ```

- **The oversized fixed-cache example is a startup refusal.** Each demo run also starts a fresh run whose fixed cache is too large for the cut budget; the memory guard refuses it on its estimate. That is not an out-of-memory crash, not a speed baseline, and not evidence that fixed caching can't work: a fixed cache sized for the smaller budget would start.
- **The model store is external** and can't be rebuilt from this repository.
- **Windows only** for the real model: the unbuffered expert reader uses Win32 APIs.

## Repository layout

- `src/engine/` — engine ported from ExpertRelay (see above). LeanMoE's changes are marked `LeanMoE change` in the code.
- `src/synthetic_moe.py` — toy fallback.
- `eval/adaptive_demo.py` — budget-cut demo and the oversized fixed-cache example.
- `eval/compare_artifacts.py` — standard-library checker for an uncut and a cut run.
- `eval/prefetch_ablation.py` — prefetch on/off experiment.
- `tests/` — standard-library regression tests.
- `configs/` — engine configs, prompt set and prefetch calibrations. `runtime.json` (the engine's default config) and `runtime_cache.json` point at the base store, which these demos don't use.
- `results/` — raw result files, indexed in [results/EVIDENCE.md](results/EVIDENCE.md).
- [DECISIONS.md](DECISIONS.md) — what was decided, why, and what was rejected.
