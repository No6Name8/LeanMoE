# Decisions Log

Each entry: **Decision** (what was chosen), **Why**, **Rejected** (what wasn't, and why). All dates are 2026-10-29 unless stated.

## Project scope
**Decision:** Single-machine (8 GB laptop) MoE expert caching and prefetching, not the multi-device ExpertRelay design.
**Why:** Only one 8 GB machine was available for the hackathon; multi-device synchronization adds a networking failure mode that couldn't be debugged in 48 hours.
**Rejected:** A multi-device version, for the infrastructure risk.

## Challenge
**Decision:** Challenge 2, "Deliver AI with Less Compute".
**Why:** No external dataset is provided for any challenge, so all three are judged on self-defined rigor. Challenge 2 comes with objective measures (time, memory) that are easy to measure and hard to fake.

## Model and store
**Decision:** Use the Qwen1.5-MoE-A2.7B-Chat int8 store that ExpertRelay had already built (14.36 GB, kept outside the repository), not OLMoE-1B-7B.
**Why:** A first OLMoE load test (through transformers) was stopped while still downloading; the model needs about 14 GB in fp16 on an 8 GB machine. The Qwen store already existed, runs under a memory budget with an expert cache, and generated correctly on this laptop.
**Rejected:** OLMoE through transformers (no evidence it fits, and no expert-level cache control). Building a new store (it needs the 28.6 GB bf16 checkpoint; the builder module was later ported for the engine tests only, and building a real store with it is untested here).
**Note:** The OLMoE test script (`test_load.py`) and its transformers/accelerate requirements were removed in the repository cleanup; it remains in the Git history.

## Synthetic fallback, labeled
**Decision:** Keep `src/synthetic_moe.py` as a toy that runs without a model store.
**Why:** Reviewers won't have the 14 GB store; this is the only part they can run as-is.
**Rejected:** Presenting it as evidence of real-model performance. It times disk loads against cache hits on 256x256 toy layers only.

## Port approach
**Decision:** Copy the ExpertRelay engine into `src/engine/` with an attribution header on every file, rewrite `expertrelay.` imports to `src.engine.`, locate stores through `LEANMOE_MODELS_ROOT`, make MindSpore optional, and include the seven extra files the requested ones import. Later LeanMoE changes in ported files are marked `LeanMoE change`.
**Why:** The engine (store, cache, predictor, Manager, LiveBudget, memory guard) is prior work; porting it unchanged keeps its behavior and allowed a token-for-token check against the original. It matched: the same 32 token ids on the same store, prompt and settings.
**Rejected:** Rewriting the engine (no time, nothing to verify against). Depending on ExpertRelay as an installed package (it is private).

## Adaptive demo design
**Decision:** Drive the existing LiveBudget through a budget file, starting at 3.5 GB and cutting to 2.5 GB after token 20, logging every token. The demo writes the file itself at a token boundary.
**Why:** The number in the budget file is the memory budget for the whole engine process, not the cache; the cache gets the budget minus the estimated rest of the process. A cut from 2.0 GB to 1.0 GB (the first plan) would have left the cache at its 13-expert minimum both times. LiveBudget only reads the file between tokens, so a write from another process during a token takes effect at the same point.
**Rejected:** Running the engine as a subprocess and writing the file from outside (no way to see token progress).

## Oversized fixed-cache example
**Decision:** Each demo run also starts a fresh run with an oversized fixed cache (the starting cache) under the cut budget; the memory guard refuses it at startup.
**Why:** It shows that a cache sized for the larger budget can't be kept when the budget shrinks, while the adaptive run shrinks instead.
**Rejected:** Calling the refusal an out-of-memory crash (the guard compares its estimate with the budget; plenty of RAM was free). Treating it as a speed baseline or as proof that fixed caching can't work: a fixed cache sized for the smaller budget would start. (An earlier version of this entry said a fixed cache's only option is to refuse; corrected after the independent review.)

## Evidence checker
**Decision:** `eval/compare_artifacts.py`, standard library only and shared with `eval/adaptive_demo.py --compare`. PASS requires two distinct files, exactly one uncut reference and one cut run whose scheduled cut lies inside the sequence, lowers the budget and records a cache shrink; complete runs (token ids, reported count and logs all match the request); valid token ids and setting values; identical settings; identical token ids. Malformed files are a FAIL, not a traceback. Covered by `tests/test_compare_artifacts.py`.
**Why:** "Outputs don't depend on cache size" was a claim in ExpertRelay's docstring, not a measurement; and the first version of the checker passed invalid evidence (found in the independent review).
**Result:** Both strict pairs pass, with identical 80-token outputs, for the tested prompt and configuration.
**Rejected:** Generalizing this into a correctness guarantee.

## Dependencies
**Decision:** requirements.txt lists the third-party packages the code imports: numpy, numba, psutil, tokenizers and jinja2 for the engine, torch for the toy fallback only.
**Why:** A reviewer runs `pip install -r requirements.txt` first. It was verified in a fresh virtual environment (clean install, `pip check` clean, all project modules import, the demo ran in it).
**Note:** A first fresh install failed only because its directory path exceeded Windows' path limit (torch has deeply nested files); the README tells Windows users to use a short path.

## Prefetch on or off
**Decision:** Keep adaptive prefetch on (the configs' default).
**Why:** A matched experiment on the Chat store (`eval/prefetch_ablation.py`; three prompts fixed in advance, 48 tokens, two runs per configuration per prompt, order alternating, fresh process each) found prefetch on faster in 6 of 6 matched pairs with a 1.1 GB cache (126 experts): mean 1.657 against 1.550 tokens/s, 4.3% to 8.9% per prompt, with identical token ids. This was the lead to check because ExpertRelay's earlier benchmark (base store, 1.25 GB cache) had found the cache alone fastest; that did not reproduce here.
**Rejected:** Turning prefetch off. An exploratory run with a 0.6 GB cache was too noisy to decide anything (other programs competed for the CPU, and a 69-expert cache can't hold one token's 96 experts), though prefetch was faster in 5 of 6 pairs there too.

## Memory accounting
**Decision:** The engine's estimate of everything except the cache now uses, as its interpreter baseline, the larger of the old fixed 200 MiB and the private memory measured when `src/engine/runtime/generate.py` is imported (before any weights load). Marked `LeanMoE change` in the code; tested in `tests/test_memory_accounting.py`.
**Why:** After a cut to 2.5 GB, measured private memory was 2.659 GB, 0.16 GB over the budget, because the estimate missed memory committed before loading. Stage-by-stage measurement found it: importing numpy commits about 34 MB per OpenBLAS thread (381 MB with 12 threads, 10 MB with 1 thread), almost none of it in the working set; the fixed baseline had been measured at about 150 MB on ExpertRelay's machine. With the correction the estimate rose from 1.99 to 2.18 GB, and after cuts to 2.5 GB and 2.7 GB measured private memory was 2.468 and 2.668 GB, 0.032 GB under each budget. The cost: at 2.5 GB the cache holds 37 experts instead of 59; decode time for tokens 21 to 80 did not change measurably (0.626 against 0.637 s per token, spread about 0.06 s). The demo's default starting cache dropped from 1.5 to 1.3 GB so the corrected estimate fits the 3.5 GB starting budget. (With 12 OpenBLAS threads. The 4-thread default adopted the next day lowers the measured baseline and recovers the cache space: 59 experts at 2.5 GB again, with 2.390 GB measured.)
**Rejected:** Adding the observed 0.16 GB as a constant (it depends on the machine's thread count). Capping OpenBLAS threads (changes prefill performance and needs its own experiment). Claiming a hard memory ceiling: the budget is still a planning target, and short spikes between samples aren't measured. Smaller remaining differences were left as they are: the tokenizer object (about 65 MB) isn't in the estimate, and model loading measured about 90 MB below its estimated share (resident weights, cache and kernel allowance together; not broken down further).

## BLAS threads (2026-10-30)
**Decision:** Default OpenBLAS to 4 threads (`src/engine/__init__.py`, set before numpy is imported; a value the user sets wins). Result files record the effective count.
**Why:** At the same 3.0 GB budget, three prompts, two runs each, alternating (`eval/config_compare.py --stage blas`): numpy committed 0.131 GB at import instead of 0.401 GB, so the cache held 117 experts instead of 95, and time to first token fell from 19 to 35 s to 7 to 9 s in 5 of 6 matched pairs. Sampled peak private memory was slightly lower (2.90 against 2.98 GB). The fresh demo pair with this default passed the strict checker with the same 80 token ids, a 59-expert cache after the 2.5 GB cut, 2.390 GB private memory after it, and 9.5 to 9.9 s to first token (27 to 28 s with 12 threads).
**Rejected:** Claiming a decode speedup: decode tokens/s fell for one prompt (fact, about 20%) and rose for two. Calling it a pure threading or pure cache effect: the comparison changes both together. Explaining the prefill gain without a test (thread oversubscription with numba's 12 threads and the I/O threads is plausible but unmeasured).
**Cost:** The sampled peak working set was higher (2.63 to 2.76 against 2.43 to 2.56 GB), because more cache sits in physical RAM.

## Memory-mapped baseline (2026-10-30)
**Decision:** Report a fresh comparison against the engine's memory-mapped source (Windows paging, no expert cache, no prefetch), both modes with 4 OpenBLAS threads, three prompts, two runs per mode, rotating order.
**Why:** The submission had no measured baseline. Result: decode mean 1.842 against 0.891 tok/s (ratio of means 2.07, median matched pair 2.44, cached faster in 5 of 6 pairs); median time to first token 7.3 against 19.1 s (lower in 5 of 6 pairs); identical token ids. Memory-mapping used far less private memory (0.33 to 0.35 against 2.90 GB) but more physical RAM (working set up to 4.74 against 2.76 GB).
**Rejected:** A single headline multiplier: matched pairs ranged from 0.92 to 11.2 because memory-mapped speed depends on how much of the store Windows has cached (one warm memory-mapped run beat its cached partner). Clearing the system cache to force cold runs (not done, by instruction). Crediting the speed difference to this submission: the cache, unbuffered reads and prefetch are ExpertRelay's design.

## Engine tests (2026-10-30)
**Decision:** Port three of ExpertRelay's pytest files into `tests/engine/` (`test_expert_cache.py`, `test_runtime_auto.py`, `test_runtime_correctness.py`), the helper that builds a test store, and the store builder module it calls (`src/engine/store/build_store.py`). Imports rewritten, attribution headers added, assertions unchanged. pytest is a development-only requirement (`requirements-dev.txt`).
**Why:** Nothing in the repository tested the engine's cache resizing, LiveBudget or output preservation; the real-model demo needs a 14 GB store reviewers won't have. These tests build a tiny random-weight store in a temporary directory and run the real engine on it. Result on Windows: 63 passed, 0 failed, 0 skipped (with the 40 existing tests, 103 passed). The committed result files were hash-checked before and after: unchanged.
**Rejected:** Porting `conftest.py` and the tokenizer fixture (none of the three files uses them), the test against Hugging Face transformers (needs transformers) and the store-pipeline and benchmark tests (outside the demo's scope). Placing the pytest files directly in `tests/`: the standard-library `python -m unittest discover -s tests` would then fail to import them without pytest.

## Reproducibility for reviewers (2026-10-30)
**Decision:** A two-path "Start here" in the README. Path 1 (quick verification) is a PowerShell setup script (`scripts/setup.ps1`: project-local `.venv`, `requirements-dev.txt`, preflight), the engine tests on a synthetic store, and the saved-evidence checker. Path 2 (full model) documents the store, memory and time requirements and the builder commands, marked not yet verified. A preflight script (`scripts/preflight.py`) reports OS, Python, packages, physical RAM, Windows commit headroom, disk, threads and store completeness. Windows-only support is stated explicitly.
**Why:** Reviewers will clone and run it on their own machines, without the 14 GB store or the private ExpertRelay checkout.
**Changes made for it:** torch moved out of `requirements.txt` into `requirements-toy.txt` (only the toy fallback imports it; it is the large, path-length-sensitive install). The store builder writes its report and build record into the store directory instead of `docs/` (absent here) and the results folder (committed evidence, named after the machine). The demo's docstring no longer contains a local path.
**Rejected:** Claiming Linux or macOS support (the engine needs Win32 unbuffered reads; tests merely skip there). Calling real-model reproduction verified: the builder has not been run from LeanMoE. Hosting a prebuilt store now: the Tongyi Qianwen license permits redistributing derivatives with conditions (license copy, change notices, attribution notice), but whether to host 14 GB is the author's decision.

## Streaming generation foundation (2026-10-30)
**Decision:** A new `stream_generate` (`src/engine/runtime/streaming.py`) beside the unchanged `generate()`: same forward pass and greedy choice, one event per token, a final event with the stop reason and timings. It checks prompt + output length against the context size before allocating, stops at the given end-of-sequence ids or the output limit, checks a cancel token between forward passes and reports how long a cancel took and in which phase, and releases the request's resources (KV cache; no expert in use, no queued prefetch) on completion, cancellation, error or an abandoned iterator. `IncrementalDecoder` (`text_stream.py`, standard library only) releases text without splitting multi-byte characters. `eval/stream_demo.py` is a command-line example.
**Why:** The dashboard needs real streaming output, normal stopping and cancellation; `generate()` runs a fixed number of tokens for the benchmarks and stays that way.
**Evidence:** 15 synthetic-store tests (tokens identical to `generate()` up to the stop, end-of-sequence, limits, three length refusals with no allocation, cancellation between tokens, during prefill and before start, an injected mid-layer read failure, consecutive requests) and 7 decoding tests. Disabling the release step makes the mid-layer failure test fail, so it checks the cleanup. Real model, two runs: "What is 487 x 36?" streamed and stopped at end-of-sequence after 20 tokens (first token 8.28 s, 1.94 tokens/s) with a wrong answer, 17,052 (correct: 17,532; the audit's "487 × 36" prompt was answered correctly); a cancel requested 3 s into prefill took effect 4.60 s later, after prefill, reported as such.
**Rejected:** Interrupting a forward pass mid-way (the engine has no safe point inside one; the delay is reported instead). Changing `generate()` (benchmarks depend on fixed-length output).

## Local evaluation dashboard (2026-10-30)
**Decision:** A standard-library HTTP server (`src/dashboard/server.py`) on 127.0.0.1 only, with one worker thread that owns at most one loaded model, and one static page (`src/dashboard/static/`). Overlapping runs get HTTP 409. Switching source or mode unloads the current model first. Every run resets the expert cache to its starting size; cancellation and budget changes must name the active request id. Inputs (prompt, source, mode, token limit, budget, starting cache, tokenized length against the 512-token context, and the engine's memory estimate) are validated before a run starts. A closed page cancels its run (detected within about a second through a heartbeat, even during prefill). Requests with a foreign Host or Origin, or a non-JSON body, are refused; a port another program already uses is refused. Model and prompt text reach the page as JSON strings and are inserted as text; math is typeset by a bundled KaTeX with trust disabled, under a strict Content-Security-Policy.
**Why:** Reviewers need to run their own prompts on the engine and see measured behaviour, with or without the 14 GB store.
**Evidence:** 29 synthetic-model tests (`tests/engine/test_dashboard.py`), 2 Node.js checks of the page scripts and math rendering (`tests/test_dashboard_js.py`), and the real-model smoke test above. Two defects were found and fixed while testing: a refusal sent before reading the request body reset the connection on Windows, and binding 127.0.0.1 succeeded on a port another program held on all interfaces. Mutation checks: with disconnect cancellation removed, the page-close test fails.
**Rejected:** A web framework (no new dependency needed). Rendering model output as Markdown or HTML (untrusted). Correcting or annotating model answers (output is shown as generated and labelled unverified). A synthetic substitute when the real store is missing (the synthetic demonstration is a separate, labelled choice). Exposing the server beyond 127.0.0.1.

## Dashboard redesign and browser check (2026-10-30)
**Decision:** Three tabs (Run, Compare, Evidence); technical settings in one collapsed Settings panel; definitions, hardware, threads, paths and caveats in Details panels beside the results they qualify; three primary live figures on Run; a prominent budget slider during real cached runs; status text limited to states the server reports (loading, prefill, generating, cancelling, the stop reason). A launcher (`scripts/dashboard.ps1`) uses the `.venv` that setup creates and says so if it is missing.
**Why:** The first page put every figure and caveat on one screen. Reviewers need the result first and the qualifications one click away, never removed.
**Evidence:** The page was driven in Microsoft Edge with Playwright (from a scratch environment, not a project dependency) against the real model: tabs, settings, synthetic run, streaming, KaTeX equations in an answer, a live budget change (138 to 34 experts), cancel during prefill (stopped 5.8 s later, after prefill), a comparison with matching tokens, and phone width. It found a horizontal overflow at phone width and a missing favicon request; both fixed and rechecked (no overflow on any tab at 390 or 1280 px, no console errors). Screenshots are in docs/screenshots. The model's quadratic answer in that run was wrong ((x - 6)(x - 1)), shown as generated.
**Rejected:** Progress percentages (the engine reports no meaningful fraction of work done). Hiding limitations to simplify the page.
**Follow-up:** Choices renamed to say who supplies what: Model (*Qwen — real model*, *Tiny model — engine test*) and Execution (*LeanMoE — cache + prefetch*, *Baseline — Windows paging*), with a "What am I running?" panel that highlights the selection. The server now echoes the submitted prompt and settings in its start event, so the page's confirmed-run line, Run details and downloadable run record come from the server, not from the form; a test checks that no local path appears in the events. Checked again in Edge with Qwen and the tiny model.

## Repository cleanup
**Decision:** Remove `test_load.py`, CONTRACT.md (its layout and rules now live in the README and results/EVIDENCE.md) and empty placeholder files; consolidate result details into results/EVIDENCE.md; keep every result file, including the earlier pair made before the accounting correction and the noisy exploratory prefetch run.
**Why:** A first-time reader should find one description of each thing, and every claim should trace to a file. Unfavorable or superseded results stay, labeled.
**Rejected:** Removing the base-store configs (`configs/runtime.json`, `configs/runtime_cache.json`): `runtime.json` is the engine's default config and both are cited in ported docstrings.
