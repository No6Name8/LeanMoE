# LeanMoE

A mixture-of-experts (MoE) model only uses a few of its experts for each token, so it doesn't need all its weights in RAM at once. LeanMoE runs Qwen1.5-MoE-A2.7B-Chat, whose int8 weights take 14.36 GB on disk, on a laptop with 8.30 GB of RAM: shared weights stay in memory, routed experts are read from the SSD when needed, a RAM cache keeps recently used experts, and a predictor starts reading the next layer's likely experts early.

This submission evaluates that design on the real model, including an **adaptive memory budget**: the budget is cut while the model is generating, the expert cache shrinks between tokens, and generation continues with the same output.

## Start here

**Remote access:** [Open the authenticated cloud demo](https://leanmoe-demo-1.eastus.cloudapp.azure.com/).
Credentials are required. The same engine runs on a Windows server; no model
download is needed for browser use. Cloud speedup is not established. The dashboard's
Cloud tab links there and retains local setup instructions. See
[deployment, recovery and teardown](deploy/README.md). Recorded laptop results
remain separate from live cloud measurements.

**Supported:** Windows 10 or 11, 64-bit, with Python 3.13 (tested on Windows 11 with Python 3.13.14; Python 3.10 to 3.12 may work but are untested). Linux and macOS are not supported: the engine reads expert weights with Windows-only unbuffered file I/O, so on those systems the engine tests are skipped (skipped is not passed) and only the saved-evidence checker runs. You also need Git and an internet connection for the package install.

A local evaluation dashboard is included (see 'Local dashboard' below); everything else runs from the command line. The quick-verification commands below were run word for word in a fresh clone of commit e8a27a8 in a new PowerShell session on the development laptop (Windows 11, Python 3.13.14), not yet on another machine. The laptop checkpoint had 171 tests. The protected deployment adds 10 access/manifest checks (181 total); unfinished task-benchmark checks are separate. See the deployment checkpoint for cloud verification.

### Path 1: quick verification (no model download)

Runs the actual engine on a tiny synthetic model that the tests generate, and checks the recorded real-model results. It does not run the real model. Copy into Windows PowerShell:

```
git clone https://github.com/No6Name8/LeanMoE.git
cd LeanMoE
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe eval\compare_artifacts.py results\adaptive_demo_20261030-060626.json results\adaptive_demo_20261030-060728.json
```

What each step does and what success looks like:

- **`setup.ps1`** creates `.venv` inside the repository, installs the packages in `requirements-dev.txt` (numpy, numba, psutil, tokenizers, jinja2, pytest) and runs the preflight check. It ends with `Setup complete.` It is safe to rerun, downloads no model files and changes no system settings (`-ExecutionPolicy Bypass` applies to that one PowerShell process only). Measured on a fresh clone on the development laptop (packages downloaded fresh, no pip cache): 114 s; a rerun takes a few seconds.
- **`pytest`** runs 171 tests and should end with `181 passed`. 63 are ExpertRelay's engine tests: each builds a tiny random-weight model and expert store in a temporary folder and runs the real engine on it, covering cache resizing, the live memory budget shrinking and growing mid-generation, and bit-identical output across expert sources, cache sizes and prefetch settings. 15 test LeanMoE's streaming generation on the same kind of store (same tokens as the batch path, end-of-sequence stop, length checks before allocation, cancellation, cleanup after errors, consecutive requests), and 31 test the local dashboard's server on it. Ten additional tests cover protected HTTP sessions, account/browser isolation, owner-only control, limits, deadlines, deployment versions and transfer manifests. The other 62 test the evidence checker, the memory-accounting baseline, streaming text decoding, system metadata, artifact-derived dashboard summaries and (with Node.js installed) the dashboard page's math rendering; without Node.js those 3 are skipped, which means not verified. These show the engine's mechanisms behave as specified on a synthetic model; they say nothing about the real model's quality or speed. Any `failed` or `skipped` count means that part was not verified. The earlier 157-test suite took 27 s; the expanded suite is verified in the development environment.
- **`compare_artifacts.py`** reads two committed result files from real-model runs and should print `PASS: identical generated token ids (80 tokens) ...`. It checks the **recorded** evidence; it does not rerun the real model.

**Streaming example (optional, no download).** `.\.venv\Scripts\python.exe -m eval.stream_demo --cancel-after-tokens 5` streams token ids from the tiny synthetic model and cancels after five; the output ids are meaningless, it shows the mechanics. With the real store, add `--real --prompt "..."` to stream text (see `--help`).

The preflight check can be rerun at any time: `.\.venv\Scripts\python.exe scripts\preflight.py`. Passing it does not guarantee a real-model run will fit in memory.

### Path 2: the full model (large download, not yet reproduced from this repository)

What you need in addition:

- **The Qwen1.5-MoE-A2.7B-Chat int8 store**: 14.36 GB, plus about 1 GB of free space for margin. It is not in this repository and there is no prebuilt download yet.
- **Memory:** developed on an 8 GB laptop. At the demo defaults the process's sampled private memory peaked at 3.18 GB, so Windows commit headroom should be comfortably above that; more free physical RAM gives a larger cache and less paging.
- **Time:** each demo run takes about two minutes on the development laptop.

**Getting the store (NOT YET VERIFIED in LeanMoE).** The store builder from ExpertRelay is included (`src/engine/store/build_store.py`). It streams the original bf16 checkpoint from Hugging Face (`Qwen/Qwen1.5-MoE-A2.7B-Chat`, revision `ec052fda178e241c7c443468d2fa1db6618996be`) with HTTP range requests, quantizes it to int8 and writes the store; it doesn't keep the 28.6 GB checkpoint on disk. When ExpertRelay built this exact store, the build read 28.6 GB over the network, took 6 h 16 min and peaked at 490 MB of RAM. It resumes if rerun after an interruption. These commands have not been run from LeanMoE:

```
$env:LEANMOE_MODELS_ROOT = "D:\leanmoe-models"
.\.venv\Scripts\python.exe -m src.engine.store.build_store --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat --revision ec052fda178e241c7c443468d2fa1db6618996be --store-dir "$env:LEANMOE_MODELS_ROOT\qwen1.5-moe-a2.7b-chat-int8"
.\.venv\Scripts\python.exe -m src.engine.store.build_store --verify --store-dir "$env:LEANMOE_MODELS_ROOT\qwen1.5-moe-a2.7b-chat-int8"
```

Replace `D:\leanmoe-models` with any folder on a drive with enough space. The model is under the Tongyi Qianwen license (redistribution of derivatives is allowed with conditions, including shipping the license and notices); see the model's LICENSE on Hugging Face. The tokenizer files are fetched from Hugging Face on the first real-model run if they aren't in the store.

**Running the real model** (verified on the development laptop with an existing store):

```
$env:LEANMOE_MODELS_ROOT = "D:\leanmoe-models"
.\.venv\Scripts\python.exe scripts\preflight.py --require-model
.\.venv\Scripts\python.exe -m eval.adaptive_demo --cut-at 0
.\.venv\Scripts\python.exe -m eval.adaptive_demo --cut-at 20
.\.venv\Scripts\python.exe -m eval.adaptive_demo --compare results\<uncut run>.json results\<cut run>.json
```

Each run writes a new result artifact in [results/](results/). **To stop a run**, press Ctrl+C; it stops at the next point Python can interrupt, which during the prompt's first pass can take several seconds, and that run's result file is not written. The experiments behind the results below are `eval\prefetch_ablation.py` and `eval\config_compare.py` (about 15 to 20 minutes each; see their `--help`).

### Local dashboard (optional)

A single page on this computer for running workloads on the engine. It is not a chatbot, and model output is labelled unverified. After running the setup script:

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dashboard.ps1
```

It opens http://127.0.0.1:8770 in the default browser and listens on this computer only (127.0.0.1; there is no option to expose it). Stop it with Ctrl+C in the PowerShell window. Add `-Port 8771` if 8770 is taken, or `-NoOpen` to skip opening the browser. The script uses the `.venv` folder (with a dot) that `setup.ps1` creates and stops with a clear message if it is missing. The equivalent direct command is `.\.venv\Scripts\python.exe -m src.dashboard --open`.

- **Run:** a prompt, the model (**Qwen**: Qwen1.5-MoE-A2.7B-Chat, int8, which generates the answers) and the execution method (**LeanMoE**: the ExpertRelay-derived expert cache and prefetch; or **Windows paging**: the memory-mapped baseline without that cache or prefetch). A collapsed Advanced section offers **Engine test**, a tiny random-weight model whose output is meaningless and is labelled so wherever it appears. A "What am I running?" panel explains each choice in a sentence, with technical details expandable, and highlights the active ones. After the server starts a run, a line confirms what it is running (for example "Live engine run · Qwen · LeanMoE"); the answer streams with three live figures (first-token time, decode speed, private memory). During a Qwen + LeanMoE run a memory-budget slider changes the budget for that run. Run details show the request ID, the submitted prompt, the configuration, token counts, stop reason and measured timings, and the run record can be downloaded as JSON (no local paths).
- **Compare:** the same prompt and settings with LeanMoE, then Windows paging, with each run's progress, side-by-side results and whether the generated tokens match. One run of each is not a benchmark; the Details panel says why.
- **Evidence:** two compact main experiment cards with prominent artifact-derived 80/80 matching tokens and a 2.07x throughput ratio, labelled Recorded and qualified directly below. The 5-of-6 comparison stays alongside the ratio. Each expands to a settings table, limitations, artifact links and PowerShell reproduction commands. Older prefetch and thread experiments are under Additional experiments. Historical hardware and ExpertRelay attribution are separate from live system measurements. Missing or invalid artifacts show unavailable.

**System details** shows five fields: Running on (This computer / Cloud server, from configuration), a concise processor name, total/available RAM, engine memory (process private committed), and idle/loading/running status with model and execution mode when applicable. It shows a relative update age; Technical details is collapsed by default and retains OS, logical cores, Windows commit headroom/limit, working set, effective threads, separate mode fields and memory definitions. The underlying backend measurements are unchanged. Values refresh on opening and every five seconds while open; snapshots 15 seconds old are marked stale. Failed requests or an eight-second timeout show disconnected and label retained values as last-known; recovery clears the warning. Missing values show `unavailable`. Usernames, machine names and absolute paths stay excluded.

The backend location is explicitly configured, never inferred from the browser: `--execution-location local` (default) or `--execution-location server`; the PowerShell launcher accepts `-ExecutionLocation local` or `-ExecutionLocation server`. Server is a label for a backend accessed through an existing forwarded session; the listener remains 127.0.0.1 only. This does not deploy or expose a server.

After a code update, a browser refresh alone does not reload Python. In the original dashboard terminal, press Ctrl+C, then restart with the same environment and model-root setting, then refresh the page. For the older development environment use `.\venv\Scripts\python.exe -m src.dashboard --open`; for a fresh setup use the launcher above. A development check may use port 8771; it must not stop the existing 8770 process.

Without the Qwen store it shows **Qwen store not found** with an explanation that the store may need downloading or its location configuring, setup instructions and an explicit **Use engine test instead** button; it never switches models on its own. For the real model, set `$env:LEANMOE_MODELS_ROOT` (see Path 2) in the same window before starting it. One run at a time. A run's first forward pass (prefill, about 6 to 10 s on the real model) can't be interrupted, so a cancel during it takes effect when it ends, and the page says so. Equations are typeset with KaTeX, bundled in the repository (MIT license), with untrusted commands disabled.

Screenshots from the browser check: [run with a budget change](docs/screenshots/run-budget-change.png), [an equation answer](docs/screenshots/run-equation-answer.png), [comparison](docs/screenshots/compare.png), [evidence](docs/screenshots/evidence-descriptive-desktop.png), [phone width](docs/screenshots/phone.png).

What has been verified: the server by 31 automated tests on the synthetic model (one run at a time, invalid inputs, cancellation, cache reset, mode switching, cleanup when the page closes or the engine fails, localhost-only binding, refusing a busy port); the page's math rendering by Node.js checks; and the page itself in Microsoft Edge driven by Playwright on the development laptop with the real model (all three tabs, settings, synthetic run, streaming, a typeset equation answer, a live budget change, cancel during prefill, comparison, phone width, no console errors; after the relabelling: the explanation panel follows the selection, the confirmed run line and Run details match the actual run, and downloaded records match the details and contain no local paths). The browser script lives outside the repository and isn't part of `pytest`; it is a one-time check, not a regression test.

Updated System details/Evidence check (no real-model inference): **170 passed** in 29.83 s in the existing development environment; focused metadata/evidence/JavaScript checks **14 passed** after final code changes. Microsoft Edge/Playwright at 1280 x 1000 and 390 x 844 verified all tabs, expandable methods, source links, freshness refresh, missing-value rendering (mocked), explicit server/local labels, popover bounds and Escape, with no horizontal overflow or console errors. The separate check server used port 8771; no stop or restart was issued against the original 8770 server. Browser scripts and Playwright dependencies remain outside the repository; these are one-time UI checks, not pytest coverage.

Updated screenshots: [Evidence desktop](docs/screenshots/evidence-descriptive-desktop.png), [Evidence phone](docs/screenshots/evidence-descriptive-phone.png), [expanded desktop](docs/screenshots/evidence-descriptive-desktop.png), [expanded phone](docs/screenshots/evidence-descriptive-phone.png), [System details desktop](docs/screenshots/system-relative-desktop.png), [System details phone](docs/screenshots/system-relative-phone.png). The phone system panel scrolls internally.

Simplified System details: [desktop](docs/screenshots/system-relative-desktop.png), [phone](docs/screenshots/system-relative-phone.png), [technical desktop](docs/screenshots/system-relative-desktop.png), [technical phone](docs/screenshots/system-relative-phone.png), [disconnected phone](docs/screenshots/system-disconnected-phone.png). The follow-up browser check verified the five-field layout, collapsed technical section, refresh without collapsing it, disconnected/recovery behavior, stale snapshots and loading/running/missing-value states at desktop and phone widths. State simulations used intercepted status responses; no inference was started.

Evidence presentation check: **42 affected tests passed**. Edge/Playwright at 1280 px and 390 px verified both headline figures and captions, Recorded badges, settings tables for all four experiments, source links, and real store-found/store-missing configurations on separate check servers. Run and Compare kept their explicit engine-test choice. No inference was started; no horizontal overflow or console errors occurred.

Screenshots: [figures desktop](docs/screenshots/evidence-descriptive-desktop.png), [figures phone](docs/screenshots/evidence-descriptive-phone.png), [settings desktop](docs/screenshots/evidence-descriptive-desktop.png), [settings phone](docs/screenshots/evidence-descriptive-phone.png), [store missing desktop](docs/screenshots/store-missing-desktop.png), [store missing phone](docs/screenshots/store-missing-phone.png).

### Common errors

| Message | Fix |
|---|---|
| `running scripts is disabled on this system` | Run the setup line exactly as shown: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1`. |
| `No Python 3.10-3.13 found` | Install Python 3.13 from python.org (with the `py` launcher or "Add python.exe to PATH"), then rerun setup. |
| pytest shows `skipped` | The engine tests only run on Windows. Skipped tests are not verified. |
| `no model store at ...` or preflight says `NOT AVAILABLE` | Set `$env:LEANMOE_MODELS_ROOT` to the folder that contains `qwen1.5-moe-a2.7b-chat-int8`. |
| `MemoryBudgetExceeded` | The startup memory guard refused the configuration. Lower `--start-gb` and `--start-cache-gb`. Each demo run also shows one deliberate refusal (the oversized fixed-cache example); that one is expected. |
| `MemoryError` or `Unable to allocate` | Windows ran out of commit headroom. Close other programs and check preflight's commit figure. |
| Dashboard: `port 8770 is already in use` | Another program uses that port. Start with `--port 8771`. |
| A path-length error while installing `torch` | Only the optional toy needs torch (`-WithToy`). Clone into a short folder such as `C:\LeanMoE`. |

**Optional toy fallback.** `src/synthetic_moe.py` times disk loads against cache hits on 8 tiny toy layers. It doesn't use the engine and says nothing about real-model performance. It needs torch: run setup with `-WithToy`, then `.\.venv\Scripts\python.exe -m src.synthetic_moe`.

## Results at a glance

Measured on one 8 GB Windows laptop; details and caveats below and in [results/EVIDENCE.md](results/EVIDENCE.md).

- **About 2x the decode throughput of Windows memory-mapping.** Against the same store memory-mapped and paged in by Windows, the cached engine with prefetch averaged 1.842 against 0.891 tokens/s (ratio of means 2.07) and was faster in 5 of 6 matched comparisons; individual pairs ranged from 0.92 to 11.2 because memory-mapped speed depends on what Windows already has cached. Three prompts, two runs each, identical output. [Paging comparison](results/config_compare_mmap_full_20261030-060448.json)
- **Budget cut mid-generation, same output.** The budget dropped from 3.5 to 2.5 GB after token 20; the expert cache shrank from 124 to 59 experts, measured private memory fell to 2.390 GB, and all 80 generated tokens match the uncut reference run. One prompt. [Uncut reference](results/adaptive_demo_20261030-060626.json) (uncut), [Adaptive run](results/adaptive_demo_20261030-060728.json) (cut)
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

## Verified results

All on one laptop (Intel Core i5-12450H, 8.30 GB RAM, NVMe SSD), int8 Chat store, numpy backend, fused int8 kernel, greedy decoding. Details, raw files and the earlier runs are in [results/EVIDENCE.md](results/EVIDENCE.md).

**1. The budget can be cut mid-generation.** Primary pair, made with the current engine (corrected memory accounting, 4 OpenBLAS threads); only the cut schedule differs.

| | Uncut | Cut to 2.5 GB after token 20 |
|---|---|---|
| Evidence file | [Uncut reference](results/adaptive_demo_20261030-060626.json) | [Adaptive run](results/adaptive_demo_20261030-060728.json) |
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
- `scripts/setup.ps1` — creates `.venv`, installs dependencies, runs preflight; `scripts/preflight.py` — environment and model-store check.
- `requirements.txt` (engine), `requirements-dev.txt` (adds pytest), `requirements-toy.txt` (adds torch for the toy only).
- `eval/adaptive_demo.py` — budget-cut demo and the oversized fixed-cache example.
- `eval/compare_artifacts.py` — standard-library checker for an uncut and a cut run.
- `eval/prefetch_ablation.py` — prefetch on/off experiment.
- `eval/config_compare.py` — BLAS-thread and memory-mapping comparisons.
- `src/dashboard/` — local evaluation dashboard: `server.py` (127.0.0.1 HTTP server and the single model worker), `static/` (page, scripts, bundled KaTeX), `synthetic.py`, `evidence.py`; launched with `python -m src.dashboard`. `eval/dashboard_smoke.py` — real-model check through its HTTP interface.
- `src/engine/runtime/streaming.py` and `text_stream.py` — streaming generation (end-of-sequence stop, limits, cancellation, safe incremental text); `eval/stream_demo.py` — command-line example.
- `tests/` — standard-library tests for the evidence checker and memory accounting; `tests/engine/` — engine tests ported from ExpertRelay (pytest, Windows), run with `python -m pytest`.
- `configs/` — engine configs, prompt set and prefetch calibrations. `runtime.json` (the engine's default config) and `runtime_cache.json` point at the base store, which these demos don't use.
- `results/` — raw result files, indexed in [results/EVIDENCE.md](results/EVIDENCE.md).
- [DECISIONS.md](DECISIONS.md) — what was decided, why, and what was rejected.
