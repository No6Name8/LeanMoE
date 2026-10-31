# Task-success evaluation

This is a small, fixed 12-task check of Qwen1.5-MoE-A2.7B-Chat answers on one Windows laptop. It asks whether a completed answer is correct and whether it arrives within 120 seconds **after model load**. It is separate from token-equality and decode-throughput checks. It does not establish general model quality, energy use or arithmetic compute savings.

## Provenance and fixed design

- [Task specification](../configs/tasks/task_benchmark_v1.json): twelve tasks in fixed order, expected answers, answer-extraction/scoring rules, predictions, greedy decoding, 128 output tokens, 512-token context, 120-second success limit and 300-second per-request cancellation target. The [specification lock](../results/task_benchmark_spec_lock.json) records its LF-normalized SHA-256 (`0b7a2713200513a5f4cc62fc582028e1a93ee15c0c461b41448ce45d2f087a35`). The runner refuses a mismatched spec.
- The untracked specification and original scorer existed on disk before the [three-task pilot](../results/task_benchmark_pilot_20261030-134509.json), whose embedded specification hash, design and included tasks match. File timestamps support that order, but cannot independently prove when untracked content was authored. The first Git commit locking them is `ba037e065ef0f851c9594a22ea0a2248b9c32b0a`, **after** the pilot and **before** any new evaluation inference.
- The pilot is a **harness check**, not the twelve-task outcome: three tasks, both modes, each with a first request and warm repeat. It recorded three rule-correct, EOS-completed answers in each mode for each request kind. All twelve pilot texts were separately checked against the fixed answers in the [manual pilot audit](../results/task_benchmark_pilot_manual_review.json), with no disagreements. It says little about the unrun tasks.
- After the pilot, the scorer's `Answer:` detector was anchored to a standalone line to avoid false passes on `Nonanswer:`. The task file, expected values and numeric/choice rules remain unchanged. The runner gained a physical-RAM guard, a wall-time ceiling, partial-run preservation and separate prefill/decode fields. These later changes are not represented as preregistered.

## Full 12-task result

The [raw full run](../results/task_benchmark_full_20261031-043821.json) used the fixed specification and hash above. It completed all 24 task-mode children in 26.0 minutes, one at a time, alternating mode order by task. All 48 requests stopped at EOS; none timed out or hit the token limit. The [separate manual audit](../results/task_benchmark_full_manual_review.json) assessed every generated text against the frozen expected answer. Each task produced identical text and token IDs across both modes and both request kinds. That consistency does **not** establish correctness.

Each correctness denominator below is **12 fixed tasks** for one mode and request kind. The manual audit found eight correct and four incorrect in every group; all eight correct answers arrived within the fixed 120-second request limit. The automatic scorer recorded six correct, three incorrect and three manual-review cases per group. Manual review resolved two of those cases as correct (`Na`, `Lina`) and one as incorrect (marbles: `6` rather than `10`); it found no disagreement with a definite automatic correct/incorrect score. The other wrong answers were multiplication (`17,052` rather than `17,532`), multistep arithmetic (`1101` rather than `692`) and the harder quadratic (`4, −5` rather than `2.5, −4`). No scoring rule was changed after seeing these answers.

| Mode | Request | Manually correct / completed | Correct within 120 s | Median request latency (range) | Median load per child | Max sampled private / working set |
|---|---|---:|---:|---:|---:|---:|
| LeanMoE | First | 8/12 | 8/12 | 14.9 s (9.0–27.6) | 4.8 s | 2.836 / 2.565 GB |
| LeanMoE | Warm repeat | 8/12 | 8/12 | 14.1 s (8.2–24.9) | 4.8 s | 2.836 / 2.565 GB |
| Windows paging | First | 8/12 | 8/12 | 38.0 s (29.1–55.4) | 2.8 s | 0.493 / 5.352 GB |
| Windows paging | Warm repeat | 8/12 | 8/12 | 46.0 s (35.0–70.2) | 2.8 s | 0.493 / 5.352 GB |

The request latency excludes model load; the first request's load and request times can be added for that child's total after startup. Load is reported once per child, so the warm repeat does not reload. The memory columns are the maximum sampled across that mode's 12 children, **not** an equal-memory comparison: Windows paging had a larger working set. A fresh child is not a cold Windows file cache. These results cover twelve short tasks, one model and one laptop; they do not establish general quality or a universal latency advantage.

| Task | LeanMoE first / warm | Paging first / warm | Manual result in all four responses |
|---|---:|---:|---|
| `arith_add` | 11.7 / 10.6 s | 31.8 / 39.9 s | Correct |
| `arith_mul` | 14.7 / 13.5 s | 39.4 / 45.3 s | Incorrect |
| `arith_multi` | 13.0 / 11.7 s | 32.2 / 40.8 s | Incorrect |
| `eq_linear` | 12.7 / 11.4 s | 36.8 / 44.3 s | Correct |
| `eq_quadratic` | 15.1 / 14.7 s | 37.4 / 46.7 s | Correct |
| `eq_hard` | 17.7 / 16.4 s | 45.4 / 56.4 s | Incorrect |
| `fact_capital` | 17.8 / 17.8 s | 44.6 / 48.5 s | Correct |
| `fact_symbol` | 9.0 / 8.2 s | 29.1 / 35.0 s | Correct after manual review |
| `fact_moon` | 12.1 / 12.0 s | 33.3 / 37.2 s | Correct |
| `read_marbles` | 27.6 / 24.9 s | 55.4 / 70.2 s | Incorrect after manual review |
| `read_train` | 16.9 / 15.0 s | 38.6 / 48.6 s | Correct |
| `read_oldest` | 18.9 / 16.3 s | 42.6 / 52.6 s | Correct after manual review |

## Earlier three-task pilot

The following figures were recalculated from the [raw pilot](../results/task_benchmark_pilot_20261030-134509.json). Each denominator is **three fixed pilot tasks** for that mode and request kind. A child loads once, then handles the first request and its warm repeat. Memory is the maximum sampled over each entire child process; the values are not a common memory allowance or a system-wide RAM measurement.

| Mode | Request | Correct / completed | Correct within 120 s | Mean request latency | Mean load per child | Peak private / working set |
|---|---|---:|---:|---:|---:|---:|
| LeanMoE | First | 3/3 | 3/3 | 13.7 s | 4.94 s | 2.84 / 2.58 GB |
| LeanMoE | Warm repeat | 3/3 | 3/3 | 13.2 s | 4.94 s | 2.84 / 2.58 GB |
| Windows paging | First | 3/3 | 3/3 | 36.8 s | 2.91 s | 0.49 / 5.67 GB |
| Windows paging | Warm repeat | 3/3 | 3/3 | 46.7 s | 2.91 s | 0.49 / 5.67 GB |

All pilot responses stopped at EOS. [Manual review](../results/task_benchmark_pilot_manual_review.json) agreed with all twelve automatic scores; none were incorrect, late, truncated or ambiguous. The two modes emitted identical token IDs for each pilot task and request kind. That token agreement is an execution check, **not** the basis for correctness. The full run above is separate from this earlier pilot.

## Run on a laptop with the separately obtained store

Use an idle Windows machine with enough available physical RAM and Windows commit headroom. Keep other inference processes stopped. The parent checks both before each fresh child and refuses a run below 3.0 GB available physical RAM or below the 3.0 GB planning budget plus 0.3 GB commit headroom. These are safety gates, not hard memory caps. The model store is **not** in GitHub.

```powershell
$env:LEANMOE_MODELS_ROOT = 'D:\leanmoe-models' # replace with the parent of qwen1.5-moe-a2.7b-chat-int8
.\.venv\Scripts\python.exe scripts\preflight.py --require-model
.\.venv\Scripts\python.exe -m src.engine.store.build_store --verify --store-dir "$env:LEANMOE_MODELS_ROOT\qwen1.5-moe-a2.7b-chat-int8"
.\.venv\Scripts\python.exe -m eval.task_benchmark --max-wall-s 3600
```

The runner uses one child at a time, alternating mode order by task: LeanMoE then paging for the first task, paging then LeanMoE for the second, and so on. Each child loads the same Chat store and fused NumPy int8 kernel, with the engine default four OpenBLAS threads; it answers once and repeats the same prompt in the still-loaded process. It saves a result after every mode. If the 60-minute ceiling intervenes, it records the remaining fixed-order schedule instead of choosing easier tasks. The 300-second cancellation takes effect at a token boundary; an unresponsive child can be terminated by the parent. Fresh processes do **not** clear the Windows file cache.

`load_s` times `load_model` only. `prefill_forward_s`/`time_to_first_token_s` time the first model forward pass, including prompt processing; `decode_forward_s` sums recorded later token forward passes, whereas `decode_wall_s` and `latency_s` include Python, tokenization/streaming and end-of-sequence overhead after the stream starts. `latency_s` excludes model load; first-request `latency_s + load_s` is reported separately. `child_wall_s` includes process startup, imports, tokenizer, loading and both requests. Decode tokens/s is tokens after the first divided by their forward-pass time, not a total-task rate. Private memory is committed process memory; working set is resident process memory. Both are sampled peaks, which may miss brief spikes. A timeout is a censored failure, **not** a completed task or a precise speed measurement.

The [scorer](../eval/task_scoring.py) parses fixed answer types and never executes generated code. Its numeric rule can still falsely pass ambiguous replies such as `Answer: 1 or 936` because it selects the last number. Inspect every auto-correct reply and keep rule scores, manual-review cases, wrong answers, truncations and failures in the raw artifact. A rule-correct reply stopped by cancellation or token limit is not counted as a completed correct task in the summary. Token identity across modes is reported separately from task correctness.

## Fresh-checkout route and what it proves

The [setup script](../scripts/setup.ps1) creates `.venv` and installs dependencies. [Preflight](../scripts/preflight.py) verifies required files, source revision and binary sizes; it does **not** hash the store. The builder's `--verify` command rechecks its own expert and resident hashes. A fresh checkout plus a **separately obtained** verified store tests the documented execution route, not reconstruction from upstream or independent transfer of model weights.

In a separate GitHub checkout of commit `ba037e065ef0f851c9594a22ea0a2248b9c32b0a`, the exact [setup](../scripts/setup.ps1) command completed, 205 tests passed, `preflight.py --require-model` discovered the existing pinned Chat store, and `build_store --verify` found **0 mismatches among 1,440 experts and 339 resident tensors**. The saved-artifact comparison command passed as well. The checkout had about 0.8–1.1 GB available physical RAM and 1.9–2.4 GB Windows commit headroom, below the safety gate; **no fresh-checkout real-model inference was run**. Thus setup, discovery and integrity are verified; execution from the clean checkout remains open until safe headroom is available. These numbers describe the verification session, not a live status or a new throughput result.

The builder is pinned to `Qwen/Qwen1.5-MoE-A2.7B-Chat` revision `ec052fda178e241c7c443468d2fa1db6618996be` in the [full-model instructions](../README.md#use-the-full-qwen-model). Building needs about 14.36 GB for the output store plus filesystem margin and roughly 28.6 GB of upstream transfer in the earlier build; time and network demand vary. Preserve the [model notices](../deploy/model-notices/) and verify the resulting store before inference. There is no public prebuilt-store download. An owner-authorized private transfer can use the [SHA-256 manifest tool](../deploy/store_manifest.py); deliver its manifest through an independent authenticated channel. The [public real-engine demo](https://leanmoe-demo-1.eastus.cloudapp.azure.com/) is an alternative for reviewers without the store; it requires no visitor login.

An upstream-to-store rebuild from a fresh LeanMoE checkout remains **not verified end to end**. No model weights, credentials or private transfer links are published here.
