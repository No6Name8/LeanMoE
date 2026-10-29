# Decisions Log

Format: **Decision** — what we chose. **Why** — reasoning. **Rejected** — what we didn't pick and why.

---

## Project scope
**Decision:** Single-machine (8GB laptop) MoE expert prefetching, not the full multi-device ExpertRelay vision.
**Why:** Only one 8GB machine available for the hackathon window; multi-device sync adds a networking failure mode we can't afford to debug in 48 hours.
**Rejected:** Multi-device Soft Bus version — too much infrastructure risk for a 2-day deadline.

## Challenge picked
**Decision:** Challenge 2 — Deliver AI with Less Compute.
**Why:** No external dataset is provided for any challenge, so all three are scored on self-defined rigor. Challenge 2 gives a built-in, objective metric (time/RAM/cost) that's easy to measure and hard to fake, which maps directly onto "Task Performance" and "Evaluation & Iteration."

---

(Add entries below as real build decisions get made — model choice, predictor design, cache policy, etc.)

## Model and store (2026-10-29)
**Decision:** Use the existing Qwen1.5-MoE-A2.7B-Chat int8 store built by ExpertRelay (about 14 GB on disk, kept outside the repo), not OLMoE-1B-7B.
**Why:** The OLMoE load test never got past downloading and was stopped; it would need about 14 GB in fp16 on a 7.7 GB machine. The Qwen int8 store already existed, already runs under a RAM budget with an expert cache, and generated correctly on this laptop tonight.
**Rejected:** OLMoE through transformers (no evidence it fits, and no expert-level cache control). Building a new store (ExpertRelay's store builder is not ported, and it needs the 27 GB bf16 checkpoint).

## Synthetic fallback kept, but labeled
**Decision:** Keep `src/synthetic_moe.py` as a toy that runs without any model store.
**Why:** Judges won't have the store; it is the only thing they can run as-is.
**Rejected:** Presenting it as evidence of real-model performance. It shows disk-load versus RAM-hit cost on 256x256 toy layers only.

## Port approach
**Decision:** Copy the ExpertRelay engine into `src/engine/` with a header on every file, rewrite `expertrelay.` imports to `src.engine.`, locate stores through `LEANMOE_MODELS_ROOT`, make MindSpore optional in the machine profile, and bring along the seven extra files the requested ones import.
**Why:** The engine (cache, predictor, manager, LiveBudget, memory guard) is prior work; porting it unchanged keeps its behavior and lets us check the port token-for-token against the original. It matched: same 32 token ids on the same store, prompt and settings.
**Rejected:** Rewriting the engine for the hackathon (no time, and nothing to verify it against). Depending on ExpertRelay as an installed package (the judge can't install a private repo).

## Adaptive demo design
**Decision:** Drive the existing LiveBudget through a budget file, starting at 3.5 GB and cutting to 2.5 GB after token 20, logging every token. The demo writes the file itself at a token boundary.
**Why:** The budget-file number is the RAM budget for the whole engine process, not the cache. About 2 GB goes to the rest of the process, so 3.5 GB to 2.5 GB moves the cache from about 1 to 1.5 GB down to 0.51 GB. The originally planned 2.0 GB to 1.0 GB would have left the cache at its 13-expert minimum both times and shown nothing. LiveBudget only reads the file between tokens, so a write from another process during a token takes effect at the same point.
**Rejected:** Running the engine as a subprocess and writing the file from outside (no way to see token progress from outside the process).

## Fixed-cache comparison
**Decision:** The naive comparison is a fresh run with a fixed 1.5 GB cache under the cut 2.5 GB budget. The memory guard refuses it at startup.
**Why:** A fixed cache cannot shrink, so under a smaller budget its only option is to refuse.
**Rejected:** Calling the refusal an out-of-memory crash. The guard compares an estimate (3.48 GB) with the configured budget (2.50 GB); about 2.7 GB was actually free when it ran.

## Correctness check for the adaptive demo
**Decision:** Strict paired check (`eval/adaptive_demo.py --compare`): two fresh runs that record all settings (store, quantization, prompt token ids, decoding, token limit, starting budget and cache, kernel, prefetch settings, calibration). The check passes only if the settings are identical apart from the cut schedule, both runs finish every token without error, and the full token-id lists match.
**Why:** "Outputs don't depend on cache size" was a claim in ExpertRelay's docstring, not something we had measured.
**Result:** PASS. Identical generated token ids, all 80, for this tested prompt and configuration (results/adaptive_demo_20261029-205033.json uncut, results/adaptive_demo_20261029-205224.json cut). The pre-cut cache differed (120 vs 112 experts) because LiveBudget also caps it by free RAM at the first token. An earlier cut run (results/adaptive_demo_20261029-200451.json) has the same token ids, but it predates settings recording, so it counts only as supplementary evidence.
**Rejected:** Generalizing this into a correctness guarantee. It is one prompt and one configuration.

## Dependency verification
**Decision:** requirements.txt lists the third-party imports actually found in the code. It was verified in a fresh virtual environment: clean install, `pip check` clean, all 35 project modules import, and both paired demo runs ran in that environment.
**Why:** A judge will run `pip install -r requirements.txt` first.
**Note:** The first fresh install failed only because its directory path was too long for Windows' path limit (torch has deeply nested files). The README tells Windows users to use a short path.
