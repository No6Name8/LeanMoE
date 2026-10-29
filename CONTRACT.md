# LeanMoE — Repo Contract

Single-machine MoE expert caching and prefetching, with an adaptive-RAM demo: cut the memory budget mid-generation and keep generating. The engine is prior work ported from ExpertRelay; see the README for exactly what is ported and what is new.

## Layout

- `src/engine/` — engine ported from ExpertRelay. Every file keeps its `# Adapted from ExpertRelay` header. Changes to ported behavior get a DECISIONS.md entry.
- `src/synthetic_moe.py` — toy fallback. Always described as toy-only; never cited as real-model evidence.
- `eval/adaptive_demo.py` — the adaptive-RAM demo and fixed-cache comparison.
- `eval/compare_artifacts.py` — standard-library-only comparison of two saved runs.
- `configs/` — engine runtime configs, prompt set and prefetch calibrations.
- `results/` — evidence files, indexed in `results/EVIDENCE.md`.
- `test_load.py` — the abandoned OLMoE load test, kept for the record (see DECISIONS.md).

## Rules

- **Evidence first.** Every number in the README or a demo comes from a committed file in `results/` or is labelled as historical. Fresh and historical measurements stay in separate sections.
- **Say what was tested, no more.** Token-id matches are stated for the tested prompt and configuration only. The fixed-cache result is a startup refusal under a smaller configured budget, not an out-of-memory crash.
- **Model stores stay outside the repo.** They are found through `LEANMOE_MODELS_ROOT`. No weights, checkpoints or generated expert files are committed.
- **Decisions get logged** in DECISIONS.md: what was chosen, why, and what was rejected.
