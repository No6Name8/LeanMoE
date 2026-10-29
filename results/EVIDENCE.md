# Evidence files

All four runs used the same prompt, 80 tokens, the Qwen1.5-MoE-A2.7B-Chat int8 store, prefetch on and the fused int8 kernel, on one 8 GB Windows laptop on 2026-10-29. The first three started with a 3.5 GB budget and a 1.5 GB cache; the reduced-budget run started with 2.6 GB and 0.6 GB.

| File | Role | Cut schedule | Settings recorded |
|---|---|---|---|
| `adaptive_demo_20261029-205033.json` | **Primary**: uncut reference | none | yes |
| `adaptive_demo_20261029-205224.json` | **Primary**: cut run | 3.5 GB to 2.5 GB after token 20 | yes |
| `adaptive_demo_20261029-200451.json` | **Supplementary**: earlier cut run | 3.5 GB to 2.5 GB after token 20 | no (predates settings recording) |
| `adaptive_demo_20261029-211301.json` | **Supplementary**: reduced-budget cut run, made during the independent review because commit headroom was too low for the defaults | 2.6 GB to 2.2 GB after token 20 | yes (differs from the primary pair in starting budget and cache) |

The two primary runs were made back to back from a fresh virtual environment. The strict comparison passes for them: identical recorded settings apart from the cut schedule, 80 of 80 tokens each with no error, and identical generated token ids for this tested prompt and configuration.

The earlier supplementary run has the same 80 token ids, but the strict comparison rejects it as a primary pair member because its settings weren't recorded. Its figures (for example 165 to 59 experts) are not the fresh pair's figures.

The reduced-budget run shrank its cache from 57 to 24 experts at token 20, generated 80 of 80 tokens with no error, and has the same 80 token ids as the primary pair. The strict comparison rejects pairing it with the primary uncut run because its starting budget and cache differ; it has no uncut partner of its own. Its measured private memory after the cut, 2.354 GB, was above its 2.2 GB budget.

Each file also holds a "fixed" section: a fresh run with an oversized fixed cache (the run's starting cache) under its cut budget, which the memory guard refused at startup on its estimate. That run never generated and never went through a live cut. It is not a speed baseline and not an out-of-memory crash; a smaller fixed cache would pass the same startup check.

To check the primary pair from the saved files (Python standard library only, no model needed; this does not rerun inference):

```
python eval/compare_artifacts.py results/adaptive_demo_20261029-205033.json results/adaptive_demo_20261029-205224.json
```
