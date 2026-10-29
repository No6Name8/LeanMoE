# Evidence files

All three runs used the same prompt, 80 tokens, the Qwen1.5-MoE-A2.7B-Chat int8 store, a 3.5 GB starting budget, a 1.5 GB starting cache, prefetch on and the fused int8 kernel, on one 8 GB Windows laptop on 2026-10-29.

| File | Role | Cut schedule | Settings recorded |
|---|---|---|---|
| `adaptive_demo_20261029-205033.json` | **Primary**: uncut reference | none | yes |
| `adaptive_demo_20261029-205224.json` | **Primary**: cut run | 3.5 GB to 2.5 GB after token 20 | yes |
| `adaptive_demo_20261029-200451.json` | **Supplementary**: earlier cut run | 3.5 GB to 2.5 GB after token 20 | no (predates settings recording) |

The two primary runs were made back to back from a fresh virtual environment. The strict comparison passes for them: identical recorded settings apart from the cut schedule, 80 of 80 tokens each with no error, and identical generated token ids for this tested prompt and configuration.

The supplementary run has the same 80 token ids, but the strict comparison rejects it as a primary pair member because its settings weren't recorded. Its figures (for example 165 to 59 experts) are not the fresh pair's figures.

Each file also holds a "fixed" section: a fresh run with a fixed 1.5 GB cache under the 2.5 GB budget, which the memory guard refused at startup. That run never generated and never went through a live cut.

To check the primary pair from the saved files (Python standard library only, no model needed; this does not rerun inference):

```
python eval/compare_artifacts.py results/adaptive_demo_20261029-205033.json results/adaptive_demo_20261029-205224.json
```
