# Development process and provenance

LeanMoE was assembled for a 48-hour hackathon using Abdullah Ali Alanazi's earlier **ExpertRelay** engine as its foundation. The int8 store format and readers, fused NumPy runtime, expert cache, predictor, LiveBudget and startup guard predate this submission. LeanMoE's Windows integration, dashboard, evaluation and cloud deployment build on that work; the [decision log](../DECISIONS.md) records the boundary in more detail.

The author used Claude Code and Codex in alternating development sessions, continuing with one when the other's usage window ended. Copilot was used for independent, blind review and testing; reported problems were then investigated and addressed in the project. This describes the working process, not independent proof that every suggestion was correct or every problem was fixed.

Review the [source history](https://github.com/No6Name8/LeanMoE/commits/main), [automated tests](../tests/), and [evidence index](../results/EVIDENCE.md) for the changes and measurements themselves. A test total applies to the version and environment in which it was run; recorded laptop and cloud results have different hardware and memory conditions. The author remains responsible for the submitted code, claims and limitations.
