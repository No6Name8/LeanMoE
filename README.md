# LeanMoE

Single-machine MoE expert prefetching: predict which expert is needed next and load it from disk into RAM before it's needed. See [CONTRACT.md](CONTRACT.md) for the build contract and [DECISIONS.md](DECISIONS.md) for the decisions log.

## Model stores

Model stores are not shipped in this repo. Set the `LEANMOE_MODELS_ROOT` environment variable to a local directory that holds them; configs in `configs/` name a store by folder name inside that directory (or pass `--store-dir` with a full path). If the variable is unset, the engine looks in `models/` at the repo root, which is gitignored.

```
set LEANMOE_MODELS_ROOT=C:\Users\h\Documents\ExpertRelay\models
python -m src.engine.runtime.generate --config configs/runtime_cache_chat.json --source cached --prompt "The capital of France is"
```
