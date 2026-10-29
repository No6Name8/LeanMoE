# Adapted from ExpertRelay (private research project, 2026)
"""Centralized filesystem locations.

Nothing in this repo should hardcode a relative path that only resolves
correctly from one particular working directory (e.g. "the script's own
directory") -- every default path used by an entry point comes from here,
anchored to `src.engine.REPO_ROOT`.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.engine import REPO_ROOT

# Model stores are multi-GB and are never shipped in the repo. LEANMOE_MODELS_ROOT
# points at the local directory that holds them (e.g. ExpertRelay's models/ folder).
MODELS_ROOT: Path = Path(os.environ.get("LEANMOE_MODELS_ROOT", REPO_ROOT / "models"))
DEFAULT_EXPERT_STORE_DIR: Path = MODELS_ROOT / "qwen1.5-moe-a2.7b-int8"
BENCHMARK_RESULTS_DIR: Path = REPO_ROOT / "results"
# Fitted prefetch calibrations ship with the repo, next to the configs that use them.
CALIBRATION_DIR: Path = REPO_ROOT / "configs" / "calibration"
