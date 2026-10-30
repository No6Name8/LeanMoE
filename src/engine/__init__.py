# Adapted from ExpertRelay (private research project, 2026)
"""ExpertRelay: virtual memory for AI.

A Manager that understands every device in a cluster, keeps hot experts in
RAM and cold experts on SSD or a networked peer, predicts which expert a
model will need next, and starts moving it before the router asks for it.

See docs/limitations.md for exactly which pieces below are implemented vs.
placeholders, and docs/setup-notes.md for the reduced-model validation work
this package grew out of.
"""

import os
from pathlib import Path

# LeanMoE change: cap OpenBLAS at 4 threads unless the caller already chose a
# value. It only takes effect if numpy hasn't been imported yet, which is the
# case for every entry point in this repository (they import the engine first);
# the effective count is recorded in result files. Measured on the Chat store
# (results/config_compare_blas_*): 4 threads committed 0.27 GB less at import,
# leaving room for a larger cache under the same budget, and cut time to first
# token from 19-35 s to 7-9 s in 5 of 6 matched pairs; decode speed changed in
# both directions. See DECISIONS.md, "BLAS threads".
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

__version__ = "0.1.0"

# src/engine/__init__.py -> engine/ -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
