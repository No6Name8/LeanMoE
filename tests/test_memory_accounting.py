"""The engine's process baseline is measured at import, not only a fixed constant.

    python -m unittest discover -s tests -v

Needs the engine's dependencies (numpy); skipped without them.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.engine.runtime import generate
except ImportError:  # numpy/numba/tokenizers not installed
    generate = None


@unittest.skipIf(generate is None, "engine dependencies not installed")
class ProcessBaselineTest(unittest.TestCase):
    def test_baseline_is_the_larger_of_constant_and_measurement(self) -> None:
        self.assertEqual(
            generate.process_baseline_bytes(),
            max(generate.PROCESS_BASELINE_BYTES, generate.IMPORT_PRIVATE_BYTES),
        )

    def test_import_measurement_covers_numpy(self) -> None:
        # numpy is imported before the measurement, so its committed memory is in it
        self.assertIn("numpy", sys.modules)
        self.assertGreater(generate.IMPORT_PRIVATE_BYTES, 0)

    def test_measurement_is_fixed_after_import(self) -> None:
        first = generate.process_baseline_bytes()
        _ = bytearray(64 * 2**20)  # later allocations must not move the baseline
        self.assertEqual(generate.process_baseline_bytes(), first)


if __name__ == "__main__":
    unittest.main()
