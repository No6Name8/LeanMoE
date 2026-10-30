"""Runs the dashboard's JavaScript render checks (tests/js/) with Node.js, if installed.

    python -m unittest discover -s tests -v

LeanMoE test. Skipped when Node.js isn't installed; a skip means these checks
did not run, not that they passed.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class DashboardJavaScriptTest(unittest.TestCase):
    @unittest.skipIf(shutil.which("node") is None, "Node.js not installed")
    def test_math_split_and_katex_settings(self) -> None:
        p = subprocess.run(["node", str(REPO / "tests" / "js" / "dashboard_render.test.js")],
                           capture_output=True, text=True, timeout=60)  # fmt: skip
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("dashboard render checks passed", p.stdout)

    @unittest.skipIf(shutil.which("node") is None, "Node.js not installed")
    def test_page_scripts_parse(self) -> None:
        for name in ("app.js", "mathsplit.js", "systemview.js"):
            p = subprocess.run(["node", "--check", str(REPO / "src" / "dashboard" / "static" / name)],
                               capture_output=True, text=True, timeout=60)  # fmt: skip
            self.assertEqual(p.returncode, 0, name + ": " + p.stderr)

    @unittest.skipIf(shutil.which("node") is None, "Node.js not installed")
    def test_system_details_presentation(self) -> None:
        p = subprocess.run(["node", str(REPO / "tests" / "js" / "system_details.test.js")],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("system details checks passed", p.stdout)


if __name__ == "__main__":
    unittest.main()
