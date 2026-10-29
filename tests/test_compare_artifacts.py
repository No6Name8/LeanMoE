"""Regression tests for eval/compare_artifacts.py (the cut-versus-uncut evidence check).

    python -m unittest discover -s tests -v

Standard library only. Every bad case is built from a temporary copy of the
committed primary pair; the committed files themselves are never written.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.compare_artifacts import check_pair  # noqa: E402

UNCUT = REPO / "results" / "adaptive_demo_20261029-205033.json"
CUT = REPO / "results" / "adaptive_demo_20261029-205224.json"
SUPPLEMENTARY = REPO / "results" / "adaptive_demo_20261029-200451.json"


class CompareArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.uncut_doc = json.loads(UNCUT.read_text(encoding="utf-8"))
        cls.cut_doc = json.loads(CUT.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ---------------------------------------------------------------- helpers

    def write(self, name: str, doc: object) -> Path:
        p = self.tmp / name
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    def uncut(self) -> dict:
        return copy.deepcopy(self.uncut_doc)

    def cut(self) -> dict:
        return copy.deepcopy(self.cut_doc)

    def assert_fails(self, a: Path, b: Path, *needles: str) -> None:
        passed, lines = check_pair(a, b)
        text = "\n".join(lines)
        self.assertFalse(passed, text)
        self.assertIn("FAIL", text)
        for needle in needles:
            self.assertIn(needle, text)

    def pair_with(self, mutate_uncut=None, mutate_cut=None) -> tuple[Path, Path]:
        u, c = self.uncut(), self.cut()
        if mutate_uncut:
            mutate_uncut(u["adaptive"])
        if mutate_cut:
            mutate_cut(c["adaptive"])
        return self.write("uncut.json", u), self.write("cut.json", c)

    # ------------------------------------------------------------ valid input

    def test_committed_pair_passes(self) -> None:
        passed, lines = check_pair(UNCUT, CUT)
        self.assertTrue(passed, "\n".join(lines))

    def test_committed_pair_passes_in_either_order(self) -> None:
        passed, lines = check_pair(CUT, UNCUT)
        self.assertTrue(passed, "\n".join(lines))

    def test_temporary_copies_pass(self) -> None:
        a, b = self.pair_with()
        passed, lines = check_pair(a, b)
        self.assertTrue(passed, "\n".join(lines))

    # -------------------------------------------------------- token ids, length

    def test_empty_ids_fail(self) -> None:
        def empty(r):
            r["generated_ids"] = []
        a, b = self.pair_with(empty, empty)
        self.assert_fails(a, b, "incomplete run")

    def test_truncated_ids_with_full_claimed_count_fail(self) -> None:
        def truncate(r):
            r["generated_ids"] = r["generated_ids"][:40]
        a, b = self.pair_with(truncate, truncate)
        self.assert_fails(a, b, "incomplete run")

    def test_short_per_token_log_fails(self) -> None:
        a, b = self.pair_with(mutate_cut=lambda r: r.update(per_token=r["per_token"][:79]))
        self.assert_fails(a, b, "per-token log")

    def test_reported_count_mismatch_fails(self) -> None:
        a, b = self.pair_with(mutate_cut=lambda r: r.update(tokens_generated=79))
        self.assert_fails(a, b, "reported generated")

    def test_zero_requested_tokens_fail(self) -> None:
        def zero(r):
            r.update(tokens_requested=0, tokens_generated=0, generated_ids=[], per_token=[], steps=[])
        a, b = self.pair_with(zero, zero)
        self.assert_fails(a, b, "positive integer")

    def test_invalid_token_id_fails(self) -> None:
        def bad(r):
            r["generated_ids"][5] = "x"
        a, b = self.pair_with(bad, bad)
        self.assert_fails(a, b, "invalid token ids")

    def test_different_token_ids_fail(self) -> None:
        def change(r):
            r["generated_ids"][30] += 1
        a, b = self.pair_with(mutate_cut=change)
        self.assert_fails(a, b, "token ids differ, first at token 31")

    def test_recorded_error_fails(self) -> None:
        a, b = self.pair_with(mutate_cut=lambda r: r.update(error="RuntimeError('boom')"))
        self.assert_fails(a, b, "recorded an error")

    # ---------------------------------------------------------------- settings

    def test_empty_settings_fail(self) -> None:
        def empty(r):
            r["settings"] = {}
        a, b = self.pair_with(empty, empty)
        self.assert_fails(a, b, "settings missing")

    def test_missing_settings_key_fails(self) -> None:
        def drop(r):
            del r["settings"]["prompt_ids"]
        a, b = self.pair_with(drop, drop)
        self.assert_fails(a, b, "settings missing", "prompt_ids")

    def test_missing_settings_object_fails(self) -> None:
        a, b = self.pair_with(mutate_cut=lambda r: r.pop("settings"))
        self.assert_fails(a, b, "missing 'settings'")

    def test_invalid_setting_value_fails(self) -> None:
        def bad(r):
            r["settings"]["start_cache_gb"] = -1
        a, b = self.pair_with(bad, bad)
        self.assert_fails(a, b, "invalid values", "start_cache_gb")

    def test_mismatched_settings_fail(self) -> None:
        a, b = self.pair_with(mutate_cut=lambda r: r["settings"].update(prefetch_k=4))
        self.assert_fails(a, b, "settings differ", "prefetch_k")

    def test_supplementary_run_without_settings_fails(self) -> None:
        self.assert_fails(SUPPLEMENTARY, UNCUT, "missing 'settings'")

    # --------------------------------------------------------------- the cut

    def test_two_uncut_runs_fail(self) -> None:
        other = self.uncut()
        other["adaptive"]["load_seconds"] += 1  # distinct bytes, still no cut
        self.assert_fails(UNCUT, self.write("uncut2.json", other), "exactly one uncut reference and one cut run")

    def test_two_cut_runs_fail(self) -> None:
        other = self.cut()
        other["adaptive"]["load_seconds"] += 1
        self.assert_fails(CUT, self.write("cut2.json", other), "exactly one uncut reference and one cut run")

    def test_missing_shrink_event_fails(self) -> None:
        def drop_event(r):
            r["budget_events"] = [e for e in r["budget_events"] if e["token"] != 20]
        a, b = self.pair_with(mutate_cut=drop_event)
        self.assert_fails(a, b, "no budget event at token 20")

    def test_cut_event_without_shrink_fails(self) -> None:
        def no_shrink(r):
            for e in r["budget_events"]:
                if e["token"] == 20:
                    e["slots_after"] = e["slots_before"]
        a, b = self.pair_with(mutate_cut=no_shrink)
        self.assert_fails(a, b, "cache did not shrink")

    def test_per_token_log_without_shrink_fails(self) -> None:
        def flat(r):
            for row in r["per_token"]:
                row["cache_slots"] = 112
        a, b = self.pair_with(mutate_cut=flat)
        self.assert_fails(a, b, "per-token log does not show the cache shrinking")

    def test_cut_outside_sequence_fails(self) -> None:
        a, b = self.pair_with(mutate_cut=lambda r: r["cut_schedule"].update(cut_at=80))
        self.assert_fails(a, b, "not inside the generated sequence")

    def test_cut_that_raises_budget_fails(self) -> None:
        a, b = self.pair_with(mutate_cut=lambda r: r["cut_schedule"].update(cut_gb=4.0))
        self.assert_fails(a, b, "not below the starting budget")

    def test_uncut_run_that_changed_budget_fails(self) -> None:
        def sneak_cut(r):
            r["budget_events"].append({"token": 30, "budget_gb": 2.5, "slots_before": 120, "slots_after": 59})
        a, b = self.pair_with(mutate_uncut=sneak_cut)
        self.assert_fails(a, b, "uncut reference changed its budget")

    def test_free_ram_adjustment_in_uncut_run_is_allowed(self) -> None:
        # The committed uncut run already has one: 172 -> 120 experts at token 1, at the starting budget.
        events = self.uncut_doc["adaptive"]["budget_events"]
        self.assertTrue(any(e["budget_gb"] == 3.5 and e["slots_after"] < e["slots_before"] for e in events))
        passed, lines = check_pair(UNCUT, CUT)
        self.assertTrue(passed, "\n".join(lines))

    # -------------------------------------------------- self-comparison, malformed

    def test_self_comparison_fails(self) -> None:
        self.assert_fails(UNCUT, UNCUT, "same file was given twice")

    def test_identical_copy_fails(self) -> None:
        copy_path = self.tmp / "copy.json"
        copy_path.write_bytes(CUT.read_bytes())
        self.assert_fails(CUT, copy_path, "identical contents")

    def test_invalid_json_fails(self) -> None:
        bad = self.tmp / "bad.json"
        bad.write_text('{"adaptive": {', encoding="utf-8")
        self.assert_fails(UNCUT, bad, "not valid JSON")

    def test_missing_file_fails(self) -> None:
        self.assert_fails(UNCUT, self.tmp / "nope.json", "cannot read")

    def test_missing_key_fails(self) -> None:
        a, b = self.pair_with(mutate_cut=lambda r: r.pop("generated_ids"))
        self.assert_fails(a, b, "missing 'generated_ids'")

    def test_wrong_type_fails(self) -> None:
        a, b = self.pair_with(mutate_cut=lambda r: r.update(per_token="oops"))
        self.assert_fails(a, b, "'per_token' is not a list")

    def test_not_an_adaptive_artifact_fails(self) -> None:
        self.assert_fails(UNCUT, self.write("other.json", {"hello": 1}), "missing 'adaptive'")


class EntryPointTest(unittest.TestCase):
    """Both command-line entry points give the same verdicts and exit codes."""

    def run_cli(self, *args: str, isolated: bool = False) -> subprocess.CompletedProcess:
        flags = ["-I", "-S"] if isolated else []
        return subprocess.run([sys.executable, *flags, *args], cwd=REPO, capture_output=True, text=True)

    def test_standalone_checker_is_stdlib_only_and_passes(self) -> None:
        p = self.run_cli("eval/compare_artifacts.py", str(UNCUT), str(CUT), isolated=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("PASS", p.stdout)

    def test_standalone_checker_fails_cleanly(self) -> None:
        p = self.run_cli("eval/compare_artifacts.py", str(UNCUT), str(UNCUT), isolated=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("FAIL", p.stdout)
        self.assertNotIn("Traceback", p.stderr)

    def test_standalone_checker_malformed_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.json"
            bad.write_text("not json", encoding="utf-8")
            p = self.run_cli("eval/compare_artifacts.py", str(UNCUT), str(bad), isolated=True)
        self.assertEqual(p.returncode, 1)
        self.assertNotIn("Traceback", p.stderr)

    def test_demo_compare_entry_point_matches(self) -> None:
        try:
            import numpy  # noqa: F401  (the demo imports the engine)
        except ImportError:
            self.skipTest("engine dependencies not installed")
        ok = self.run_cli("-m", "eval.adaptive_demo", "--compare", str(UNCUT), str(CUT))
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertIn("PASS", ok.stdout)
        bad = self.run_cli("-m", "eval.adaptive_demo", "--compare", str(UNCUT), str(UNCUT))
        self.assertEqual(bad.returncode, 1)
        self.assertIn("FAIL", bad.stdout)
        self.assertNotIn("Traceback", bad.stderr)


if __name__ == "__main__":
    unittest.main()
