"""Summary arithmetic of eval/task_benchmark.py on fabricated records (no model is run)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.task_benchmark import memory_guard_reason, summarize  # noqa: E402


def req(kind, score, latency, ids=(1, 2), stop="eos", within=True):
    return {"kind": kind, "score": score, "latency_s": latency, "within_time_limit": within, "stop_reason": stop,
            "time_to_first_token_s": 5.0, "decode_tokens_per_s": 2.0, "tokens_generated": len(ids),
            "token_ids": list(ids), "memory": {"peak_private_gb": 1.0, "peak_working_set_gb": 2.0}}


def run(task, mode, first, warm, load=10.0):
    return {"task_id": task, "mode": mode, "error": None, "load_s": load, "child_wall_s": 50.0, "requests": [first, warm]}


class Summary(unittest.TestCase):
    def setUp(self):
        self.data = {"design": {"time_limit_s": 120}, "runs": [
            run("a", "cached", req("first_request", "correct", 30), req("warm_repeat", "correct", 20)),
            run("a", "mmap", req("first_request", "correct", 150, within=False), req("warm_repeat", "correct", 60)),
            run("b", "cached", req("first_request", "incorrect", 40, stop="limit"), req("warm_repeat", "incorrect", 30, stop="limit")),
            run("b", "mmap", req("first_request", "manual_review", 50, ids=(9,)), req("warm_repeat", "manual_review", 40, ids=(9,))),
            {"task_id": "c", "mode": "mmap", "error": "child failed", "child_wall_s": 5.0},
        ]}

    def test_counts_and_rates(self):
        s = summarize(self.data)["by_mode"]
        c, m = s["cached"]["first_request"], s["mmap"]["first_request"]
        self.assertEqual((c["attempted"], c["correct"], c["correct_within_time_limit"]), (2, 1, 1))
        self.assertEqual(c["latency_s_per_correct_task"], 70.0)  # (30 + 40) / 1: failed attempts count
        self.assertEqual(c["latency_plus_load_s_per_correct_task"], 90.0)
        self.assertEqual(c["truncated_at_token_limit"], 1)
        self.assertEqual((m["attempted"], m["correct"], m["correct_within_time_limit"], m["manual_review_open"]), (3, 1, 0, 1))
        self.assertEqual(s["mmap"]["failed_runs"], 1)
        self.assertEqual(s["cached"]["warm_repeat"]["latency_s_per_correct_task"], 50.0)
        self.assertIsNone(s["cached"]["warm_repeat"]["total_latency_plus_load_s"])

    def test_manual_verdicts_and_consistency(self):
        out = summarize(self.data, {"b": {"mmap": {"first_request": "correct"}}})
        self.assertEqual(out["by_mode"]["mmap"]["first_request"]["correct"], 2)
        self.assertEqual(out["by_mode"]["mmap"]["warm_repeat"]["manual_review_open"], 1)
        self.assertEqual(out["consistency"], {"a": {"first_request": True, "warm_repeat": True},
                                              "b": {"first_request": False, "warm_repeat": False}})

    def test_partial_first_request_survives_a_child_failure(self):
        self.data["runs"].append({"task_id": "d", "mode": "cached", "error": "child timeout",
                                  "load_s": 3, "requests": [req("first_request", "correct", 15)]})
        s = summarize(self.data)["by_mode"]["cached"]
        self.assertEqual(s["first_request"]["attempted"], 3)
        self.assertEqual(s["first_request"]["correct"], 2)
        self.assertEqual(s["warm_repeat"]["attempted"], 2)

    def test_truncated_rule_pass_is_not_a_completed_task(self):
        self.data["runs"].append(run("d", "cached", req("first_request", "correct", 30, stop="limit"),
                                     req("warm_repeat", "correct", 20, stop="cancelled")))
        s = summarize(self.data)["by_mode"]["cached"]
        self.assertEqual(s["first_request"]["rule_correct_incomplete"], 1)
        self.assertEqual(s["warm_repeat"]["rule_correct_incomplete"], 1)
        self.assertEqual(s["first_request"]["correct"], 1)

    def test_both_memory_headroom_gates(self):
        design = {"memory_budget_gb": 3.0}
        self.assertIsNone(memory_guard_reason({"commit_available_gb": 8, "physical_available_gb": 4}, design))
        self.assertIn("physical", memory_guard_reason({"commit_available_gb": 8, "physical_available_gb": 0.8}, design))
        self.assertIn("commit", memory_guard_reason({"commit_available_gb": 2, "physical_available_gb": 4}, design))


if __name__ == "__main__":
    unittest.main()
