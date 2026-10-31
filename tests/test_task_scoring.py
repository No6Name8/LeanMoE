"""Scoring rules of the task benchmark (eval/task_scoring.py), on hand-written replies."""

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.task_scoring import CORRECT, INCORRECT, MANUAL, answer_field, numbers, score  # noqa: E402

TASKS = {t["id"]: t for t in json.loads((REPO / "configs" / "tasks" / "task_benchmark_v1.json").read_text(encoding="utf-8"))["tasks"]}


def s(task_id, text, stop="eos"):
    return score(TASKS[task_id], text, stop)["score"]


class AnswerField(unittest.TestCase):
    def test_last_answer_line_markdown(self):
        self.assertEqual(answer_field("Answer: 3\nwork\n**Answer:** 936"), "936")
        self.assertEqual(answer_field("answer : Lina is oldest\nmore"), "Lina is oldest")
        self.assertIsNone(answer_field("The result is 936."))
        self.assertIsNone(answer_field("Nonanswer: 936"))
        self.assertIsNone(answer_field("The answer: 936"))

    def test_numbers(self):
        self.assertEqual(numbers("17,532"), [17532.0])
        self.assertEqual(numbers("x = 5/2 or x = −4"), [2.5, -4.0])
        self.assertEqual(numbers("1, 2"), [1.0, 2.0])


class Rules(unittest.TestCase):
    def test_last_number(self):
        self.assertEqual(s("arith_add", "Answer: 936"), CORRECT)
        self.assertEqual(s("arith_mul", "Answer: 17,532"), CORRECT)
        self.assertEqual(s("arith_mul", "Answer: 17,052"), INCORRECT)
        self.assertEqual(s("arith_multi", "Answer: 23 × 47 − 389 = 692"), CORRECT)
        self.assertEqual(s("arith_multi", "Answer: 692 (approximately 700)"), MANUAL)
        self.assertEqual(s("eq_linear", "Answer: x = 6"), CORRECT)
        self.assertEqual(s("fact_moon", "Answer: July 1969"), CORRECT)
        self.assertEqual(s("read_marbles", "Answer: none"), INCORRECT)

    def test_root_set(self):
        self.assertEqual(s("eq_quadratic", "Answer: x = 2 and x = 3"), CORRECT)
        self.assertEqual(s("eq_quadratic", "Answer: 3, 2"), CORRECT)
        self.assertEqual(s("eq_quadratic", "Answer: x = 6 and x = 1"), INCORRECT)
        self.assertEqual(s("eq_quadratic", "Answer: (x-2)(x-3)=0, so x = 2, 3"), MANUAL)
        self.assertEqual(s("eq_hard", "Answer: x = 5/2 or x = -4"), CORRECT)
        self.assertEqual(s("eq_hard", "Answer: x = 2.5, x = −4"), CORRECT)
        self.assertEqual(s("eq_hard", "Answer: x = 4 or x = -2.5"), INCORRECT)

    def test_choice(self):
        self.assertEqual(s("fact_capital", "Answer: Canberra"), CORRECT)
        self.assertEqual(s("fact_capital", "Answer: Sydney"), INCORRECT)
        self.assertEqual(s("fact_capital", "Answer: Canberra, not Sydney"), MANUAL)
        self.assertEqual(s("fact_symbol", "Answer: Na"), CORRECT)
        self.assertEqual(s("fact_symbol", "Answer: NA"), INCORRECT)
        self.assertEqual(s("read_train", "Answer: Train B"), CORRECT)
        self.assertEqual(s("read_train", "Answer: A"), INCORRECT)
        self.assertEqual(s("read_oldest", "Answer: lina"), CORRECT)

    def test_missing_answer_line(self):
        self.assertEqual(s("arith_add", "It is 936.", "eos"), MANUAL)
        self.assertEqual(s("arith_add", "It is 936 and", "limit"), INCORRECT)
        self.assertEqual(s("arith_add", "", "cancelled"), INCORRECT)

    def test_ambiguous_answer_is_not_an_unqualified_quality_claim(self):
        # This is a known limitation of the fixed last-number rule. Keep the
        # original rule for comparability, and require manual inspection of
        # automatically correct replies before making a quality claim.
        self.assertEqual(s("arith_add", "Answer: 1 or 936"), CORRECT)


if __name__ == "__main__":
    unittest.main()
