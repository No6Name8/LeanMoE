"""Saved-artifact summaries: arithmetic, pairing and honest failure states."""
import json
from pathlib import Path
import shutil

import pytest

from src.dashboard.evidence import recorded_evidence, MMAP_FULL, PRIMARY_CUT, PRIMARY_UNCUT

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_committed_results_and_details():
    cards = recorded_evidence(RESULTS)
    assert len(cards) == 4 and [c["additional"] for c in cards] == [False, False, True, True]
    assert "80 tokens match" in cards[0]["value"] and "2.390 GB" in cards[0]["value"]
    assert "2.07x" in cards[1]["value"] and "5/6 pairs" in cards[1]["value"]
    assert "1.07x" in cards[2]["value"] and "6/6 pairs" in cards[2]["value"]
    assert "7.8 vs 23.9 s" in cards[3]["value"]
    assert all(c["methodology"] and c["limitations"] and c["commands"] and c["files"] for c in cards)


def test_missing_and_invalid_files_are_explicit(tmp_path):
    assert all("unavailable" in c["value"] for c in recorded_evidence(tmp_path))
    (tmp_path / MMAP_FULL).write_text("{broken")
    assert "unavailable" in recorded_evidence(tmp_path)[1]["value"]


@pytest.mark.parametrize("mutation", ["rates", "tokens", "missing_pair", "error"])
def test_mmap_recomputed_and_failures_not_filtered(tmp_path, mutation):
    data = json.loads((RESULTS / MMAP_FULL).read_text())
    if mutation == "rates":
        for r in data["runs"]:
            r["decode_tokens_per_s"] = 4 if r["mode"] == "cached" else 2
    elif mutation == "tokens":
        data["runs"][0]["generated_ids"][0] += 1
    elif mutation == "missing_pair":
        data["runs"].pop()
    else:
        data["runs"][0]["error"] = "failed"
    (tmp_path / MMAP_FULL).write_text(json.dumps(data))
    result = recorded_evidence(tmp_path)[1]["value"]
    if mutation == "rates":
        assert "2.00x" in result and "6/6 pairs" in result
    elif mutation == "tokens":
        assert "every pair: no" in result
    else:
        assert "unavailable" in result


def test_budget_uses_strict_checker(tmp_path):
    for name in (PRIMARY_CUT, PRIMARY_UNCUT):
        shutil.copy(RESULTS / name, tmp_path / name)
    data = json.loads((tmp_path / PRIMARY_CUT).read_text())
    data["adaptive"]["generated_ids"].pop()
    (tmp_path / PRIMARY_CUT).write_text(json.dumps(data))
    assert "unavailable" in recorded_evidence(tmp_path)[0]["value"]


def test_historical_blas_reproduction_can_select_twelve_threads():
    from argparse import Namespace
    from eval.config_compare import configs_for

    assert configs_for(Namespace(stage="blas", baseline_blas=12)) == [("cached", 12), ("cached", 4)]
    assert configs_for(Namespace(stage="blas", baseline_blas=None)) == [("cached", None), ("cached", 4)]
