import importlib.util
import json
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).parents[1] / "scripts" / "benchmark.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def test_score_soft_gold_and_floor():
    s = bm.score({"target": [0.75, 0.25]}, [0.5, 0.5])
    assert s["ok"] and s["brier"] == pytest.approx(0.125)
    assert s["logloss"] == pytest.approx(math.log(2))
    # a zero on the gold option is floored, not infinite
    assert bm.score({"target": [0.0, 1.0]}, [1.0, 0.0])["logloss"] == pytest.approx(-math.log(1e-6))


def test_selective_ranks_by_confidence():
    oks = [True, False, True, True]
    conf = [0.9, 0.2, 0.8, 0.7]
    out = bm.selective(oks, conf)
    assert out["acc@50%"] == 1.0 and out["acc@80%"] == 1.0  # the wrong answer is least confident
    assert out["aurc"] == pytest.approx((0 + 0 + 0 + 0.25) / 4)
    assert bm.selective(oks, [0.1, 0.9, 0.5, 0.4])["aurc"] > out["aurc"]


def test_run_records_round_trip(tmp_path):
    rows = [{"id": f"x:{i}", "source": "agnews", "type": "choice", "options": ["a", "b"], "target": t}
            for i, t in enumerate([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])]
    res = [{"p": [0.9, 0.1], "p_correct": 0.8, "ms": 1.0}, {"p": [0.6, 0.4], "p_correct": 0.3, "ms": 2.0},
           {"skip": "error"}]
    path = bm.per_row_path(str(tmp_path / "bench_results.json"), "groot")
    assert path.name == "bench_results.groot.jsonl"
    bm.save_per_row(path, rows, res)
    loaded = bm.load_per_row(path, rows)
    assert loaded == res
    summary = bm.summarize(rows, loaded)["agnews"]
    assert summary["n"] == 2 and summary["skipped"] == 1 and summary["acc"] == 0.5
    assert summary["logloss"] == pytest.approx((-math.log(0.9) - math.log(0.4)) / 2)
    assert summary["p_correct"]["acc@50%"] == 1.0
    assert "Groot log loss" in bm.calibration_table({"groot": {"agnews": summary}})
    with pytest.raises(SystemExit):
        bm.load_per_row(path, rows[::-1])
