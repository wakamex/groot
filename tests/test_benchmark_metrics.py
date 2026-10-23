import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).parents[1] / "scripts" / "benchmark.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def test_score_soft_gold_and_floor():
    s = bm.score({"target": [0.75, 0.25], "type": "choice"}, [0.5, 0.5])
    assert s["ok"] and s["brier"] == pytest.approx(0.125)
    assert s["logloss"] == pytest.approx(math.log(2))
    entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert s["kl"] == pytest.approx(s["logloss"] - entropy)
    assert s["tv"] == pytest.approx(0.25)
    assert "rps" not in s
    # a zero on the gold option is floored, not infinite
    assert bm.score({"target": [0.0, 1.0], "type": "noul"}, [1.0, 0.0])["logloss"] == pytest.approx(-math.log(1e-6))


def test_rps_on_ordered_levels():
    # gold on level 0; all mass on level 2 is worse than all mass on level 1
    far = bm.score({"target": [1.0, 0.0, 0.0], "type": "score"}, [0.0, 0.0, 1.0])["rps"]
    near = bm.score({"target": [1.0, 0.0, 0.0], "type": "score"}, [0.0, 1.0, 0.0])["rps"]
    assert far == pytest.approx(1.0) and near == pytest.approx(0.5)


def test_selective_ranks_by_confidence():
    ok = np.array([True, False, True, True])
    out = bm.selective(ok, np.array([0.9, 0.2, 0.8, 0.7]))
    assert out["acc@50%"] == 1.0 and out["acc@80%"] == 1.0  # the wrong answer is least confident
    assert out["aurc"] == pytest.approx((0 + 0 + 0 + 0.25) / 4)
    assert bm.selective(ok, np.array([0.1, 0.9, 0.5, 0.4]))["aurc"] > out["aurc"]


def test_ece_and_reliability():
    ok = np.array([True] * 8 + [False] * 2)
    conf = np.full(10, 0.85)
    assert bm.ece(ok, conf) == pytest.approx(0.05)
    assert bm.reliability(ok, conf) == [{"bin": [0.8, 0.9], "n": 10, "conf": pytest.approx(0.85), "acc": 0.8}]


def test_resamples_keep_states_together():
    rows = [{"id": i, "state_key": "ab"[i % 2]} for i in range(6)]
    for ix in bm.resamples(rows)[:20]:
        assert len(ix) == 6 and all(sum(1 for i in ix if i % 2 == s) % 3 == 0 for s in (0, 1))


def test_run_records_round_trip(tmp_path):
    rows = [{"id": f"x:{i}", "source": "agnews", "type": "choice", "options": ["a", "b"], "target": t, "state_key": str(i)}
            for i, t in enumerate([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])]
    res = [{"p": [0.9, 0.1], "p_correct": 0.8, "ms": 1.0}, {"p": [0.6, 0.4], "p_correct": 0.3, "ms": 2.0},
           {"skip": "error"}]
    out = str(tmp_path / "bench_results.json")
    path = bm.per_row_path(out, "groot")
    assert path.name == "bench_results.groot.jsonl"
    bm.save_per_row(path, rows, res)
    loaded = bm.load_per_row(path, rows)
    assert loaded == res
    summary = bm.summarize(rows, loaded)["agnews"]
    assert summary["n"] == 2 and summary["skipped"] == 1 and summary["acc"] == 0.5
    assert summary["logloss"] == pytest.approx((-math.log(0.9) - math.log(0.4)) / 2)
    assert summary["p_correct"]["acc@50%"] == 1.0
    lo, hi = summary["ci"]["acc"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    other = [{"p": [0.6, 0.4], "ms": 1.0}, {"p": [0.6, 0.4], "ms": 1.0}, {"p": [0.5, 0.5], "ms": 1.0}]
    bm.save_per_row(bm.per_row_path(out, "jev"), rows, other)
    results = {"groot": {"agnews": summary}, "jev": bm.summarize(rows, other)}
    bm.compare(rows, out, results)
    pair = results["paired"]["jev"]["agnews"]
    assert pair["n"] == 2 and pair["acc"]["diff"] == pytest.approx(0.0)
    assert "Groot log loss" in bm.all_tables(results) and "Groot minus Jev" in bm.all_tables(results)
    with pytest.raises(SystemExit):
        bm.load_per_row(path, rows[::-1])
