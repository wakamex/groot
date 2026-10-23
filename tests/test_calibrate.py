import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from groot1.calibration import Record, softmax

spec = importlib.util.spec_from_file_location("calibrate", Path(__file__).parents[1] / "scripts" / "calibrate.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def overconfident(n, qtype, scale, seed):
    """Labels drawn from softmax(z), logits scale * z: the fitted temperature should land near scale."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.normal(0, 1, 3)
        target = np.zeros(3)
        target[rng.choice(3, p=softmax(z))] = 1.0
        out.append(Record(logits=scale * z, target=target, qtype=qtype))
    return out


def test_ensemble_nll_averages_members():
    a = [Record(logits=np.array([2.0, 0.0]), target=np.array([1.0, 0.0]), qtype=0)]
    b = [Record(logits=np.array([0.0, 2.0]), target=np.array([1.0, 0.0]), qtype=0)]
    p = 0.5 * softmax(np.array([1.0, 0.0])) + 0.5 * softmax(np.array([0.0, 4.0]))
    assert cal.ensemble_nll([a, b], [0.5, 0.5], [2.0, 0.5], [0]) == pytest.approx(-np.log(p[0]))


def test_refit_ensemble_writes_linked_model(tmp_path, monkeypatch):
    model = tmp_path / "model"
    for m in ("s", "m"):
        (model / m).mkdir(parents=True)
        (model / m / "model.safetensors").write_bytes(b"weights")
        (model / m / "calibration.json").write_text(json.dumps({"temperature": [1.0, 1.0, 1.0]}))
    (model / "ensemble.json").write_text(json.dumps({"name": "e", "members": ["s", "m"], "weights": [0.5, 0.5]}))
    (model / "README.md").write_text("card")
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(800)))

    # choice rows from 3x overconfident logits, true/false rows too; no score rows
    fake = {"s": overconfident(400, 0, 3.0, 0) + overconfident(400, 2, 3.0, 1),
            "m": overconfident(400, 0, 3.0, 0) + overconfident(400, 2, 3.0, 1)}
    monkeypatch.setattr(cal, "member_records", lambda path, rows, device: (fake[path.name], [1.0, 1.0, 1.0]))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["calibrate.py", "--model", str(model), "--rows", str(rows), "--out", str(out), "--types", "choice"])
    cal.main()

    for m in ("s", "m"):
        temps = json.loads((out / m / "calibration.json").read_text())["temperature"]
        assert 2.0 < temps[0] < 4.5          # choice: refit, flattening the overconfident logits
        assert temps[1] == 1.0               # score: no calibration rows, released value kept
        assert temps[2] == 1.0               # true/false: not in --types, released value kept
        assert (out / m / "model.safetensors").is_symlink()
        assert (out / m / "model.safetensors").resolve() == (model / m / "model.safetensors").resolve()
        assert json.loads((model / m / "calibration.json").read_text())["temperature"] == [1.0, 1.0, 1.0]
    assert (out / "README.md").is_symlink()
    assert json.loads((out / "ensemble.json").read_text())["members"] == ["s", "m"]
    report = json.loads((out / "calibration-report.json").read_text())
    assert report["type0"]["nll_refit"] < report["type0"]["nll_released"]
    assert "type2" not in report and "type1" not in report
