import importlib.util
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

spec = importlib.util.spec_from_file_location("confidence_head", Path(__file__).parents[1] / "scripts" / "confidence_head.py")
ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch)


def test_fit_selects_and_writes_new_head(tmp_path):
    rng = np.random.default_rng(0)
    n, dim, hidden = 400, 6, 8
    X = rng.normal(0, 1, (n, dim)).astype(np.float32)
    ok = (rng.random(n) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(int)  # correctness depends on the first input
    model, feats = tmp_path / "model", tmp_path / "feats"
    feats.mkdir()
    for m in ("s", "m"):
        (model / m).mkdir(parents=True)
        save_file({"conf_head.0.weight": np.zeros((hidden, dim), np.float32), "conf_head.0.bias": np.zeros(hidden, np.float32),
                   "conf_head.2.weight": np.zeros((1, hidden), np.float32), "conf_head.2.bias": np.full(1, 3.0, np.float32),
                   "scorer.1.weight": np.ones((2, 2), np.float32)}, str(model / m / "model.safetensors"))
        (model / m / "calibration.json").write_text("{}")
        np.save(feats / f"X_{m}.npy", X)
    (model / "ensemble.json").write_text(json.dumps({"members": ["s", "m"], "weights": [0.5, 0.5]}))
    (feats / "meta.json").write_text(json.dumps({
        "members": ["s", "m"], "weights": [0.5, 0.5], "ok": ok.tolist(), "top": rng.uniform(0.3, 1, n).tolist(),
        "type": ["choice"] * n, "source": ["a", "b"] * (n // 2), "state": [str(i) for i in range(n)]}))
    out = tmp_path / "out"
    ch.fit(str(model), str(feats), str(out), None)

    report = json.loads((out / "confidence-report.json").read_text())
    crossfit = report["crossfit"]["all"]
    # the released head (a constant 0.95) is overconfident; both learned heads beat it on held-out folds
    assert crossfit["released head"]["brier"] > max(crossfit["new head"]["brier"], crossfit["fine-tune released head"]["brier"])
    assert report["selected"] in ("new head", "fine-tune released head")
    for m in ("s", "m"):
        new, old = load_file(str(out / m / "model.safetensors")), load_file(str(model / m / "model.safetensors"))
        assert not np.allclose(new["conf_head.2.bias"], old["conf_head.2.bias"])
        assert np.array_equal(new["scorer.1.weight"], old["scorer.1.weight"])  # only the head changes
        assert old["conf_head.2.bias"][0] == 3.0                              # the original is untouched
        assert (out / m / "calibration.json").is_symlink()
    assert (out / "ensemble.json").is_symlink()
    p = ch.apply_head({k[len("conf_head."):]: v for k, v in load_file(str(out / "s" / "model.safetensors")).items() if k.startswith("conf_head.")}, X)
    assert np.corrcoef(p, X[:, 0])[0, 1] > 0.8  # the new head reads the input that drives correctness
