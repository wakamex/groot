"""Refit the per-type temperatures of a released model or ensemble on calibration rows.

    python scripts/calibrate.py --model Groot-1 --rows data/calibration.jsonl --out Groot-1-recal --types choice noul

For a single model this is groot1.calibration.fit_temperatures on its logits. For an ensemble (a directory with
ensemble.json), each member is first fitted on its own; the members' probabilities are then averaged, and averaging
flattens them, so each question type's member temperatures are multiplied by one shared scale chosen to minimise the
log loss of the averaged output (one parameter per type, so small calibration sets do not trade one member against
the other). Types not named in --types, or with no calibration rows, keep their released temperatures.

The output directory links every file of the model to the original and writes new calibration.json files, so it
loads like the original (groot1.serve.GrootEngine) without copying the weights.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groot1.calibration import Record, fit_temperatures, softmax  # noqa: E402
from groot1.data import read_jsonl  # noqa: E402

FLOOR = 1e-6
SCALES = np.exp(np.linspace(np.log(0.5), np.log(2.0), 41))


def member_records(path: Path, rows: list[dict], device) -> tuple[list[Record], list[float]]:
    from transformers import AutoTokenizer

    from groot1.model import GrootDecisionModel
    from groot1.serialize import Serializer
    from groot1.train import evaluate

    model = GrootDecisionModel.from_pretrained(path, device=device)
    ser = Serializer(AutoTokenizer.from_pretrained(path / "tokenizer"), model.cfg)
    records, _ = evaluate(model, rows, ser, device, 16, None)
    return records, model.temperature.tolist()


def ensemble_nll(members: list[list[Record]], weights: list[float], temps: list[float], idx: list[int]) -> float:
    total = 0.0
    for i in idx:
        p = sum(w * softmax(recs[i].logits / t) for w, recs, t in zip(weights, members, temps))
        t_ = members[0][i].target
        total -= float((t_ * np.log(np.maximum(p, FLOOR))).sum())
    return total / max(1, len(idx))


def link_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        target = dst / f.name
        if f.name in ("calibration.json", "ensemble.json") or target.exists():
            continue
        target.symlink_to(f.resolve(), target_is_directory=f.is_dir())


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", nargs="+", default=["choice", "score", "noul"], help="question types to refit")
    args = ap.parse_args()
    root, out = Path(args.model), Path(args.out)
    rows = list(read_jsonl(args.rows))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ens = json.loads((root / "ensemble.json").read_text()) if (root / "ensemble.json").exists() else None
    names = ens["members"] if ens else ["."]
    weights = ens["weights"] if ens else [1.0]
    fitted = [member_records(root / n, rows, device) for n in names]
    members = [r for r, _ in fitted]
    released = [t for _, t in fitted]
    separate = [fit_temperatures(r, n_types=len(t)) for r, t in fitted]
    report = {"rows": len(rows), "released": released, "separate": separate}
    new = [list(t) for t in released]
    from groot1.config import GrootConfig

    qtypes = list(GrootConfig().qtypes)
    for q in range(len(released[0])):
        idx = [i for i, r in enumerate(members[0]) if r.qtype == q]
        if not idx or qtypes[q] not in args.types:
            continue
        start = [s[q] for s in separate]
        if len(members) == 1:
            best = start
        else:  # one shared scale on the members' separate fits, chosen on the averaged probabilities
            best = min(([t * a for t in start] for a in SCALES), key=lambda ts: ensemble_nll(members, weights, ts, idx))
        before = ensemble_nll(members, weights, [t[q] for t in released], idx)
        after = ensemble_nll(members, weights, best, idx)
        for m, t in enumerate(best):
            new[m][q] = float(t)
        report[f"type{q}"] = {"rows": len(idx), "nll_released": before, "nll_refit": after}
    report["refit"] = new
    out.mkdir(parents=True, exist_ok=True)
    if ens:
        shutil.copy2(root / "ensemble.json", out / "ensemble.json")
        link_tree(root, out)
    for n, t in zip(names, new):
        d = out / n
        if ens:
            (d).unlink() if d.is_symlink() else None
            link_tree(root / n, d)
        else:
            link_tree(root, d)
        (d / "calibration.json").write_text(json.dumps({"temperature": t}, indent=2) + "\n")
    (out / "calibration-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
