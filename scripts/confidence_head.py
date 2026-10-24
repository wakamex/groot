"""Retrain the correctness (p_correct) head on held-out predictions of a frozen model or ensemble.

    python scripts/confidence_head.py features --model Groot-1 --rows heldout.jsonl --out feats
    python scripts/confidence_head.py fit --model Groot-1 --features feats --out Groot-1-conf

The released head was trained on the same batches the model was fitting, so it learned the model's accuracy on
rows it had seen. Here the encoder, scorer and temperatures stay frozen; only conf_head is retrained, on rows the
model never trained on, to predict whether the (ensemble) answer is right. Its architecture and inputs are
unchanged (state root, question token and four probability features), so the new weights load in place.

`features` runs every member over the rows and saves conf_head's inputs, the members' probabilities and the
correctness of the averaged answer. `fit` compares candidate heads by 5-fold cross-fitting over states (folds
never split a state), prints the cross-fitted metrics, trains the selected candidate on every row and writes a model
directory that links to the original files except for each member's model.safetensors, which gets the new head.
It writes nothing unless the best learned head beats the released one on cross-fitted Brier (or --choose names one).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FOLDS = 5
BINS = 10


def members_of(root: Path) -> tuple[list[str], list[float]]:
    if (root / "ensemble.json").exists():
        ens = json.loads((root / "ensemble.json").read_text())
        return ens["members"], ens["weights"]
    return ["."], [1.0]


def features(model_dir: str, rows_path: str, out: str, batch_size: int = 16) -> None:
    import torch
    from transformers import AutoTokenizer

    from groot1.data import read_jsonl
    from groot1.model import GrootDecisionModel
    from groot1.serialize import Serializer
    from groot1.train import build_batch

    root, out_dir = Path(model_dir), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(rows_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, weights = members_of(root)
    probs_all = []
    for name in names:
        model = GrootDecisionModel.from_pretrained(root / name, device=device)
        ser = Serializer(AutoTokenizer.from_pretrained(root / name / "tokenizer"), model.cfg)
        captured = []
        hook = model.conf_head.register_forward_pre_hook(lambda mod, args: captured.append(args[0].detach().float().cpu()))
        X, P = [], []
        with torch.no_grad():
            for s in range(0, len(rows), batch_size):
                chunk = rows[s : s + batch_size]
                batch, _ = build_batch(chunk, ser, None)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    o = model(batch.to(device))
                X.append(captured.pop().numpy())
                pr = o["probabilities"].float().cpu().numpy()
                P += [pr[i, : len(r["options"])] for i, r in enumerate(chunk)]
        hook.remove()
        np.save(out_dir / f"X_{name.replace('/', '_')}.npy", np.concatenate(X))
        probs_all.append(P)
        del model
        torch.cuda.empty_cache()
    ens = [sum(w * p[i] for w, p in zip(weights, probs_all)) for i in range(len(rows))]
    ok = np.array([int(np.argmax(e) == np.argmax(r["target"])) for e, r in zip(ens, rows)])
    meta = {"model": model_dir, "rows": rows_path, "members": names, "weights": weights,
            "source": [r["source"] for r in rows], "type": [r["type"] for r in rows],
            "state": [r.get("state_key", r["id"]) for r in rows], "top": [float(e.max()) for e in ens], "ok": ok.tolist()}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    print(json.dumps({"rows": len(rows), "accuracy": float(ok.mean())}))


# ------------------------------------------------------------------ metrics

def ece(ok: np.ndarray, c: np.ndarray) -> float:
    b = np.minimum((c * BINS).astype(int), BINS - 1)
    gap = np.abs(np.bincount(b, ok, BINS) - np.bincount(b, c, BINS))
    return float(gap.sum() / len(c))


def aurc(ok: np.ndarray, c: np.ndarray) -> float:
    r = ok[np.argsort(-c, kind="stable")] == 0
    return float((np.cumsum(r) / np.arange(1, len(r) + 1)).mean())


def summary(ok: np.ndarray, c: np.ndarray) -> dict:
    return {"ece": ece(ok, c), "brier": float(((c - ok) ** 2).mean()), "aurc": aurc(ok, c), "mean": float(c.mean()), "acc": float(ok.mean())}


# ------------------------------------------------------------------ candidates

def train_head(X: np.ndarray, y: np.ndarray, like: dict, init: bool, steps: int, lr: float, wd: float, seed: int = 0) -> dict:
    """conf_head (Linear 2d+4 -> h, GELU, Linear h -> 1) by full-batch AdamW on binary cross-entropy; shaped like the
    released head `like`, and starting from its weights when init."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(X.shape[1], like["0.weight"].shape[0]), nn.GELU(), nn.Linear(like["0.weight"].shape[0], 1))
    if init:
        head.load_state_dict({k: torch.as_tensor(v) for k, v in like.items()})
    xs, ys = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(head(xs).squeeze(-1), ys).backward()
        opt.step()
    return {k: v.detach().numpy() for k, v in head.state_dict().items()}


def apply_head(w: dict, X: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        z = torch.nn.functional.gelu(torch.as_tensor(X) @ torch.as_tensor(w["0.weight"]).T + torch.as_tensor(w["0.bias"]))
        return torch.sigmoid(z @ torch.as_tensor(w["2.weight"]).T + torch.as_tensor(w["2.bias"])).squeeze(-1).numpy()


def platt_np(c_fit, y_fit, t_fit, c_ev, t_ev):
    """per-type logistic map of the ensemble's top probability: the bar a learned head has to beat"""
    out = np.empty(len(c_ev))
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    for t in np.unique(t_ev):
        x, y = lg(c_fit[t_fit == t]), y_fit[t_fit == t]
        a, b = 1.0, 0.0
        for _ in range(100):
            p = 1 / (1 + np.exp(-np.clip(a * x + b, -30, 30)))
            w = p * (1 - p) + 1e-9
            g = np.array([((p - y) * x).sum(), (p - y).sum()])
            H = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]]) + 1e-6 * np.eye(2)
            a, b = np.array([a, b]) - np.linalg.solve(H, g)
        out[t_ev == t] = 1 / (1 + np.exp(-np.clip(a * lg(c_ev[t_ev == t]) + b, -30, 30)))
    return out


CANDIDATES = {
    "released head": None,
    "fine-tune released head": dict(init=True, steps=300, lr=3e-4, wd=1e-2),
    "new head": dict(init=False, steps=600, lr=1e-3, wd=1e-2),
}


def old_head(root: Path, name: str) -> dict:
    from safetensors.numpy import load_file

    sd = load_file(str(root / name / "model.safetensors"))
    return {k[len("conf_head."):]: sd[k].astype(np.float32) for k in sd if k.startswith("conf_head.")}


def fit(model_dir: str, feats: str, out: str, choose: str | None) -> None:
    root, fdir = Path(model_dir), Path(feats)
    meta = json.loads((fdir / "meta.json").read_text())
    names, weights = meta["members"], meta["weights"]
    X = {n: np.load(fdir / f"X_{n.replace('/', '_')}.npy") for n in names}
    old = {n: old_head(root, n) for n in names}
    ok, top = np.array(meta["ok"]), np.array(meta["top"])
    qtype, src = np.array(meta["type"]), np.array(meta["source"])
    states = sorted(set(meta["state"]))
    rng = np.random.default_rng(0)
    rng.shuffle(states)
    fold_of = {s: i % FOLDS for i, s in enumerate(states)}
    fold = np.array([fold_of[s] for s in meta["state"]])

    preds = {k: np.zeros(len(ok)) for k in [*CANDIDATES, "Platt on top probability"]}
    for f in range(FOLDS):
        tr, ev = fold != f, fold == f
        preds["Platt on top probability"][ev] = platt_np(top[tr], ok[tr], qtype[tr], top[ev], qtype[ev])
        for cand, cfg in CANDIDATES.items():
            members = []
            for n in names:
                w = old[n] if cfg is None else train_head(X[n][tr], ok[tr], old[n], cfg["init"], cfg["steps"], cfg["lr"], cfg["wd"])
                members.append(apply_head(w, X[n][ev]))
            preds[cand][ev] = sum(wt * p for wt, p in zip(weights, members))

    groups = {"all": np.ones(len(ok), bool), **{s: src == s for s in sorted(set(src))}}
    report = {g: {k: summary(ok[m], p[m]) for k, p in preds.items()} for g, m in groups.items()}
    for metric in ("ece", "brier", "aurc"):
        print(f"\ncross-fitted {metric}:\n\n| rows | n | " + " | ".join(preds) + " |\n|" + "---|" * (2 + len(preds)))
        for g, m in groups.items():
            print(f"| {g} | {int(m.sum())} | " + " | ".join(f"{report[g][k][metric]:.4f}" for k in preds) + " |")
    learned = [k for k in CANDIDATES if CANDIDATES[k] is not None]
    pick = choose or min(learned, key=lambda k: report["all"][k]["brier"])
    if not choose and report["all"][pick]["brier"] >= report["all"]["released head"]["brier"]:
        print(f"\nno learned head beats the released one on cross-fitted Brier (best: {pick}); nothing written")
        return
    print(f"\nselected: {pick}")
    if out:
        cfg = CANDIDATES[pick]
        write_model(root, Path(out), names, {n: train_head(X[n], ok, old[n], cfg["init"], cfg["steps"], cfg["lr"], cfg["wd"]) for n in names})
        (Path(out) / "confidence-report.json").write_text(json.dumps({"selected": pick, "crossfit": report, "features": str(fdir)}, indent=2) + "\n")


def link(src: Path, dst: Path, skip: set[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in skip and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve(), target_is_directory=f.is_dir())


def write_model(root: Path, out: Path, names: list[str], heads: dict) -> None:
    from safetensors.numpy import load_file, save_file

    ensemble = names != ["."]
    link(root, out, set(names) if ensemble else {"model.safetensors"})
    for n in names:
        d = out / n if ensemble else out
        if ensemble:
            link(root / n, d, {"model.safetensors"})
        sd = load_file(str(root / n / "model.safetensors"))
        for k, v in heads[n].items():
            assert sd[f"conf_head.{k}"].shape == v.shape, k
            sd[f"conf_head.{k}"] = v.astype(sd[f"conf_head.{k}"].dtype)
        save_file(sd, str(d / "model.safetensors"), metadata={"format": "pt", "family": "groot", "version": "1"})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--model", required=True)
    f.add_argument("--rows", required=True)
    f.add_argument("--out", required=True)
    t = sub.add_parser("fit")
    t.add_argument("--model", required=True)
    t.add_argument("--features", required=True)
    t.add_argument("--out", default=None, help="write the model with the selected head here")
    t.add_argument("--choose", default=None, choices=[k for k in CANDIDATES if CANDIDATES[k] is not None])
    args = ap.parse_args()
    if args.cmd == "features":
        features(args.model, args.rows, args.out)
    else:
        fit(args.model, args.features, args.out, args.choose)


if __name__ == "__main__":
    main()
