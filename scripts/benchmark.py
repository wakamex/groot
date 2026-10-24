"""Head-to-head benchmark: Groot vs Julia-1 vs Jev on identical requests.

    python scripts/benchmark.py build --test data/v1/test.jsonl --per-source 500 --sources typed_decisions agnews ... --out bench.jsonl
    python scripts/benchmark.py run --rows bench.jsonl --groot runs/x/final --groot-set state_retrieval=true \
        --julia models/Julia-1 --jev --out bench_results.json
    python scripts/benchmark.py table --results bench_results.json
    python scripts/benchmark.py rescore --rows bench.jsonl --out bench_results.json

Accuracy is argmax agreement with the gold distribution's argmax; Brier is the
multiclass Brier score against the (possibly soft) gold distribution. Log loss is
the cross-entropy against the gold distribution, with each probability floored at
1e-6 so that a single zero cannot make the mean infinite; KL is the same quantity
minus the gold distribution's entropy, and TV is the total variation distance from
it. ECE compares the top probability with accuracy in 10 equal-width bins, and the
bins are saved as a reliability table. RPS, for score questions only, is the ranked
probability score over the ordered levels, divided by (levels - 1). Selective
accuracy at coverage c is the accuracy of the c most confident answers, ranked by
top probability; AURC is the area under the risk-coverage curve (mean error rate of
the k most confident answers over every k; lower is better). For Groot the same
ranking is also reported with its p_correct head.

Intervals are 95% percentile intervals from 1,000 bootstrap resamples of states
(rows sharing a state_key are resampled together, since typed decisions asks
several questions about each state). `paired` holds Groot minus each other
system on the rows both answered, with the same resamples applied to both. Latency is the median wall
time of one request (one question). Groot and Julia run in-process on the same
GPU; Jev is called over its public HTTPS API, so its latency includes the
network round trip.

`run` writes each system's per-row probabilities next to the results
(`bench_results.groot.jsonl`, ...), and `rescore` recomputes every summary from
those files without calling any model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def build(test: str, sources: list[str], per_source: int, out: str, seed: int = 0) -> None:
    by: dict[str, list[dict]] = defaultdict(list)
    for line in open(test, encoding="utf-8"):
        r = json.loads(line)
        if r["source"] in sources:
            by[r["source"]].append(r)
    rng = random.Random(seed)
    rows, counts = [], {}
    for s in sources:
        xs = by.get(s, [])
        rng.shuffle(xs)
        take = xs if s == "typed_decisions" else xs[:per_source]
        counts[s] = len(take)
        rows += take
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(counts))


def to_question(r: dict) -> tuple[dict, list[str]]:
    if r["type"] == "choice":
        crit = {}
        for o in r["options"]:
            name, sep, desc = o.partition(": ")
            crit[name] = desc if sep else None
        return {"type": "choice", "instructions": r["question"], "criteria": crit}, list(crit)
    if r["type"] == "score":
        levels = [o.split(": ", 1)[1] if o.startswith("level ") and ": " in o else o for o in r["options"]]
        return {"type": "score", "instructions": r["question"], "criteria": levels}, [str(i) for i in range(len(levels))]
    q = {"type": "noul", "instructions": r["question"]}
    crit = {}
    for o, k in zip(r["options"], ("false", "true")):
        _, sep, desc = o.partition(": ")
        if sep and desc:
            crit[k] = desc
    if crit:  # keep the dataset's Boolean descriptions (false/true order); literal false/true otherwise
        q["criteria"] = crit
    return q, ["false", "true"]


def probs_of(answer: dict, kind: str, keys: list[str]) -> list[float]:
    if kind == "noul":
        p = float(answer["noul"])
        return [1 - p, p]
    pr = answer["probabilities"]
    v = [float(pr.get(k, 0.0)) for k in keys]
    s = sum(v) or 1.0
    return [x / s for x in v]


SYSTEMS = ("groot", "julia", "jev", "nimble")
LOG_FLOOR = 1e-6
COVERAGES = (0.5, 0.8)
ECE_BINS = 10
BOOTSTRAP = 1000
INTERVAL = ("acc", "brier", "logloss", "kl", "tv", "ece", "rps", "acc@80%", "aurc")
PAIRED = ("acc", "brier", "logloss", "kl", "ece", "acc@80%", "aurc")


def argmax(v: list[float]) -> int:
    return max(range(len(v)), key=v.__getitem__)


def score(r: dict, p: list[float]) -> dict:
    t = r["target"]
    out = {"ok": argmax(p) == argmax(t), "conf": max(p), "brier": sum((a - b) ** 2 for a, b in zip(p, t)),
           "logloss": -sum(b * math.log(max(a, LOG_FLOOR)) for a, b in zip(p, t) if b > 0),
           "kl": sum(b * math.log(b / max(a, LOG_FLOOR)) for a, b in zip(p, t) if b > 0),
           "tv": sum(abs(a - b) for a, b in zip(p, t)) / 2}
    if r["type"] == "score" and len(p) > 1:
        cp = ct = rps = 0.0
        for a, b in zip(p[:-1], t[:-1]):
            cp, ct = cp + a, ct + b
            rps += (cp - ct) ** 2
        out["rps"] = rps / (len(p) - 1)
    return out


def selective(ok: np.ndarray, conf: np.ndarray) -> dict:
    """Accuracy of the most confident answers at fixed coverages, and the area under the risk-coverage curve."""
    ranked = ok[np.argsort(-conf, kind="stable")]
    risks = np.cumsum(~ranked) / np.arange(1, len(ranked) + 1)
    out = {f"acc@{c:.0%}": float(1 - risks[max(1, round(c * len(ranked))) - 1]) for c in COVERAGES}
    out["aurc"] = float(risks.mean())
    return out


def bins_of(conf: np.ndarray) -> np.ndarray:
    return np.minimum((conf * ECE_BINS).astype(int), ECE_BINS - 1)


def ece(ok: np.ndarray, conf: np.ndarray) -> float:
    b = bins_of(conf)
    n = np.bincount(b, minlength=ECE_BINS)
    gap = np.abs(np.bincount(b, ok, ECE_BINS) - np.bincount(b, conf, ECE_BINS))
    return float(gap.sum() / max(1, n.sum()))


def reliability(ok: np.ndarray, conf: np.ndarray) -> list[dict]:
    b = bins_of(conf)
    return [{"bin": [i / ECE_BINS, (i + 1) / ECE_BINS], "n": int((b == i).sum()),
             "conf": float(conf[b == i].mean()), "acc": float(ok[b == i].mean())} for i in range(ECE_BINS) if (b == i).any()]


def metrics(x: dict, idx: np.ndarray | None = None) -> dict:
    """Every summary metric over the rows idx (all rows when None) of the per-row arrays x."""
    if idx is not None:
        x = {k: v[idx] for k, v in x.items()}
    ok, conf = x["ok"], x["conf"]
    out = {"acc": float(ok.mean()), "brier": float(x["brier"].mean()), "logloss": float(x["logloss"].mean()),
           "kl": float(x["kl"].mean()), "tv": float(x["tv"].mean()), "ece": ece(ok, conf), **selective(ok, conf)}
    rps = x["rps"][~np.isnan(x["rps"])]
    if rps.size:
        out["rps"] = float(rps.mean())
    if "p_correct" in x:
        out["p_correct"] = selective(ok, x["p_correct"])
    return out


def arrays(rows: list[dict], res: list[dict]) -> dict:
    scored = [score(r, x["p"]) for r, x in zip(rows, res)]
    x = {k: np.array([sc[k] for sc in scored], dtype=bool if k == "ok" else float) for k in ("ok", "conf", "brier", "logloss", "kl", "tv")}
    x["rps"] = np.array([sc.get("rps", np.nan) for sc in scored])
    if all("p_correct" in r for r in res):
        x["p_correct"] = np.array([r["p_correct"] for r in res])
    return x


def resamples(rows: list[dict], seed: int = 0) -> list[np.ndarray]:
    """Row indices of BOOTSTRAP resamples that draw whole states with replacement."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.get("state_key", r.get("id", i))].append(i)
    clusters = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    return [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]) for _ in range(BOOTSTRAP)]


def interval(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def groups(rows: list[dict]) -> dict[str, list[int]]:
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["source"]].append(i)
        if r["source"] == "typed_decisions":
            by["typed_decisions / " + r["type"]].append(i)
        if r["source"].startswith("massive_"):  # scenario (18 labels) and intent (60 labels) questions
            by[f"{r['source']} / {len(r['options'])} labels"].append(i)
    return by

def run_groot(rows, ckpt, overrides):
    from groot1.config import parse_overrides
    from groot1.schema import SystemOneRequest
    from groot1.serve import GrootEngine

    eng = GrootEngine(ckpt, overrides=parse_overrides(overrides))
    out = []
    for r in rows:
        q, keys = to_question(r)
        req = SystemOneRequest(state=r["state"], questions={"q": q})
        t0 = time.perf_counter()
        resp = eng.answer(req.state, req.questions, extended=True)
        dt = (time.perf_counter() - t0) * 1000
        out.append({"p": probs_of(resp.answers["q"].model_dump(), r["type"], keys),
                    "p_correct": resp.x_groot["q"]["p_correct"], "ms": dt})
    return out


def run_julia(rows, path, device="cuda"):
    import julia

    try:  # Julia's optional fast path breaks on recent transformers; the standard forward is numerically equivalent
        import julia.router.encoder as je

        je.specialize_decision_encoder = lambda model: False
    except Exception:
        pass
    eng = julia.load_model(path, device=device, max_length=8192, head_length=512)
    out = []
    for r in rows:
        q, keys = to_question(r)
        if r["type"] == "choice" and not 2 <= len(keys) <= 20:
            out.append({"skip": f"{len(keys)} options > Julia's 20-option native limit"})
            continue
        if q["type"] == "choice":  # Julia requires a nonempty description per option; bare labels describe themselves
            q = {**q, "criteria": {k: v or k.replace("_", " ") for k, v in q["criteria"].items()}}
        try:
            t0 = time.perf_counter()
            res = eng.predict(state=r["state"], questions={"q": q})
            dt = (time.perf_counter() - t0) * 1000
            pr = res["answers"]["q"]["probabilities"]
            probs = [float(pr.get(k, 0.0)) for k in keys]
            s = sum(probs) or 1.0
            out.append({"p": [x / s for x in probs], "ms": dt})
        except Exception as e:
            out.append({"skip": f"error: {str(e)[:120]}"})
    return out


def run_jev(rows, workers=8):
    import httpx

    key = os.environ["TYPESAFE_API_KEY"]

    def one(r):
        q, keys = to_question(r)
        body = {"model": "jev-latest", "state": r["state"], "questions": {"q": q}}
        headers = {"Authorization": f"Bearer {key}"}
        err = None
        for attempt in range(5):
            try:
                t0 = time.perf_counter()
                resp = httpx.post("https://api.typesafe.ai/v1/systemone", json=body, headers=headers, timeout=120)
                dt = (time.perf_counter() - t0) * 1000
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    return {"skip": f"{resp.status_code} {resp.text[:100]}"}
                return {"p": probs_of(resp.json()["answers"]["q"], r["type"], keys), "ms": dt}
            except Exception as e:
                err = str(e)[:100]
                time.sleep(2 ** attempt)
        return {"skip": f"error: {err}"}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def nimble_field(q: dict) -> dict:
    """A System One question as a Nimble schema field, as nimble/serving/compiler.py maps them."""
    field = {"description": q.get("instructions") or ""}
    if q["type"] == "noul":
        crit = q.get("criteria") or {}
        field.update(type="boolean", choices=[False, True], choice_descriptions={k: crit.get(k) or k for k in ("false", "true")})
    elif q["type"] == "choice":
        field.update(type="enum", choices=list(q["criteria"]), choice_descriptions={k: v or k for k, v in q["criteria"].items()})
    else:
        field.update(type="enum", choices=[str(i) for i in range(len(q["criteria"]))],
                     choice_descriptions={str(i): t for i, t in enumerate(q["criteria"])})
    return field


def run_nimble(rows, model_path: str, repo: str, max_input_tokens: int = 4096):
    """Bespoke Nimble through its own CUDA candidate scorer (bespokelabsai/nimble checkout at `repo`), release
    temperature, one full-prompt prefill per question."""
    import sys

    sys.path.insert(0, repo)
    from nimble.scoring.cuda_scorer import CudaCandidateScorer

    scorer = CudaCandidateScorer(model_path, Path(model_path).name, None, max_input_tokens=max_input_tokens)
    out = []
    for r in rows:
        q, keys = to_question(r)
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        try:
            t0 = time.perf_counter()
            res = scorer.score(state, {"q": nimble_field(q)})
            dt = (time.perf_counter() - t0) * 1000
            scores = res["fields"]["q"]["scores"]
            p = [float(scores[k]) for k in keys]
            out.append({"p": [x / (sum(p) or 1.0) for x in p], "ms": dt})
        except Exception as e:
            out.append({"skip": f"error: {str(e)[:120]}"})
    return out


def summarize(rows, res):
    out = {}
    for k, idx in groups(rows).items():
        done = [i for i in idx if "skip" not in res[i]]
        if not done:
            out[k] = {"n": 0, "skipped": len(idx), "reason": res[idx[0]].get("skip")}
            continue
        sub_rows, sub_res = [rows[i] for i in done], [res[i] for i in done]
        x = arrays(sub_rows, sub_res)
        boot = [metrics(x, ix) for ix in resamples(sub_rows)]
        m = metrics(x)
        out[k] = {"n": len(done), "correct": int(x["ok"].sum()), **m,
                  "ci": {name: interval([b[name] for b in boot]) for name in INTERVAL if name in m},
                  "reliability": reliability(x["ok"], x["conf"]),
                  "p50_ms": statistics.median(r["ms"] for r in sub_res), "skipped": len(idx) - len(done)}
    return out


def paired(rows, a: list[dict], b: list[dict]) -> dict:
    """a minus b on the rows both answered, per group, with intervals from the same state resamples."""
    out = {}
    for k, idx in groups(rows).items():
        both = [i for i in idx if "skip" not in a[i] and "skip" not in b[i]]
        if not both:
            continue
        sub_rows = [rows[i] for i in both]
        xa, xb = arrays(sub_rows, [a[i] for i in both]), arrays(sub_rows, [b[i] for i in both])
        ma, mb = metrics(xa), metrics(xb)
        boot = [(metrics(xa, ix), metrics(xb, ix)) for ix in resamples(sub_rows)]
        out[k] = {"n": len(both), **{name: {"diff": ma[name] - mb[name], "ci": interval([u[name] - v[name] for u, v in boot])} for name in PAIRED}}
    return out


def compare(rows, out: str, results: dict) -> None:
    """results["paired"][other] = Groot minus other, from the saved per-row files."""
    if not per_row_path(out, "groot").exists():
        return
    g = load_per_row(per_row_path(out, "groot"), rows)
    results["paired"] = {name: paired(rows, g, load_per_row(per_row_path(out, name), rows))
                         for name in SYSTEMS[1:] if per_row_path(out, name).exists()}


def per_row_path(out: str, name: str) -> Path:
    p = Path(out)
    return p.with_name(f"{p.stem}.{name}.jsonl")


def save_per_row(path: Path, rows, res) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r, x in zip(rows, res):
            f.write(json.dumps({"id": r.get("id"), **x}, ensure_ascii=False) + "\n")


def load_per_row(path: Path, rows) -> list[dict]:
    res = [json.loads(l) for l in open(path, encoding="utf-8")]
    if [x["id"] for x in res] != [r.get("id") for r in rows]:
        raise SystemExit(f"{path} does not match the rows file")
    return [{k: v for k, v in x.items() if k != "id"} for x in res]


def table(results: dict) -> str:
    systems = [s for s in SYSTEMS if s in results]
    keys = sorted({k for s in systems for k in results[s]}, key=lambda k: (" / " in k, k))
    lines = ["| Benchmark | n | " + " | ".join(f"{s.capitalize()} acc | {s.capitalize()} Brier" for s in systems) + " |",
             "|" + "---|" * (2 + 2 * len(systems))]
    for k in keys:
        n = max(results[s].get(k, {}).get("n", 0) for s in systems)
        cells = []
        for s in systems:
            v = results[s].get(k)
            cells += [f"{v['acc']:.3f}", f"{v['brier']:.3f}"] if v and v.get("n") else ["n/a", "n/a"]
        lines.append(f"| {k} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def calibration_table(results: dict) -> str:
    """Log loss and selective accuracy at 80% coverage per system, plus Groot ranked by p_correct."""
    systems = [s for s in SYSTEMS if s in results]
    keys = sorted({k for s in systems for k in results[s]}, key=lambda k: (" / " in k, k))
    head = [f"{s.capitalize()} log loss | {s.capitalize()} acc@80%" for s in systems]
    if "groot" in systems:
        head.append("Groot acc@80% (p_correct)")
    lines = ["| Benchmark | n | " + " | ".join(head) + " |", "|" + "---|" * (2 + 2 * len(systems) + ("groot" in systems))]
    for k in keys:
        n = max(results[s].get(k, {}).get("n", 0) for s in systems)
        cells = []
        for s in systems:
            v = results[s].get(k)
            cells += [f"{v['logloss']:.3f}", f"{v['acc@80%']:.3f}"] if v and "logloss" in v else ["n/a", "n/a"]
        if "groot" in systems:
            pc = results["groot"].get(k, {}).get("p_correct")
            cells.append(f"{pc['acc@80%']:.3f}" if pc else "n/a")
        lines.append(f"| {k} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def detail_table(results: dict) -> str:
    """ECE, KL and (score questions) RPS per system."""
    systems = [s for s in SYSTEMS if s in results]
    keys = sorted({k for s in systems for k in results[s]}, key=lambda k: (" / " in k, k))
    lines = ["| Benchmark | n | " + " | ".join(f"{s.capitalize()} ECE | {s.capitalize()} KL | {s.capitalize()} RPS" for s in systems) + " |",
             "|" + "---|" * (2 + 3 * len(systems))]
    for k in keys:
        n = max(results[s].get(k, {}).get("n", 0) for s in systems)
        cells = []
        for s in systems:
            v = results[s].get(k, {})
            cells += [f"{v[m]:.3f}" if m in v else "n/a" for m in ("ece", "kl", "rps")]
        lines.append(f"| {k} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_table(results: dict) -> str:
    """Groot minus each other system, with 95% intervals, for Brier, log loss and accuracy at 80% coverage."""
    lines = []
    for other, by in results.get("paired", {}).items():
        lines += [f"Groot minus {other.capitalize()} (negative Brier and log loss, positive acc@80%, favour Groot)", "",
                  "| Benchmark | n | Brier | Log loss | acc@80% |", "|---|---|---|---|---|"]
        for k in sorted(by, key=lambda k: (" / " in k, k)):
            v = by[k]
            cells = [f"{v[m]['diff']:+.3f} [{v[m]['ci'][0]:+.3f}, {v[m]['ci'][1]:+.3f}]" for m in ("brier", "logloss", "acc@80%")]
            lines.append(f"| {k} | {v['n']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def all_tables(results: dict) -> str:
    return "\n\n".join(t for t in (table(results), calibration_table(results), detail_table(results), paired_table(results)) if t)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--test", required=True)
    b.add_argument("--sources", nargs="+", required=True)
    b.add_argument("--per-source", type=int, default=500)
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rows", required=True)
    r.add_argument("--groot", default=None)
    r.add_argument("--groot-set", action="append", default=[])
    r.add_argument("--julia", default=None)
    r.add_argument("--jev", action="store_true")
    r.add_argument("--nimble", default=None, help="merged Nimble checkpoint directory")
    r.add_argument("--nimble-repo", default=None, help="bespokelabsai/nimble checkout providing its scorer")
    r.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("--results", required=True)
    rs = sub.add_parser("rescore")
    rs.add_argument("--rows", required=True)
    rs.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.test, args.sources, args.per_source, args.out)
        return
    if args.cmd == "table":
        print(all_tables(json.loads(Path(args.results).read_text())))
        return
    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    if args.cmd == "rescore":
        for name in SYSTEMS:
            if per_row_path(args.out, name).exists():
                results[name] = summarize(rows, load_per_row(per_row_path(args.out, name), rows))
        compare(rows, args.out, results)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(all_tables(results))
        return
    for name, fn in (("groot", lambda: run_groot(rows, args.groot, args.groot_set) if args.groot else None),
                     ("julia", lambda: run_julia(rows, args.julia) if args.julia else None),
                     ("jev", lambda: run_jev(rows) if args.jev else None),
                     ("nimble", lambda: run_nimble(rows, args.nimble, args.nimble_repo) if args.nimble else None)):
        res = fn()
        if res is None:
            continue
        save_per_row(per_row_path(args.out, name), rows, res)
        results[name] = summarize(rows, res)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"{name} done", flush=True)
    compare(rows, args.out, results)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(all_tables(results))


if __name__ == "__main__":
    main()
