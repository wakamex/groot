"""Full-size benchmark inputs for the Bend engine, and the PyTorch reference that runs the same requests.

    python scripts/bend_bench.py prepare --model Groot-1-S --out bench-bend
    python scripts/bend_bench.py reference --model Groot-1-S --request bench-bend/req-1024.json --device cpu --threads 16

`prepare` writes, from the model and this repository's own text:

    model.gbw          flat float32 weights (groot-export bend, no vocabulary pruning)
    layout.bend        the matching bend/groot/layout.bend; copy it over before building bend/main.bend
    req-N.bin          u32 request blob for a state of N tokens (groot1.bendio.RequestBlob)
    req-N.json         the same request as token structures, for `reference`
    ref-N.txt          PyTorch float32 outputs on CPU, one value per line, in the Bend output order
    manifest.json      sizes and SHA-256 of every file

The state is the start of this repository's README, license and Python sources, cut to N tokens.
The questions are fixed: a cycle of choice (2, 4, 8 and 16 options), score and true/false questions.
Nothing is random, so every run of `prepare` on the same commit and model writes identical files.

`reference` runs one request through PyTorch in float32 and prints the outputs like the Bend binary
(`./groot model.gbw req-N.bin --threads T`), with the forward time on stderr.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groot1.bendio import RequestBlob, data_offset, layout_from_gbw, reference_outputs  # noqa: E402
from groot1.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion  # noqa: E402
from groot1.serialize import EncodedQuestion  # noqa: E402

SIZES = (1024, 8192, 32768)
CORPUS = ["README.md", "LICENSE", "NOTICE", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "groot1").glob("*.py"))]
TOPICS = {
    "training": "Model training, optimisation and checkpoints",
    "data": "Datasets, conversion and sampling of training rows",
    "serving": "The HTTP server and request handling",
    "calibration": "Temperatures, confidence and probability scoring",
    "export": "Exporting weights to Bend or ONNX",
    "evaluation": "Benchmarks and evaluation metrics",
    "licensing": "Licences, notices and attribution",
    "architecture": "Encoder, tree memory and fusion layers",
    "tokenization": "Tokenizers, token ids and serialization",
    "retrieval": "Selecting pieces of a long state",
    "schema": "Request and response formats",
    "parallelism": "Batching, threads and devices",
    "testing": "Unit tests and fixtures",
    "documentation": "Installation and usage instructions",
    "performance": "Latency and memory use",
    "languages": "Multilingual and Portuguese support",
}
CHOICE = ["Which subject does this text cover most?", "Which area would a reader consult this text for?",
          "What is the main concern of this text?", "Which part of the project does this text describe?"]
SCORE = ["How technical is this text?", "How complete is the documentation in this text?",
         "How much of this text is source code?", "How formal is the language of this text?"]
LEVELS = ["not at all", "slightly", "moderately", "mostly", "entirely"]
NOUL = ["The text describes a machine learning model.", "The text is written in Portuguese.",
        "The text contains Python code.", "The text explains how to train the model.",
        "The text mentions a software license.", "The text is a recipe for cooking."]


def questions(n: int) -> dict:
    """n fixed questions: choice, score and true/false in turn; choice questions offer 2, 4, 8 or 16 topics."""
    keys = list(TOPICS)
    out = {}
    for i in range(n):
        kind, j = i % 3, i // 3
        if kind == 0:
            k = (2, 4, 8, 16)[j % 4]
            names = [keys[(j + t) % len(keys)] for t in range(k)]
            out[f"q{i}"] = ChoiceQuestion(instructions=CHOICE[j % len(CHOICE)], criteria={t: TOPICS[t] for t in names})
        elif kind == 1:
            out[f"q{i}"] = ScoreQuestion(instructions=SCORE[j % len(SCORE)], criteria=LEVELS)
        else:
            out[f"q{i}"] = NoulQuestion(instructions=NOUL[j % len(NOUL)])
    return out


def state_of(tok, ser, ids: list[int], n: int) -> str:
    """The longest prefix of the corpus whose serialized state has at most n tokens.

    The serializer normalizes whitespace, so a prefix of k corpus tokens serializes to fewer than k."""
    lo, hi = 0, len(ids)
    if ser.encode_state(tok.decode(ids)).tokens < n:
        raise SystemExit(f"the corpus serializes to fewer than {n} state tokens")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ser.encode_state(tok.decode(ids[:mid])).tokens <= n:
            lo = mid
        else:
            hi = mid - 1
    return tok.decode(ids[:lo])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(model_dir: str, out: Path, sizes: list[int], n_questions: int) -> None:
    from transformers import AutoTokenizer

    from groot1.export import write_gbw
    from groot1.model import GrootDecisionModel
    from groot1.serialize import Serializer

    out.mkdir(parents=True, exist_ok=True)
    model = GrootDecisionModel.from_pretrained(model_dir, device="cpu").float().eval()
    tok = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
    write_gbw(model, out / "model.gbw")
    layout_from_gbw(out / "model.gbw", out / "layout.bend")
    ser = Serializer(tok, model.cfg)
    text = "\n\n".join((ROOT / p).read_text(encoding="utf-8") for p in CORPUS)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    qs = questions(n_questions)
    requests = {}
    for n in sizes:
        enc = ser.encode_request(state_of(tok, ser, ids, n), qs)
        chunks, eqs = enc.state.chunks, enc.questions
        RequestBlob.write(RequestBlob.build(chunks, eqs), out / f"req-{n}.bin")
        (out / f"req-{n}.json").write_text(json.dumps({"chunks": chunks, "questions": [dataclasses.asdict(q) for q in eqs]}))
        ref = reference_outputs(model, chunks, eqs)
        (out / f"ref-{n}.txt").write_text("".join(f"{float(v)!r}\n" for v in ref))
        requests[n] = {"state_tokens": enc.state.tokens, "chunks": len(chunks), "questions": len(eqs),
                       "question_tokens": sum(len(q.ids) for q in eqs), "options": sum(q.n_options for q in eqs),
                       "outputs": int(ref.size), "truncated": enc.state.truncated or any(q.truncated for q in eqs)}
        print(json.dumps({"size": n, **requests[n]}), flush=True)
    files = sorted(p.name for p in out.iterdir() if p.name != "manifest.json")
    manifest = {"model": str(model_dir), "weights_data_offset": data_offset(out / "model.gbw"), "requests": requests,
                "sha256": {f: sha256(out / f) for f in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def reference(model_dir: str, request: Path, device: str, threads: int | None) -> None:
    import torch

    from groot1.model import GrootDecisionModel

    if threads:
        torch.set_num_threads(threads)
    req = json.loads(request.read_text())
    eqs = [EncodedQuestion(**q) for q in req["questions"]]
    model = GrootDecisionModel.from_pretrained(model_dir, device=device).float().eval()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = reference_outputs(model, req["chunks"], eqs)
    dt = (time.perf_counter() - t0) * 1000
    sys.stdout.write("".join(f"{float(v)!r}\n" for v in out))
    print(f"torch forward ms = {dt:.0f}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--questions", type=int, default=64)
    r = sub.add_parser("reference")
    r.add_argument("--model", required=True)
    r.add_argument("--request", required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.model, Path(args.out), args.sizes, args.questions)
    else:
        reference(args.model, Path(args.request), args.device, args.threads)


if __name__ == "__main__":
    main()
