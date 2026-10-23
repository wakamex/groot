"""Folha news benchmark rows: zero-shot newsroom-section classification in Portuguese.

    python scripts/folha.py --out data/folha.jsonl
    python scripts/folha.py --dev --out data/folha-dev.jsonl    # disjoint development rows, for calibration

400 articles from emdemor/news-of-the-brazilian-newspaper (Folha de S.Paulo), 50 per section for
8 sections, drawn with a fixed seed. Every section is offered with a one-line Portuguese
description; the gold label is the section the newsroom filed the article under. This follows the
README's Folha protocol, but the question, the section descriptions and the sampling are this
script's own, so its rows are not the 400 articles behind the README's main tables. The rows use
the benchmark row format, so they can be appended to the file `benchmark.py build` writes.
"""
from __future__ import annotations

import argparse
import json
import random

from groot1.data import make_row

SECTIONS = {
    "poder": "Política, governo, eleições, Congresso e Justiça",
    "mercado": "Economia, empresas, finanças e mercado de trabalho",
    "mundo": "Notícias internacionais e política externa",
    "esporte": "Esportes, atletas, clubes e competições",
    "cotidiano": "Cidades, segurança, trânsito, educação e vida urbana",
    "ilustrada": "Cultura, artes, cinema, música, televisão e celebridades",
    "tec": "Tecnologia, internet, aplicativos e ciência da computação",
    "equilibrioesaude": "Saúde, medicina, bem-estar e qualidade de vida",
}
QUESTION = "Em qual editoria do jornal esta notícia foi publicada?"


def build(per_section: int = 50, seed: int = 0, dev: bool = False, max_chars: int = 4000) -> list[dict]:
    """The test rows, or with dev the same number per section drawn (seed + 1) from the articles the test leaves out."""
    from datasets import load_dataset

    ds = load_dataset("emdemor/news-of-the-brazilian-newspaper", split="train")
    by: dict[str, list[int]] = {s: [] for s in SECTIONS}
    for i, (cat, text) in enumerate(zip(ds["category"], ds["text"])):
        if cat in by and text and text.strip():
            by[cat].append(i)
    rng = random.Random(seed)
    options = [f"{k}: {v}" for k, v in SECTIONS.items()]
    rows = []
    picks = []
    for section, idx in by.items():
        if len(idx) < (2 if dev else 1) * per_section:
            raise SystemExit(f"only {len(idx)} articles in {section}")
        picks.append(sorted(rng.sample(idx, per_section)))
    if dev:
        drng = random.Random(seed + 1)
        picks = [sorted(drng.sample(sorted(set(idx) - set(test)), per_section)) for idx, test in zip(by.values(), picks)]
    split = "val" if dev else "test"
    for g, chosen in enumerate(picks):
        for i in chosen:
            r = ds[i]
            state = f"{(r['title'] or '').strip()}\n\n{r['text'].strip()[:max_chars]}".strip()
            target = [0.0] * len(SECTIONS)
            target[g] = 1.0
            rows.append(make_row("folha_news", i, state, QUESTION, "choice", options, target, split, "pt"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-section", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    rows = build(args.per_section, args.seed, args.dev)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "sections": len(SECTIONS)}))


if __name__ == "__main__":
    main()
