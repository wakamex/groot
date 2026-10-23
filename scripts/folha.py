"""Folha news benchmark rows: zero-shot newsroom-section classification in Portuguese.

    python scripts/folha.py --out data/folha.jsonl

400 articles from emdemor/news-of-the-brazilian-newspaper (Folha de S.Paulo), 50 per section for
8 sections, drawn with a fixed seed. Every section is offered with a one-line Portuguese
description; the gold label is the section the newsroom filed the article under. The rows use
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


def build(per_section: int = 50, seed: int = 0, max_chars: int = 4000) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("emdemor/news-of-the-brazilian-newspaper", split="train")
    by: dict[str, list[int]] = {s: [] for s in SECTIONS}
    for i, (cat, text) in enumerate(zip(ds["category"], ds["text"])):
        if cat in by and text and text.strip():
            by[cat].append(i)
    rng = random.Random(seed)
    options = [f"{k}: {v}" for k, v in SECTIONS.items()]
    rows = []
    for g, (section, idx) in enumerate(by.items()):
        if len(idx) < per_section:
            raise SystemExit(f"only {len(idx)} articles in {section}")
        for i in sorted(rng.sample(idx, per_section)):
            r = ds[i]
            state = f"{(r['title'] or '').strip()}\n\n{r['text'].strip()[:max_chars]}".strip()
            target = [0.0] * len(SECTIONS)
            target[g] = 1.0
            rows.append(make_row("folha_news", i, state, QUESTION, "choice", options, target, "test", "pt"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-section", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = build(args.per_section, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "sections": len(SECTIONS)}))


if __name__ == "__main__":
    main()
