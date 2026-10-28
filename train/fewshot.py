"""Pick few-shot examples for the v2 Gemini prompt from Tom's gold labels (SPEC-he.md §0.6).

python -m train.fewshot [--per-cell 2]

Only gold train buckets (< 70) are used, so the gold calibration/test items stay unseen by the teacher.
Per channel (news, social) x class (-2..2) it takes up to --per-cell items, items where v1 Gemini
disagreed with Tom first. Items Gemini called off-target while Tom gave them a direction are left out:
that is a labeling-standard question, not teacher calibration. Writes sentiment/fewshot.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json

from sentiment.llm import FEWSHOT
from train.meta_train import target_cls

from .io import CORPUS, GOLD, TEACHER, META_RESERVED, bucket, read_jsonl

from sentiment.shots import MAX_TEXT, POOL, example_text  # noqa: F401  (re-exported)


def select(corpus_rows, gold_rows, teacher_rows, per_cell: int | None = 2, max_bucket: int = META_RESERVED) -> list[dict]:
    corpus = {}
    for r in corpus_rows:
        corpus.setdefault((r["item"]["id"], r["item"]["ticker"]), r)
    gold = {}
    for g in gold_rows:
        gold[g["id"]] = g
    teacher = {}
    for t in teacher_rows:
        teacher.setdefault((t["id"], t.get("ticker")), t)
    cells: dict[tuple, list] = {}
    for gid, g in gold.items():
        key = (gid, g.get("ticker"))
        if g.get("skipped") or not g.get("relevant", True) or bucket(gid) >= max_bucket or key not in corpus:
            continue
        item = corpus[key]["item"]
        channel = "social" if item.get("channel") == "social" else "news"
        t, tom = teacher.get(key), target_cls(g)
        if t and not t.get("relevant", True) and tom != 0:
            continue
        text = example_text(item)
        if len(text) < 15:
            continue
        disagree = bool(t) and target_cls(t) != tom
        rank = (not disagree, hashlib.sha1(gid.encode()).hexdigest())
        cells.setdefault((channel, tom), []).append((rank, {"id": gid, "ticker": g["ticker"], "channel": channel,
                                                           "text": text, "cls": tom}))
    out = []
    for channel in ("news", "social"):
        for cls in (-2, -1, 0, 1, 2):
            out += [e for _, e in sorted(cells.get((channel, cls), []), key=lambda x: x[0])[:per_cell]]  # [:None] keeps all
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train.fewshot")
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--pool", action="store_true", help="write the full kNN pool (buckets < 65) to sentiment/fewshot_pool.json")
    a = ap.parse_args(argv)
    if a.pool:
        exs = select(read_jsonl(CORPUS), read_jsonl(GOLD), read_jsonl(TEACHER), per_cell=None, max_bucket=65)
        POOL.write_text(json.dumps(exs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(exs)} pool entries -> {POOL}")
        return 0
    exs = select(read_jsonl(CORPUS), read_jsonl(GOLD), read_jsonl(TEACHER), a.per_cell)
    FEWSHOT.write_text(json.dumps(exs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(exs)} examples -> {FEWSHOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
