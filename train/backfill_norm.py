"""Cross-section normalization backfill (aggregate_norm.json).

python -m train.backfill_norm score [--device auto] [--chunk 512]
python -m train.backfill_norm build [--out sentiment/aggregate_norm.json] [--dump runs/backfill_norm.json] [--min-n 30]

score: every corpus item is scored once by the runtime models (resumable cache data/backfill/scored.jsonl).
build: replays each ticker's snapshots at the real collection times (latest per UTC date), collects the raw
channel / composite scores from aggregate.summarize and writes mean, std and percentile cuts per group.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sentiment import config
from sentiment.features import extract
from sentiment.types import ModelOutput, ScoredItem

from .io import CORPUS, append_jsonl, company_from_json, item_from_json, read_jsonl

CACHE = config.ROOT / "data" / "backfill" / "scored.jsonl"
CHANNELS = ("news", "filings", "social")
GROUPS = CHANNELS + ("composite",)


def _key(r: dict) -> tuple:
    it = r["item"]
    return (it["id"], it.get("ticker"))


def corpus_rows() -> list[dict]:
    seen, out = set(), []
    for r in read_jsonl(CORPUS):
        k = _key(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _aware(d: datetime) -> datetime:
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def _utc(s: str) -> datetime:
    return _aware(datetime.fromisoformat(s))


def ticker_days(rows) -> dict[str, list[datetime]]:
    """{ticker: sorted as_of list}: the latest collected_at of each UTC date (naive = UTC)."""
    best: dict[str, dict] = defaultdict(dict)
    for r in rows:
        t = r["item"].get("ticker")
        c = _utc(r["collected_at"])
        d = c.date()
        if d not in best[t] or c > best[t][d]:
            best[t][d] = c
    return {t: sorted(v.values()) for t, v in best.items()}


def build_norm(records: list[dict], min_n: int = 30) -> dict:
    """records: per ticker-day dicts {"raw": {ch: float|None}, "raw_composite": float|None, ...}."""
    from sentiment.aggregate import percentile_cuts
    groups = {}
    for g in GROUPS:
        vals = [(r["raw_composite"] if g == "composite" else (r.get("raw") or {}).get(g)) for r in records]
        x = np.array([v for v in vals if v is not None], dtype=float)
        if len(x) < max(min_n, 2):
            continue
        mu, sd = float(x.mean()), float(x.std(ddof=1))
        if sd <= 0:
            continue
        cuts = percentile_cuts([float(z) for z in (x - mu) / sd])
        groups[g] = {"mu": round(mu, 4), "sd": round(sd, 4), "n": int(len(x)),
                     "cuts": [round(float(c), 4) for c in cuts]}
    return groups


def cmd_score(args) -> int:
    from sentiment.models.runner import ModelRunner
    done = {(r["id"], r["ticker"]) for r in read_jsonl(CACHE)}
    rows = [r for r in corpus_rows() if _key(r) not in done]
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["item"].get("ticker"), r["item"]["channel"])].append(r)
    runner = ModelRunner(device=args.device)
    total = len(done)
    n_groups = len(by_key)
    for i, ((ticker, channel), group) in enumerate(by_key.items(), 1):
        company = company_from_json(group[0]["company"])
        for s in range(0, len(group), args.chunk):
            items = [item_from_json(r["item"]) for r in group[s:s + args.chunk]]
            scored = runner.score_items(items, company, channel)
            out = []
            for it in items:
                f, ev, drivers = extract(it, company)
                outs = scored.get(it.id, [])
                out.append({"id": it.id, "ticker": it.ticker,
                            "outputs": {o.model: list(o.probs) for o in outs},
                            "features": f, "events": ev, "drivers": drivers})
            append_jsonl(CACHE, out)
            total += len(out)
        print(f"{i}/{n_groups} {ticker} {channel} {len(group)} done={total}", flush=True)
    return 0


def scored_by_ticker(rows: list[dict] | None = None, cache: dict | None = None,
                     meta=None) -> dict[str, list[tuple[ScoredItem, datetime, datetime]]]:
    """{ticker: [(ScoredItem, published_at UTC, collected_at UTC)]} rebuilt from the score cache and the corpus
    rows (meta predicted on CPU from the cached model outputs). Defaults load the cache, corpus and Meta();
    corpus rows without a cached score are left out."""
    if cache is None:
        cache = {(r["id"], r["ticker"]): r for r in read_jsonl(CACHE)}
    if rows is None:
        rows = [r for r in corpus_rows() if _key(r) in cache]
    if meta is None:
        from sentiment.meta import Meta
        meta = Meta()
    scored: dict[str, list[tuple[ScoredItem, datetime, datetime]]] = defaultdict(list)
    for r in rows:
        c = cache.get(_key(r))
        if c is None:
            continue
        item = item_from_json(r["item"])
        outputs = [ModelOutput(m, tuple(p)) for m, p in c["outputs"].items()]
        m = meta.predict(item.channel, outputs, c["features"], c["events"], None)
        si = ScoredItem(item=item, relevance=r["relevance"], cluster_id=r["cluster_id"], novelty=r["novelty"],
                        outputs=outputs, features=c["features"], events=c["events"], probs5=m.probs5,
                        score=m.score, cls=m.cls, confidence=m.confidence, stance=m.stance, llm=None,
                        drivers=c.get("drivers", []))
        scored[item.ticker].append((si, _aware(item.published_at), _utc(r["collected_at"])))
    return scored


def cmd_build(args) -> int:
    from sentiment.aggregate import summarize
    from sentiment.meta import Meta
    cache = {(r["id"], r["ticker"]): r for r in read_jsonl(CACHE)}
    if not cache:
        print("no scored items; run score first")
        return 1
    rows = [r for r in corpus_rows() if _key(r) in cache]
    meta = Meta()
    scored = scored_by_ticker(rows, cache, meta)
    records = []
    for ticker, as_ofs in ticker_days(rows).items():
        for as_of in as_ofs:
            lo = as_of - timedelta(days=config.LOOKBACK_DAYS)
            items = [si for si, pub, col in scored.get(ticker, []) if lo <= pub <= as_of and col <= as_of]
            s = summarize(items, as_of)
            records.append({"ticker": ticker, "as_of": as_of.isoformat(), "n_items": len(items),
                            "raw": {ch: s["channels"][ch].raw for ch in CHANNELS},
                            "raw_composite": s["composite"].raw})
    groups = build_norm(records, args.min_n)
    doc = {"version": 1, "built_at": datetime.now(timezone.utc).isoformat(), "source": "corpus-backfill",
           "meta_version": meta.version, "n_ticker_days": len(records), "groups": groups}
    for path, payload in ((args.out, doc), (args.dump, records)):
        p = Path(path)
        if not p.is_absolute():
            p = config.ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'group':<10}{'n':>6}{'mu':>10}{'sd':>10}  cuts")
    for g, v in groups.items():
        print(f"{g:<10}{v['n']:>6}{v['mu']:>10}{v['sd']:>10}  {v['cuts']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="train.backfill_norm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--device", default="auto")
    s.add_argument("--chunk", type=int, default=512)
    b = sub.add_parser("build")
    b.add_argument("--out", default="sentiment/aggregate_norm.json")
    b.add_argument("--dump", default="runs/backfill_norm.json")
    b.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args(argv)
    return cmd_score(args) if args.cmd == "score" else cmd_build(args)


if __name__ == "__main__":
    sys.exit(main())
