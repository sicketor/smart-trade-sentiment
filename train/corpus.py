"""Collect a training corpus of real items over a ticker universe (SPEC §5.0).

python -m train.corpus [--universe data/universe.txt] [--tickers NVDA,AMD] [--days 7] [--limit-tickers N]
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sentiment import config
from sentiment.dedup import cluster
from sentiment.http import Http
from sentiment.linking import relevance
from sentiment.sources import collect
from sentiment.text import normalize
from sentiment.tickers import get_company

from .io import CORPUS, append_jsonl, company_to_json, item_to_json, read_jsonl

log = logging.getLogger(__name__)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_UNIVERSE = config.ROOT / "data" / "universe.txt"


def make_embed():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    return lambda texts: model.encode(texts, normalize_embeddings=True, batch_size=64)


def collect_ticker(ticker: str, http, days: int, embed, info: dict | None = None) -> list[dict]:
    """Rows for one ticker. `info` (optional) receives {"ok": [...], "failed": {...}}."""
    company = get_company(ticker, http)
    if company is None:
        raise ValueError(f"unknown US ticker: {ticker}")
    now = datetime.now(timezone.utc)
    raw, ok, failed = collect(company, http, now - timedelta(days=days))
    if info is not None:
        info["ok"], info["failed"] = ok, failed
    return to_rows(raw, company, embed, now)


def to_rows(items: list, company, embed, now: datetime) -> list[dict]:
    """Normalize, drop irrelevant/untitled items, cluster, and build corpus rows."""
    clean = [dataclasses.replace(it, title=normalize(it.title), text=normalize(it.text)) for it in items]
    rel = {it.id: relevance(it, company) for it in clean}
    clean = [it for it in clean if rel[it.id] > 0 and it.title]
    if not clean:
        return []
    clusters = cluster(clean, embed)
    cj = company_to_json(company)
    stamp = now.isoformat()
    rows = []
    for it in clean:
        cid, nov = clusters.get(it.id, (it.id[:12], 1.0))
        rows.append({"item": item_to_json(it), "company": cj, "relevance": round(rel[it.id], 3),
                     "cluster_id": cid, "novelty": round(nov, 4), "collected_at": stamp})
    return rows


def read_universe(path) -> list[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = line.split("#", 1)[0].strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train.corpus")
    ap.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    ap.add_argument("--tickers", default="")
    ap.add_argument("--days", type=int, default=config.LOOKBACK_DAYS)
    ap.add_argument("--limit-tickers", type=int, default=0)
    a = ap.parse_args(argv)

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()] or read_universe(a.universe)
    if a.limit_tickers > 0:
        tickers = tickers[: a.limit_tickers]
    seen = {(r["item"]["id"], r["item"].get("ticker")) for r in read_jsonl(CORPUS) if "item" in r}
    http, embed = Http(), make_embed()
    total = 0
    for t in tickers:
        info: dict = {}
        try:
            rows = collect_ticker(t, http, a.days, embed, info)
            new = [r for r in rows if (r["item"]["id"], r["item"]["ticker"]) not in seen]
            append_jsonl(CORPUS, new)
            seen.update((r["item"]["id"], r["item"]["ticker"]) for r in new)
            total += len(new)
            failed = ",".join(info.get("failed", {})) or "-"
            print(f"{t}: new={len(new)} ok={len(info.get('ok', []))} failed={failed}", flush=True)
        except Exception as e:
            print(f"{t}: ERROR {type(e).__name__}: {e}", flush=True)
    print(f"total new rows: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
