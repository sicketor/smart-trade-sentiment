"""Weak labels from Marketaux: news with a per-ticker entity sentiment score.

harvest (round-robin over tickers, resumable by a published_before cursor, request budget per run)
-> raw jsonl -> ingest into the corpus as source "marketaux"; `stats` reports how the Marketaux score
agrees with Tom's gold labels and with the Gemini teacher labels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timedelta, timezone

from sentiment import config
from sentiment.http import HttpError
from sentiment.sources._util import item_id, parse_iso, strip_html
from sentiment.text import title_key
from sentiment.types import RawItem

from . import corpus
from .corpus import DEFAULT_UNIVERSE, read_universe
from .io import CORPUS, GOLD, TEACHER, append_jsonl, latest_teacher, read_jsonl

API = "https://api.marketaux.com/v1/news/all"
WEAK = config.ROOT / "data" / "weak"
RAW = WEAK / "marketaux_raw.jsonl"
STATE = WEAK / "marketaux_state.json"


def article_row(a: dict, ticker: str) -> dict | None:
    entities = a.get("entities") or []
    ent = next((e for e in entities if str(e.get("symbol") or "").upper() == ticker.upper()), None)
    if ent is None or ent.get("sentiment_score") is None:
        return None
    title = strip_html(a.get("title"))
    dt = parse_iso(a.get("published_at"))
    if not title or dt is None:
        return None
    return {
        "uuid": a["uuid"],
        "ticker": ticker,
        "title": title,
        "description": strip_html(a.get("description")),
        "snippet": strip_html(a.get("snippet")),
        "url": a.get("url") or "",
        "source": a.get("source") or "",
        "published_at": dt.isoformat(),
        "score": float(ent["sentiment_score"]),
        "match_score": ent.get("match_score"),
        "highlights": [{"text": strip_html(h.get("highlight")), "sentiment": h.get("sentiment")}
                       for h in (ent.get("highlights") or [])[:5]],
        "n_entities": len(entities),
    }


def harvest(tickers, http, state: dict, seen: set, key: str, *, max_requests: int = 95, days: int = 365,
            now: datetime | None = None) -> tuple[list[dict], int, str | None]:
    now = now or datetime.now(timezone.utc)
    horizon = now - timedelta(days=days)
    for t in tickers:
        state.setdefault(t, {"cursor": None, "done": False, "n": 0})
    out: list[dict] = []
    requests = 0
    while requests < max_requests:
        active = [t for t in tickers if not state[t]["done"]]
        if not active:
            break
        for t in active:
            if requests >= max_requests:
                break
            st = state[t]
            params = {"symbols": t, "filter_entities": "true", "language": "en", "api_token": key}
            old = parse_iso(st["cursor"]) if st["cursor"] else None
            if old is not None:
                params["published_before"] = old.strftime("%Y-%m-%dT%H:%M:%S")
            requests += 1
            try:
                j = http.get_json(API, params=params)
            except HttpError as e:
                return out, requests, str(e)
            data = (j or {}).get("data") or []
            if not data:
                st["done"] = True
                continue
            kept = 0
            for a in data:
                r = article_row(a, t)
                if r is None or (r["uuid"], t) in seen:
                    continue
                seen.add((r["uuid"], t))
                out.append(r)
                kept += 1
            st["n"] += kept
            dts = [d for d in (parse_iso(a.get("published_at")) for a in data) if d is not None]
            if not dts:
                st["done"] = True
                continue
            new = min(dts) - timedelta(seconds=1)
            if old is not None and new >= old:
                st["done"] = True
            elif new <= horizon:
                st["done"] = True
            st["cursor"] = new.isoformat()
    return out, requests, None


def to_label(score: float, t: float = 0.15) -> int:
    return 1 if score >= t else -1 if score <= -t else 0


def raw_to_item(r: dict) -> RawItem:
    return RawItem(id=item_id("marketaux", r["url"], r["title"]), source="marketaux", channel="news",
                   publisher=r["source"], title=r["title"], text=r["description"] or r["snippet"],
                   url=r["url"], published_at=datetime.fromisoformat(r["published_at"]), ticker=r["ticker"],
                   meta={"mx_score": r["score"], "mx_uuid": r["uuid"]})


def match_gold(raw_rows, corpus_rows, gold_rows) -> list[tuple[dict, int]]:
    latest: dict = {}
    for g in gold_rows:
        if "id" in g:
            latest[(g["id"], g.get("ticker"))] = g
    gold = {k: g for k, g in latest.items() if not g.get("skipped") and g.get("cls") is not None}
    by_url: dict = {}
    by_title: dict = {}
    for c in corpus_rows:
        it = c.get("item") or {}
        k = (it.get("id"), it.get("ticker"))
        if k not in gold:
            continue
        if it.get("url"):
            by_url.setdefault((it.get("ticker"), it["url"]), k)
        if it.get("title"):
            by_title.setdefault((it.get("ticker"), title_key(it["title"])), k)
    used: set = set()
    out = []
    for r in raw_rows:
        cands = [by_url.get((r["ticker"], r["url"])) if r.get("url") else None,
                 by_title.get((r["ticker"], title_key(r["title"]))) if r.get("title") else None,
                 (raw_to_item(r).id, r["ticker"])]
        k = next((c for c in cands if c is not None and c in gold), None)
        if k is None or k in used:
            continue
        used.add(k)
        g = gold[k]
        out.append((r, 0 if g.get("relevant") is False else int(g["cls"])))
    return out


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def agreement(pairs: list[tuple[float, int]], t: float = 0.15) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    ref = [_sign(c) for _, c in pairs]
    mx = [to_label(s, t) for s, _ in pairs]
    polar = [(m, r) for m, r in zip(mx, ref) if m != 0 and r != 0]
    rho = None
    if n >= 3 and len({s for s, _ in pairs}) > 1 and len({c for _, c in pairs}) > 1:
        from scipy.stats import spearmanr
        v = float(spearmanr([s for s, _ in pairs], [c for _, c in pairs])[0])
        rho = None if math.isnan(v) else round(v, 3)
    cnt = Counter(mx)
    return {"n": n, "dir_acc": round(sum(m == r for m, r in zip(mx, ref)) / n, 3),
            "polar_n": len(polar),
            "polar_acc": round(sum(m == r for m, r in polar) / len(polar), 3) if polar else None,
            "spearman": rho, "mx_dist": {"-1": cnt[-1], "0": cnt[0], "1": cnt[1]}}


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train.weak_marketaux")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    h.add_argument("--tickers", default="")
    h.add_argument("--max-requests", type=int, default=95)
    h.add_argument("--days", type=int, default=365)
    sub.add_parser("ingest")
    s = sub.add_parser("stats")
    s.add_argument("--t", type=float, default=0.15)
    a = ap.parse_args(argv)

    if a.cmd == "harvest":
        from dotenv import load_dotenv
        from sentiment.http import Http
        load_dotenv(config.ROOT / ".env")
        key = os.environ.get("MARKETAUX_API_KEY")
        if not key:
            print("MARKETAUX_API_KEY missing")
            return 1
        tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()] or read_universe(a.universe)
        seen = {(r["uuid"], r["ticker"]) for r in read_jsonl(RAW) if "uuid" in r}
        state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
        rows, n, err = harvest(tickers, Http(), state, seen, key, max_requests=a.max_requests, days=a.days)
        append_jsonl(RAW, rows)
        _save_state(state)
        done = sum(1 for t in tickers if state.get(t, {}).get("done"))
        print(f"requests={n} new={len(rows)} done={done}/{len(tickers)}")
        if err:
            print(f"stopped: {err}")
        return 0
    if a.cmd == "ingest":
        from sentiment.http import Http
        from sentiment.tickers import get_company
        by_t: dict[str, list[dict]] = {}
        for r in read_jsonl(RAW):
            by_t.setdefault(r["ticker"], []).append(r)
        have = {(r["item"]["id"], r["item"].get("ticker")) for r in read_jsonl(CORPUS) if "item" in r}
        http, embed, total = Http(), None, 0
        for t, rs in by_t.items():
            items, ids = [], set()   # two uuids can share a url, hence one item id
            for it in map(raw_to_item, rs):
                if (it.id, t) not in have and it.id not in ids:
                    ids.add(it.id)
                    items.append(it)
            if not items:
                continue
            company = get_company(t, http)
            if company is None:
                print(f"{t}: unknown ticker, skipped")
                continue
            if embed is None:
                embed = corpus.make_embed()
            rows = corpus.to_rows(items, company, embed, datetime.now(timezone.utc))
            append_jsonl(CORPUS, rows)
            have.update((x["item"]["id"], t) for x in rows)
            total += len(rows)
            print(f"{t}: new={len(rows)}", flush=True)
        print(f"total new rows: {total}")
        return 0

    raw = list(read_jsonl(RAW))
    corpus_rows = list(read_jsonl(CORPUS))
    gold = list(read_jsonl(GOLD))
    teacher = {(t["id"], t.get("ticker")): t for t in latest_teacher(read_jsonl(TEACHER))}
    tp = [(r["score"], teacher[k]["cls"]) for r in raw
          if (k := (raw_to_item(r).id, r["ticker"])) in teacher and teacher[k].get("cls") is not None]
    pairs = [(r["score"], c) for r, c in match_gold(raw, corpus_rows, gold)]
    print(json.dumps({
        "raw": len(raw),
        "sources": dict(Counter(r.get("source", "") for r in raw).most_common(10)),
        "vs_gold": agreement(pairs, a.t),
        "vs_teacher": agreement(tp, a.t),
        "vs_gold_by_t": {str(t): agreement(pairs, t) for t in (0.05, 0.15, 0.3)},
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
