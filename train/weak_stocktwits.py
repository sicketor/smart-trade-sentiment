"""Weak labels from StockTwits: messages whose authors tagged them Bullish/Bearish.

Resumable backfill (~300 days) plus nightly new messages -> raw jsonl -> filtered, balanced training file;
`stats` also checks how often the authors' tags agree with Tom's gold labels.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone

from sentiment import config
from sentiment.http import Http
from sentiment.sources._util import account_age_days, item_id, parse_iso, strip_html
from sentiment.sources.stocktwits import URL
from sentiment.text import title_key

from .corpus import DEFAULT_UNIVERSE, read_universe
from .io import CORPUS, GOLD, append_jsonl, read_jsonl

WEAK = config.ROOT / "data" / "weak"
RAW = WEAK / "stocktwits_raw.jsonl"
STATE = WEAK / "stocktwits_state.json"
OUT = WEAK / "stocktwits_train.jsonl"

LABELS = {"Bullish": 1, "Bearish": -1}


def message_row(m: dict, ticker: str, now: datetime | None = None) -> dict | None:
    body = strip_html(m.get("body"))
    user = m.get("user") or {}
    username = user.get("username")
    mid = m.get("id")
    dt = parse_iso(m.get("created_at"))
    if not body or not username or mid is None or dt is None:
        return None
    sent = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
    if sent not in LABELS:
        return None
    url = f"https://stocktwits.com/{username}/message/{mid}"
    return {
        "mid": int(mid),
        "id": item_id("stocktwits", url, body),
        "ticker": ticker,
        "body": body,
        "label": sent,
        "user": username,
        "followers": user.get("followers"),
        "account_age_days": account_age_days(user.get("join_date"), now),
        "symbols": [s["symbol"] for s in m.get("symbols") or [] if s.get("symbol")],
        "created_at": dt.isoformat(),
        "likes": (m.get("likes") or {}).get("total"),
    }


def harvest_ticker(ticker, http, state: dict, seen_mids: set, *, days=300, max_pages=150, now=None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    st = state.setdefault(ticker, {"newest": None, "oldest": None, "backfill_done": False})
    st.setdefault("newest", None)
    st.setdefault("oldest", None)
    st.setdefault("backfill_done", False)
    url = URL.format(t=ticker)
    out: list[dict] = []
    pages = 0

    def take(msgs, keep) -> None:
        for m in msgs:
            if not keep(m):
                continue
            r = message_row(m, ticker, now)
            if r is not None and r["mid"] not in seen_mids:
                seen_mids.add(r["mid"])
                out.append(r)

    def mids_of(msgs) -> list[int]:
        return [int(m["id"]) for m in msgs if m.get("id") is not None]

    def note(mids) -> None:
        if not mids:
            return
        st["newest"] = max(mids) if st["newest"] is None else max(st["newest"], max(mids))
        st["oldest"] = min(mids) if st["oldest"] is None else min(st["oldest"], min(mids))

    try:
        # 0. fill a gap left by an interrupted "newest" phase
        gap = st.get("gap")
        if gap:
            lo, hi = int(gap[0]), int(gap[1])
            while hi > lo and pages < max_pages:
                data = http.get_json(url, params={"max": hi})
                pages += 1
                msgs = (data or {}).get("messages") or []
                mids = mids_of(msgs)
                if not mids:
                    hi = lo
                    break
                take(msgs, lambda m: m.get("id") is not None and int(m["id"]) > lo)
                note(mids)
                if min(mids) <= lo:
                    hi = lo
                    break
                hi = min(mids) - 1
                st["gap"] = [lo, hi]
            if hi <= lo:
                st.pop("gap", None)
        # 1. new messages since the last run
        if st["newest"] is not None:
            newest0 = st["newest"]
            params = None
            top = None
            low = None
            complete = False
            err = None
            try:
                while pages < max_pages:
                    data = http.get_json(url, params=params)
                    pages += 1
                    msgs = (data or {}).get("messages") or []
                    mids = mids_of(msgs)
                    if not mids:
                        complete = True
                        break
                    take(msgs, lambda m: m.get("id") is not None and int(m["id"]) > newest0)
                    top = max(mids) if top is None else max(top, max(mids))
                    low = min(mids)
                    st["oldest"] = low if st["oldest"] is None else min(st["oldest"], low)
                    if low <= newest0:
                        complete = True
                        break
                    params = {"max": low - 1}
            except Exception as e:
                err = e
            if top is not None:
                st["newest"] = max(newest0, top)
                if not complete:
                    g = st.get("gap")
                    lo, hi = newest0, low - 1
                    if g:
                        lo, hi = min(lo, int(g[0])), max(hi, int(g[1]))
                    st["gap"] = [lo, hi]
            if err is not None:
                raise err
        # 2. backfill towards the date window
        while not st["backfill_done"] and pages < max_pages:
            params = {"max": st["oldest"] - 1} if st["oldest"] is not None else None
            data = http.get_json(url, params=params)
            pages += 1
            msgs = (data or {}).get("messages") or []
            mids = mids_of(msgs)
            if not mids:
                st["backfill_done"] = True
                break
            dates = [d for d in (parse_iso(m.get("created_at")) for m in msgs) if d]
            take(msgs, lambda m: (parse_iso(m.get("created_at")) or cutoff) >= cutoff)
            note(mids)
            if dates and min(dates) < cutoff:
                st["backfill_done"] = True
    except Exception as e:
        print(f"{ticker}: stopped ({type(e).__name__}: {e})", flush=True)
    return out


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NOISE_RE = re.compile(r"\$[A-Za-z][\w.]*|@\w+")


def _words(body: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", _NOISE_RE.sub(" ", _URL_RE.sub(" ", body or ""))))


def build(raw_rows, gold_rows, *, per_user_cap=20, min_age_days=30, seed=0, balance=True) -> tuple[list[dict], dict]:
    """balance=False keeps the natural bullish/bearish ratio (the balanced set hurt the social student, SPEC §0.9)."""
    raw_rows = list(raw_rows)
    counts: dict = {"raw": len(raw_rows)}
    gold_ids = {g["id"] for g in gold_rows if "id" in g}

    rows = [r for r in raw_rows if r.get("label") in LABELS]
    counts["tagged"] = len(rows)
    rows = [r for r in rows if r["ticker"] in (r.get("symbols") or []) and len(set(r["symbols"])) <= 2]
    counts["symbols"] = len(rows)
    rows = [r for r in rows if _words(r["body"]) >= 4]
    counts["words"] = len(rows)
    rows = [r for r in rows if r.get("account_age_days") is not None and r["account_age_days"] >= min_age_days]
    counts["account_age"] = len(rows)

    gold_keys = {(r["ticker"], title_key(r["body"])) for r in raw_rows if r.get("id") in gold_ids}
    rows = [r for r in rows if r["id"] not in gold_ids and (r["ticker"], title_key(r["body"])) not in gold_keys]
    counts["not_gold"] = len(rows)

    best: dict = {}
    for r in rows:
        k = (r["ticker"], title_key(r["body"]))
        cur = best.get(k)
        if cur is None or (r["created_at"], r["mid"]) < (cur["created_at"], cur["mid"]):
            best[k] = r
    rows = list(best.values())
    counts["dedup"] = len(rows)

    per_user: dict = {}
    for r in sorted(rows, key=lambda r: (r["created_at"], r["mid"])):
        per_user.setdefault((r["ticker"], r["user"]), []).append(r)
    rows = [r for lst in per_user.values() for r in lst[:per_user_cap]]
    counts["user_cap"] = len(rows)

    groups: dict = {}
    for r in rows:
        groups.setdefault((r["ticker"], r["created_at"][:7]), []).append(r)
    rng = random.Random(seed)
    kept = []
    for key in sorted(groups):
        if not balance:
            kept += groups[key]
            continue
        bull = sorted((r for r in groups[key] if r["label"] == "Bullish"), key=lambda r: r["mid"])
        bear = sorted((r for r in groups[key] if r["label"] == "Bearish"), key=lambda r: r["mid"])
        n = min(len(bull), len(bear))
        if n == 0:
            continue
        kept += rng.sample(bull, n) + rng.sample(bear, n)
    kept.sort(key=lambda r: (r["ticker"], r["created_at"], r["mid"]))
    counts["balanced" if balance else "natural"] = len(kept)
    out = [{"id": r["id"], "mid": r["mid"], "ticker": r["ticker"], "text": r["body"],
            "label": LABELS[r["label"]], "created_at": r["created_at"], "user": r["user"]} for r in kept]
    return out, counts


def tag_agreement(corpus_rows, gold_rows) -> dict:
    latest: dict = {}
    for g in gold_rows:
        if "id" in g:
            latest[g["id"]] = g  # later rows win
    cnt: dict = {}
    for r in corpus_rows:
        it = r.get("item") or {}
        tag = (it.get("meta") or {}).get("user_label")
        g = latest.get(it.get("id"))
        if it.get("source") != "stocktwits" or tag not in LABELS or g is None or g.get("skipped"):
            continue
        tom = int(g["cls"]) if g.get("relevant", True) else 0
        sign = (tom > 0) - (tom < 0)
        k = f"{tag}:{sign}"
        cnt[k] = cnt.get(k, 0) + 1
    n = sum(cnt.values())
    nonzero = sum(v for k, v in cnt.items() if not k.endswith(":0"))
    agree = cnt.get("Bullish:1", 0) + cnt.get("Bearish:-1", 0)
    return {"n": n, "counts": cnt,
            "direction_agree": agree / nonzero if nonzero else None,
            "share_tom_zero": (n - nonzero) / n if n else None}


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)


def _build_counts(balance=True):
    rows, counts = build(read_jsonl(RAW), list(read_jsonl(GOLD)), balance=balance)
    return rows, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train.weak_stocktwits")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    h.add_argument("--tickers", default="")
    h.add_argument("--days", type=int, default=300)
    h.add_argument("--max-pages", type=int, default=150)
    b = sub.add_parser("build")
    b.add_argument("--natural", action="store_true", help="keep the natural bullish/bearish ratio (no balancing)")
    b.add_argument("--out", default=str(OUT))
    sub.add_parser("stats")
    a = ap.parse_args(argv)

    if a.cmd == "harvest":
        WEAK.mkdir(parents=True, exist_ok=True)
        tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()] or read_universe(a.universe)
        seen = {r["mid"] for r in read_jsonl(RAW) if "mid" in r}
        state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
        http, total = Http(), 0
        for t in tickers:
            rows = harvest_ticker(t, http, state, seen, days=a.days, max_pages=a.max_pages)
            append_jsonl(RAW, rows)
            _save_state(state)
            total += len(rows)
            s = state[t]
            print(f"{t}: +{len(rows)} newest={s['newest']} oldest={s['oldest']} done={s['backfill_done']}", flush=True)
        print(f"total new rows: {total}")
    elif a.cmd == "build":
        rows, counts = _build_counts(balance=not a.natural)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(json.dumps(counts))
        print(f"wrote {len(rows)} rows -> {out}")
    else:
        _, counts = _build_counts()
        print(json.dumps(counts))
        print(json.dumps(tag_agreement(read_jsonl(CORPUS), read_jsonl(GOLD))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
