"""Point-in-time market validation pilot (SPEC §9.3).

python -m eval.market [--refresh] [--n-boot 2000] [--out runs/market_validation.json] [--report docs/market-validation.md]

Replays each ticker's daily score at the US close (20:00 UTC) of every trading day from the cached corpus
backfill, using only items published up to that moment, and tests the level (z) and its daily change (dz)
against next-day / 3-day excess returns vs SPY (plus the same-day return as a "does sentiment follow price"
diagnostic). Rank IC with day- and ticker-cluster bootstrap CIs, and a per-grade return table.
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
import zlib
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from sentiment import config

log = logging.getLogger(__name__)

ROOT = config.ROOT
CACHE_DIR = ROOT / "data" / "market"
BENCH = "SPY"
GROUPS = ("news", "filings", "social", "composite")
EVAL_GROUPS = ("news", "social", "composite")
HORIZONS = (-1, 1, 3)
SIGNALS = ("z", "dz")
MIN_TICKERS_PER_DAY = 30
MIN_ROWS_PER_DAY_IC = 10
MIN_FILINGS_N = 30


# --- prices ----------------------------------------------------------------------------------

def yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def _fetch(symbol: str, http) -> dict[str, float]:
    from sentiment.sources.context import _chart
    return {r["date"]: float(r["close"]) for r in _chart(yahoo_symbol(symbol), http)}


def load_prices(symbols, http=None, cache_dir: Path = CACHE_DIR, refresh: bool = False) -> dict[str, dict[str, float]]:
    """{symbol: {YYYY-MM-DD: close}}, one cached JSON per symbol. Bars dated on or after the fetch day (UTC)
    are not cached: they may be partial. A cache is stale (refetched) when it was fetched before today (UTC)
    or its last bar is older than SPY's. A failed or empty fetch logs a warning and falls back to a stale
    cache when there is one, otherwise skips the symbol."""
    cache_dir = Path(cache_dir)
    want = list(dict.fromkeys([BENCH, *symbols]))   # SPY first: its last bar is the freshness reference
    today = datetime.now(timezone.utc).date().isoformat()
    out: dict[str, dict[str, float]] = {}
    for sym in want:
        path = cache_dir / f"{yahoo_symbol(sym)}.json"
        cached: dict[str, float] | None = None
        if path.is_file():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                cached = {d: float(c) for d, c in doc["closes"].items()}
                fetched = str(doc.get("fetched_at") or "")[:10]
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                log.warning("price cache unreadable for %s (%s); refetching", sym, e)
                cached, fetched = None, ""
            spy_last = max(out[BENCH]) if BENCH in out and out[BENCH] else ""
            fresh = (cached and fetched >= today and (sym == BENCH or max(cached) >= spy_last))
            if fresh and not refresh:
                out[sym] = cached
                continue
        if http is None:
            from sentiment.http import Http
            http = Http()
        try:
            closes = _fetch(sym, http)
        except Exception as e:   # one bad symbol must not stop the run
            closes = {}
            log.warning("price fetch failed for %s: %s", sym, e)
        closes = {d: c for d, c in closes.items() if d < today}
        if not closes:
            if cached:
                log.warning("no fresh price bars for %s; using the stale cache", sym)
                out[sym] = cached
            else:
                log.warning("no price bars for %s", sym)
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"symbol": sym, "fetched_at": datetime.now(timezone.utc).isoformat(),
                                    "closes": dict(sorted(closes.items()))}), encoding="utf-8")
        out[sym] = closes
    if BENCH not in out:
        raise RuntimeError(f"{BENCH} prices are required but could not be loaded")
    return out


def trading_days(spy: dict[str, float], today: date) -> list[str]:
    """Sorted SPY bar dates strictly before today (today's bar may be partial)."""
    t = today.isoformat()
    return sorted(d for d in spy if d < t)


def as_of_for(day: str) -> datetime:
    """The US close of day in UTC (16:00 EDT; every pilot date is before the 1 Nov 2026 DST switch)."""
    return datetime.combine(date.fromisoformat(day), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=20)


def excess_returns(prices: dict[str, float], spy: dict[str, float], day: str, days_list: list[str],
                   h: int) -> float | None:
    """(C[D+h]/C[D] - 1) - (SPY[D+h]/SPY[D] - 1) over h trading days of days_list; h = -1 is the same-day
    return C[D-1] -> C[D] minus SPY's. None when a day falls outside days_list or a close is missing."""
    if h == 0:
        raise ValueError("h must be non-zero")
    i = bisect.bisect_left(days_list, day)
    if i >= len(days_list) or days_list[i] != day:
        return None
    if h > 0:
        j = i + h
        if j >= len(days_list):
            return None
        a, b = day, days_list[j]
    else:
        j = i + h
        if j < 0:
            return None
        a, b = days_list[j], day
    if any(x.get(d) in (None, 0) for x in (prices, spy) for d in (a, b)):
        return None
    return (prices[b] / prices[a] - 1.0) - (spy[b] / spy[a] - 1.0)


# --- point-in-time replay ----------------------------------------------------------------------

def pit_records(scored_by_ticker: dict, days: list[str], norm: dict | None) -> list[dict]:
    """One record per (ticker, trading day) with at least one item published in
    [as_of - LOOKBACK_DAYS, as_of]. collected_at is ignored on purpose (the corpus was collected on 26/10 and
    28/10, so a collected_at filter would empty every earlier day); published_at alone bounds look-ahead."""
    from sentiment.aggregate import summarize
    lookback = timedelta(days=config.LOOKBACK_DAYS)
    records = []
    for ticker in sorted(scored_by_ticker):
        entries = sorted(scored_by_ticker[ticker], key=lambda e: e[1])
        pubs = [e[1] for e in entries]
        for day in days:
            as_of = as_of_for(day)
            lo, hi = bisect.bisect_left(pubs, as_of - lookback), bisect.bisect_right(pubs, as_of)
            items = [e[0] for e in entries[lo:hi]]
            if not items:
                continue
            s = summarize(items, as_of, norm=norm)
            rec = {"ticker": ticker, "day": day, "n_items": len(items)}
            for g in GROUPS:
                snap = s["composite"] if g == "composite" else s["channels"][g]
                rec[g] = {"raw": snap.raw, "z": snap.z, "grade": snap.grade, "score": snap.score}
            records.append(rec)
    return records


# --- statistics --------------------------------------------------------------------------------

def spearman(x, y) -> float | None:
    """Spearman rho (average ranks); None for fewer than 3 rows or a constant side."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if den <= 0:
        return None
    return float((rx * ry).sum() / den)


def cluster_ci(x: np.ndarray, y: np.ndarray, clusters: np.ndarray, n_boot: int,
               rng: np.random.Generator) -> tuple[list[float] | None, int]:
    """(95% percentile CI, number of valid draws) of the pooled rho when whole clusters are resampled with
    replacement; draws with a constant side are dropped."""
    keys = np.unique(clusters)
    if len(keys) < 2:
        return None, 0
    idx_of = [np.flatnonzero(clusters == k) for k in keys]
    stats = []
    for draw in rng.integers(0, len(keys), (n_boot, len(keys))):
        idx = np.concatenate([idx_of[k] for k in draw])
        r = spearman(x[idx], y[idx])
        if r is not None:
            stats.append(r)
    if not stats:
        return None, 0
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)], len(stats)


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def grade_table(grades: list[int], ex: list[float]) -> dict:
    """Per-grade n / mean / median excess return, hit rate on grade != 0, strong (|g|=3) vs weak (|g|=1)
    mean of ex·sign(g), and the base rate (share ex > 0)."""
    by: dict[int, list[float]] = defaultdict(list)
    for g, e in zip(grades, ex):
        by[int(g)].append(e)
    rows = {}
    for g in range(-3, 4):
        v = by.get(g, [])
        rows[str(g)] = {"n": len(v), "mean": float(np.mean(v)) if v else None,
                        "median": float(np.median(v)) if v else None}
    nz = [(g, e) for g, e in zip(grades, ex) if g != 0]
    decided = [(g, e) for g, e in nz if e != 0]     # ex == 0 is neither a hit nor a miss
    hit = sum(_sign(e) == _sign(g) for g, e in decided) / len(decided) if decided else None
    strong = [e * _sign(g) for g, e in nz if abs(g) == 3]
    weak = [e * _sign(g) for g, e in nz if abs(g) == 1]
    return {"grades": rows, "n_nonzero": len(nz), "n_hit": len(decided), "hit_rate": hit,
            "strong_vs_weak": {"strong": float(np.mean(strong)) if strong else None, "n_strong": len(strong),
                               "weak": float(np.mean(weak)) if weak else None, "n_weak": len(weak)},
            "base_rate": sum(e > 0 for e in ex) / len(ex) if ex else None}


def with_dz(records: list[dict], days_list: list[str], field: str = "z") -> list[dict]:
    """Copies of the records with rec[g]["dz"] = field(D) - field(previous trading day) for the same ticker;
    None when either side is missing (no record that day or a None value)."""
    pos = {d: i for i, d in enumerate(days_list)}
    by_key = {(r["ticker"], r["day"]): r for r in records}
    out = []
    for r in records:
        i = pos.get(r["day"])
        prev = by_key.get((r["ticker"], days_list[i - 1])) if i is not None and i > 0 else None
        nr = dict(r)
        for g in GROUPS:
            cur = (r.get(g) or {}).get(field)
            old = (prev.get(g) or {}).get(field) if prev else None
            nr[g] = {**(r.get(g) or {}), "dz": None if cur is None or old is None else cur - old}
        out.append(nr)
    return out


def pilot_days(records: list[dict], min_tickers: int = MIN_TICKERS_PER_DAY) -> list[str]:
    count: dict[str, set] = defaultdict(set)
    for r in records:
        count[r["day"]].add(r["ticker"])
    return sorted(d for d, t in count.items() if len(t) >= min_tickers)


def evaluate(records: list[dict], prices: dict[str, dict[str, float]], spy: dict[str, float],
             days_list: list[str], n_boot: int = 2000, seed: int = 0, field: str = "z",
             min_tickers: int = MIN_TICKERS_PER_DAY) -> dict:
    """IC / grade tables over the pilot days (>= min_tickers records). field is the level signal ("z", or "raw"
    when no norm is loaded); "dz" is its change vs the previous trading day."""
    days = pilot_days(records, min_tickers)
    keep = set(days)
    recs = [r for r in with_dz(records, days_list, field) if r["day"] in keep]
    ex_cache: dict[tuple, float | None] = {}

    def ex_of(r: dict, h: int) -> float | None:
        k = (r["ticker"], r["day"], h)
        if k not in ex_cache:
            p = prices.get(r["ticker"])
            ex_cache[k] = None if p is None else excess_returns(p, spy, r["day"], days_list, h)
        return ex_cache[k]

    summary, grades = [], {}
    for g in (*EVAL_GROUPS, "filings"):
        for sig in SIGNALS:
            key = field if sig == "z" else "dz"
            for h in HORIZONS:
                rows = [(r[g][key], ex_of(r, h), r["day"], r["ticker"]) for r in recs]
                rows = [t for t in rows if t[0] is not None and t[1] is not None]
                if g == "filings" and len(rows) < MIN_FILINGS_N:
                    continue
                entry = {"group": g, "signal": sig, "h": h, "n": len(rows),
                         "n_days": len({t[2] for t in rows}), "n_tickers": len({t[3] for t in rows})}
                if rows:
                    x = np.array([t[0] for t in rows], dtype=float)
                    y = np.array([t[1] for t in rows], dtype=float)
                    dcol = np.array([t[2] for t in rows])
                    tcol = np.array([t[3] for t in rows])
                    rho = spearman(x, y)
                    ics = []
                    for d in sorted(set(dcol)):
                        m = dcol == d
                        if m.sum() >= MIN_ROWS_PER_DAY_IC:
                            ics.append({"day": d, "n": int(m.sum()), "ic": spearman(x[m], y[m])})
                    valid = [c["ic"] for c in ics if c["ic"] is not None]
                    # per-entry stream: an entry's CI does not move when other entries are added or removed
                    rng = np.random.default_rng([seed, zlib.crc32(f"{g}|{sig}|{h}".encode())])
                    ci_d, nv_d = cluster_ci(x, y, dcol, n_boot, rng)
                    ci_t, nv_t = cluster_ci(x, y, tcol, n_boot, rng)
                    entry.update({
                        "rho": None if rho is None else round(rho, 4),
                        "ci_days": ci_d, "ci_days_valid": nv_d,
                        "ci_tickers": ci_t, "ci_tickers_valid": nv_t,
                        "daily_ic": [{**c, "ic": None if c["ic"] is None else round(c["ic"], 4)} for c in ics],
                        "mean_ic": round(float(np.mean(valid)), 4) if valid else None,
                        "base_rate": round(float((y > 0).mean()), 4)})
                summary.append(entry)
                if sig == "z":
                    gr = [(r[g]["grade"], ex_of(r, h)) for r in recs]
                    gr = [(a, b) for a, b in gr if a is not None and b is not None]
                    if gr:
                        grades[f"{g}|{h}"] = grade_table([a for a, _ in gr], [b for _, b in gr])
    return {"pilot_days": days, "n_ticker_days": len(recs), "n_tickers": len({r["ticker"] for r in recs}),
            "signal_field": field, "summary": summary, "grades": grades}


# --- report ------------------------------------------------------------------------------------

def _f(v, nd=3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _ci(c) -> str:
    return "-" if not c else f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:+.2f}%"


def summary_lines(res: dict) -> list[str]:
    lines = [f"{'group':<10}{'sig':<4}{'h':>3}{'n':>6}{'rho':>8}  {'CI days':<18}  {'CI tickers':<18}{'mean_ic':>8}"]
    for e in res["summary"]:
        lines.append(f"{e['group']:<10}{e['signal']:<4}{e['h']:>3}{e['n']:>6}{_f(e.get('rho')):>8}  "
                     f"{_ci(e.get('ci_days')):<18}  {_ci(e.get('ci_tickers')):<18}{_f(e.get('mean_ic')):>8}")
    return lines


def markdown(res: dict) -> str:
    days = res["pilot_days"]
    rng = f"{days[0]} .. {days[-1]}" if days else "none"
    out = ["# Market validation pilot (SPEC §9.3)", "",
           f"{res['n_ticker_days']} ticker-days, {res['n_tickers']} tickers, days {rng} ({len(days)}). "
           "pilot: one market week, not a calibration.", ""]
    if res["signal_field"] != "z":
        out += ["Note: no aggregate norm loaded, so z is None; the level signal is the raw score.", ""]
    out += ["Excess return vs SPY; h = -1 is the same-day return (diagnostic). "
            "CIs: 95% cluster bootstrap over days / tickers (valid draws in brackets).", "",
            "| group | signal | h | n | days | rho | CI days | CI tickers | mean IC | base rate |",
            "|---|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for e in res["summary"]:
        out.append(f"| {e['group']} | {e['signal']} | {e['h']} | {e['n']} | {e['n_days']} | {_f(e.get('rho'))} | "
                   f"{_ci(e.get('ci_days'))} ({e.get('ci_days_valid', 0)}) | "
                   f"{_ci(e.get('ci_tickers'))} ({e.get('ci_tickers_valid', 0)}) | {_f(e.get('mean_ic'))} | "
                   f"{_f(e.get('base_rate'), 2)} |")
    for key, t in res["grades"].items():
        g, h = key.split("|")
        sw = t["strong_vs_weak"]
        out += ["", f"## Grades: {g}, h = {h}", "",
                f"hit rate {_f(t['hit_rate'], 3)} (n {t['n_hit']}), base rate {_f(t['base_rate'], 3)}, "
                f"strong |g|=3 {_pct(sw['strong'])} (n {sw['n_strong']}) vs weak |g|=1 {_pct(sw['weak'])} "
                f"(n {sw['n_weak']})", "",
                "| grade | n | mean | median |", "|---:|---:|---:|---:|"]
        for gr, v in t["grades"].items():
            out.append(f"| {int(gr):+d} | {v['n']} | {_pct(v['mean'])} | {_pct(v['median'])} |")
    h3 = max((e["n_days"] for e in res["summary"] if e["h"] == 3), default=0)
    out += ["", "## Caveats", "",
            "| caveat | effect |", "|---|---|",
            "| The norm (mu, sd) and the grade cuts were fitted on the same corpus window | "
            "rho is unaffected (a fixed monotone map per group); the grade cut points are in-sample |",
            "| Feeds keep only their latest items, so older days are thinner (survivorship) | "
            "early days carry fewer, possibly non-random items |",
            f"| The day CI rests on only {len(days)} day-clusters | indicative only |",
            f"| h = 3 has only {h3} start days, with overlapping windows | h = 3 results are anecdotal |"]
    return "\n".join(out) + "\n"


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval.market")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/market_validation.json")
    ap.add_argument("--report", default="docs/market-validation.md")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    from sentiment.aggregate import load_norm
    from train.backfill_norm import scored_by_ticker
    scored = scored_by_ticker()
    if not scored:
        print("no scored items; run python -m train.backfill_norm score first")
        return 1
    norm = load_norm()
    prices = load_prices(sorted(scored), refresh=args.refresh)
    spy = prices[BENCH]
    days_list = trading_days(spy, datetime.now(timezone.utc).date())
    first_pub = min(pub for v in scored.values() for _, pub, _ in v).date().isoformat()
    replay = [d for d in days_list if d >= first_pub]
    records = pit_records(scored, replay, norm)
    res = evaluate(records, prices, spy, days_list, n_boot=args.n_boot, seed=args.seed,
                   field="z" if norm is not None else "raw")
    res.update({"built_at": datetime.now(timezone.utc).isoformat(), "n_boot": args.n_boot, "seed": args.seed,
                "norm": None if norm is None else norm.get("source"),
                "missing_prices": sorted(t for t in scored if t not in prices)})
    out, rep = _path(args.out), _path(args.report)
    for p in (out, rep):
        p.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    rep.write_text(markdown(res), encoding="utf-8")
    d = res["pilot_days"]
    print(f"{res['n_ticker_days']} ticker-days, {res['n_tickers']} tickers, {len(d)} days "
          f"{d[0] if d else '-'}..{d[-1] if d else '-'}, missing prices {len(res['missing_prices'])}")
    print("\n".join(summary_lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
