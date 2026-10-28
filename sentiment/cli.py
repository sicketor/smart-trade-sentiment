"""python -m sentiment analyze NVDA AMD [--tickers-file f.txt] [--shard i/n] [--no-llm] [--days 7]
[--sources a,b] [--items] [--out f.json]"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

from . import config


def read_tickers_file(path: str) -> list[str]:
    """One ticker per line; blank lines and # comments (whole-line or trailing) are skipped."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.split("#", 1)[0].strip()
            if t:
                out.append(t)
    return out


def parse_shard(spec: str) -> tuple[int, int]:
    """'i/n' (1-based) -> (i, n); ValueError when malformed or out of range."""
    i, sep, n = spec.partition("/")
    if not sep:
        raise ValueError(f"shard must look like i/n, got {spec!r}")
    i_, n_ = int(i), int(n)
    if n_ < 1 or not 1 <= i_ <= n_:
        raise ValueError(f"shard needs 1 <= i <= n, got {spec!r}")
    return i_, n_


def select_shard(tickers: list[str], i: int, n: int) -> list[str]:
    """Tickers at sorted-unique positions idx where idx % n == i - 1."""
    return [t for idx, t in enumerate(sorted(set(tickers))) if idx % n == i - 1]


def _unique(tickers: list[str]) -> list[str]:
    return list(dict.fromkeys(tickers))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sentiment")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze US tickers (up to 5 positional, any number via --tickers-file)")
    a.add_argument("tickers", nargs="*")
    a.add_argument("--tickers-file", help="one ticker per line; # comments and blank lines skipped")
    a.add_argument("--shard", help="i/n (1-based): keep sorted-unique tickers at idx %% n == i-1")
    a.add_argument("--no-llm", action="store_true", help="skip Gemini even when GEMINI_API_KEY is set")
    a.add_argument("--days", type=int, default=config.LOOKBACK_DAYS)
    a.add_argument("--sources", help="comma-separated adapter ids (default: all)")
    a.add_argument("--items", action="store_true", help="include every scored item")
    a.add_argument("--device", default="auto")
    a.add_argument("--no-save", action="store_true", help="do not append results to data/snapshots/")
    a.add_argument("--out", help="write JSON here instead of stdout")
    args = p.parse_args(argv)

    try:
        shard = parse_shard(args.shard) if args.shard else None
        raw = list(args.tickers) + (read_tickers_file(args.tickers_file) if args.tickers_file else [])
    except (ValueError, OSError) as e:
        a.error(str(e))
    tickers = _unique([t.strip().upper() for t in raw if t.strip()])
    if not tickers:
        a.error("at least one ticker is required (positional or --tickers-file)")
    if shard is not None:
        tickers = select_shard(tickers, *shard)
    elif not args.tickers_file:
        tickers = tickers[:5]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if not tickers:  # an empty shard (n > universe) is not a failure
        logging.warning("shard %s selects no tickers", args.shard)
    from .pipeline import Engine, snapshot_to_dict

    from .snapshots import history, record

    engine = Engine(use_llm=not args.no_llm, device=args.device) if tickers else None
    results, errors = [], {}
    for t in tickers:
        t0 = time.perf_counter()
        now_utc = datetime.now(timezone.utc)
        try:
            snap, items = engine.analyze(t, days=args.days,
                                         only=args.sources.split(",") if args.sources else None,
                                         history=history(t, now_utc), as_of=now_utc)
        except Exception as e:
            logging.exception("analyze %s failed", t)
            errors[t] = f"{type(e).__name__}: {e}"
            continue
        if not args.no_save:
            try:
                record(snapshot_to_dict(snap, None))
            except Exception:
                logging.exception("saving snapshot for %s failed", t)
        d = snapshot_to_dict(snap, items if args.items else None)
        d["elapsed_s"] = round(time.perf_counter() - t0, 1)
        results.append(d)

    out = json.dumps({"results": results, "errors": errors}, ensure_ascii=False, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out + "\n")
    return 0 if results or not tickers else 1
