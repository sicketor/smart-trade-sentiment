"""python -m sentiment analyze NVDA AMD [--no-llm] [--days 7] [--sources a,b] [--items] [--out f.json]"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from . import config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sentiment")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze up to 5 US tickers, print JSON")
    a.add_argument("tickers", nargs="+")
    a.add_argument("--no-llm", action="store_true", help="skip Gemini even when GEMINI_API_KEY is set")
    a.add_argument("--days", type=int, default=config.LOOKBACK_DAYS)
    a.add_argument("--sources", help="comma-separated adapter ids (default: all)")
    a.add_argument("--items", action="store_true", help="include every scored item")
    a.add_argument("--device", default="auto")
    a.add_argument("--out", help="write JSON here instead of stdout")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    tickers = [t.upper() for t in args.tickers][:5]
    from .pipeline import Engine, snapshot_to_dict

    engine = Engine(use_llm=not args.no_llm, device=args.device)
    results, errors = [], {}
    for t in tickers:
        t0 = time.perf_counter()
        try:
            snap, items = engine.analyze(t, days=args.days,
                                         only=args.sources.split(",") if args.sources else None)
        except Exception as e:
            logging.exception("analyze %s failed", t)
            errors[t] = f"{type(e).__name__}: {e}"
            continue
        d = snapshot_to_dict(snap, items if args.items else None)
        d["elapsed_s"] = round(time.perf_counter() - t0, 1)
        results.append(d)

    out = json.dumps({"results": results, "errors": errors}, ensure_ascii=False, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out + "\n")
    return 0 if results else 1
