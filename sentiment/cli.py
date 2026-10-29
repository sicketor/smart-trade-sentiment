"""python -m sentiment analyze NVDA AMD [--tickers-file f.txt] [--shard i/n] [--no-llm] [--days 7]
[--sources a,b] [--items] [--out f.json]"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


def _store(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)   # its request lines print the project URL to public Actions logs
    from .store import Store, canonical, parse_requests

    reqs = []
    if args.requests:
        try:
            reqs = parse_requests(args.requests)   # validated before touching the DB
        except ValueError as e:
            logging.error("invalid --requests: %s", e)
            return 1
    finalize = bool(reqs) and not args.mark
    store = Store.from_env()
    if store is None:
        return 0
    failed = False
    ids = [i for _, i in reqs]
    try:
        if args.mark:
            try:
                store.mark_requests(ids, args.mark)
            except Exception as e:
                logging.error("%s: %s", type(e).__name__, e)
                failed = True
        if args.pull_history:
            try:
                logging.info("pulled %d history rows",
                             store.pull_history([t for t, _ in reqs], Path(args.to)))
            except Exception as e:
                logging.error("%s: %s", type(e).__name__, e)
                failed = True
        if args.history:
            try:
                logging.info("history: %d rows", store.import_history(Path(args.history)))
            except Exception as e:   # any failure: log it and still run the remaining steps
                logging.error("%s: %s", type(e).__name__, e)
                failed = True
        results, errors = [], {}
        for path in args.results:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                results.extend(data["results"])
                if isinstance(data.get("errors"), dict):
                    errors.update({canonical(str(k)): str(v) for k, v in data["errors"].items()})
            except FileNotFoundError:
                logging.warning("results file %s not found; skipping", path)
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as e:
                logging.error("cannot read %s: %s", path, e)
                failed = True
        write_failed = False
        if args.results:
            try:
                r = store.write_results(results, args.run)
                logging.info("stored %d snapshots, %d latest updated", r["snapshots"], r["latest"])
            except Exception as e:   # any failure: log it and still run the remaining steps
                logging.error("%s: %s", type(e).__name__, e)
                failed = write_failed = True
        if finalize:
            try:
                if write_failed:
                    store.mark_requests(ids, "failed", "store failed")
                else:
                    have = {canonical(str(d["ticker"])) for d in results
                            if isinstance(d, dict) and d.get("ticker") and d.get("as_of")}
                    done = [i for t, i in reqs if t in have]
                    if done:
                        store.mark_requests(done, "done")
                    for t, i in reqs:
                        if t not in have:
                            store.mark_requests([i], "failed", errors.get(t) or "no result")
            except Exception as e:
                logging.error("%s: %s", type(e).__name__, e)
                failed = True
        if args.universe:
            try:
                logging.info("universe: %d tickers", store.set_universe(read_tickers_file(args.universe)))
            except Exception as e:
                logging.error("%s: %s", type(e).__name__, e)
                failed = True
    finally:
        store.close()
    return 1 if failed else 0


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
    a.add_argument("--run", default="manual", choices=("main", "refresh", "manual", "ondemand"),
                   help="run kind recorded in the snapshot (main = the pre-open daily run)")
    a.add_argument("--out", help="write JSON here instead of stdout")
    s = sub.add_parser("store", help="write results / universe / history to the HUB Supabase DB")
    s.add_argument("--results", nargs="+", action="extend", default=[], help="analyze --out JSON files")
    s.add_argument("--run", default="main", choices=("main", "refresh", "manual", "ondemand", "list"))
    s.add_argument("--universe", help="tickers file that replaces the pre-analyzed universe")
    s.add_argument("--history", help="directory of <TICKER>.jsonl snapshot files to import")
    s.add_argument("--requests", help="on-demand requests: TICKER=request_id,... (up to 5)")
    s.add_argument("--mark", choices=("running",), help="mark the --requests as running")
    s.add_argument("--pull-history", action="store_true", help="download the --requests tickers' history")
    s.add_argument("--to", default="data/snapshots", help="directory for --pull-history")
    args = p.parse_args(argv)
    if args.cmd == "store":
        if (args.mark or args.pull_history) and not args.requests:
            s.error("--mark / --pull-history require --requests")
        if not (args.results or args.universe or args.history or args.requests):
            s.error("at least one of --results / --universe / --history / --requests is required")
        return _store(args)

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
                record({**snapshot_to_dict(snap, None), "run": args.run})
            except Exception:
                logging.exception("saving snapshot for %s failed", t)
        d = snapshot_to_dict(snap, items if args.items else None)
        d["elapsed_s"] = round(time.perf_counter() - t0, 1)
        d["timings_s"] = engine.last_timings
        logging.info("%s timings %s items=%d", t, engine.last_timings, len(items))
        results.append(d)

    out = json.dumps({"results": results, "errors": errors}, ensure_ascii=False, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out + "\n")
    return 0 if results or not tickers else 1
