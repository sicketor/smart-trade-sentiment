"""Label corpus items with Gemini (the teacher), resumable and quota-aware (SPEC §5.0).

python -m train.teacher [--max-items 5000] [--min-interval 4.5] [--channels news,filings,social] [--relabel-old]

Rows carry the prompt version `pv`; readers keep the highest one (train.io.latest_teacher). With --relabel-old,
items labeled only by an older prompt are queued again after the never-labeled ones.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sentiment import config

from .io import CORPUS, TEACHER, append_jsonl, company_from_json, item_from_json, latest_teacher, read_jsonl


def pending(corpus_rows, labeled_ids, channels=("news", "filings", "social"),
            only: set[str] | None = None) -> dict[str, list[dict]]:
    """Unlabeled rows (restricted to `only` ids if given) grouped by ticker, first row per item id,
    newest first within a ticker."""
    channels, labeled = set(channels), set(labeled_ids)
    out: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for row in corpus_rows:
        it = row["item"]
        if (it["id"] in labeled or it["id"] in seen or it.get("channel") not in channels
                or (only is not None and it["id"] not in only)):
            continue
        seen.add(it["id"])
        out.setdefault(it["ticker"], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: r["item"]["published_at"], reverse=True)
    return out


def _chunks(by_ticker: dict[str, list[dict]], size: int):
    """Round-robin: one chunk of <= size rows from each ticker per round."""
    queues = {t: list(rows) for t, rows in by_ticker.items()}
    while any(queues.values()):
        for t in list(queues):
            if queues[t]:
                chunk, queues[t] = queues[t][:size], queues[t][size:]
                yield t, chunk


def run(labeler, corpus_rows, labeled_ids, max_items=5000, min_interval=4.5,
        channels=("news", "filings", "social"), out_path=TEACHER, only: set[str] | None = None) -> Counter:
    dist: Counter = Counter()
    by_ticker = pending(corpus_rows, labeled_ids, channels, only)
    sent = batches = written = 0
    last = None
    for ticker, chunk in _chunks(by_ticker, config.GEMINI_BATCH):
        if labeler.exhausted:
            break
        chunk = chunk[: max_items - sent]
        if not chunk:
            break
        if last is not None:
            wait = min_interval - (time.monotonic() - last)
            time.sleep(max(0.0, wait))
        last = time.monotonic()
        company = company_from_json(chunk[0]["company"])
        try:
            answers = labeler.label(company, [item_from_json(r["item"]) for r in chunk])
        except Exception as e:
            print(f"{ticker}: label error {type(e).__name__}", flush=True)
            answers = {}
        sent += len(chunk)
        batches += 1
        ts = datetime.now(timezone.utc).isoformat()
        rows = [{"id": i, "ticker": ticker, "cls": a["cls"], "stance": a["stance"], "event": a["event"],
                 "relevant": a["relevant"], "reason": a.get("reason", ""), "ts": ts,
                 "pv": getattr(labeler, "prompt_version", 1), **({"about": a["about"]} if "about" in a else {})}
                for i, a in answers.items()]
        append_jsonl(out_path, rows)
        written += len(rows)
        dist.update(r["cls"] for r in rows)
        if batches % 10 == 0:
            print(f"batches={batches} sent={sent} labeled={written}", flush=True)
        if sent >= max_items:
            break
    if labeler.exhausted:
        print("quota exhausted — resume tomorrow")
    print(f"done: batches={batches} sent={sent} labeled={written}")
    print("class distribution:", {k: dist[k] for k in sorted(dist)})
    return dist


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train.teacher")
    ap.add_argument("--max-items", type=int, default=5000)
    ap.add_argument("--min-interval", type=float, default=4.5)
    ap.add_argument("--channels", default="news,filings,social")
    ap.add_argument("--relabel-old", action="store_true")
    a = ap.parse_args(argv)
    from sentiment.llm import GeminiLabeler
    labeler = GeminiLabeler()
    if not labeler.available:
        print("GEMINI_API_KEY missing")
        return 2
    # one run at a time: two runs label the same pending items twice and burn the free quota
    lock = Path(TEACHER).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"another teacher run is active (delete {lock} if it crashed)")
        return 3
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        rows = latest_teacher(read_jsonl(TEACHER))
        labeled = {r["id"] for r in rows}
        channels = tuple(c.strip() for c in a.channels.split(",") if c.strip())
        dist = run(labeler, read_jsonl(CORPUS), labeled, a.max_items, a.min_interval, channels)
        left = a.max_items - sum(dist.values())
        if a.relabel_old and left > 0 and not labeler.exhausted:
            current = {r["id"] for r in rows if r.get("pv", 1) >= labeler.prompt_version}
            old = labeled - current
            print(f"relabeling up to {left} of {len(old)} items from older prompts")
            run(labeler, read_jsonl(CORPUS), set(), left, a.min_interval, channels, only=old)
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
