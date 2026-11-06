"""Build the stock-grade calibration batch (SPEC §9.3, route 1).

Picks tickers spread across the engine's composite grades in the latest daily snapshots (git branch `data`),
re-runs the engine on each one now, and saves the items it weighs most, so Tom can grade each stock blind
(-3..+3) in `python -m labeling.stocks`. Compare afterwards with `python -m eval.stock_grades`.

    python -m train.stock_batch [--n 45] [--seed 7] [--ref origin/data] [--out data/gold/stock_batch.json] [--no-llm]
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from sentiment import config

OUT = config.ROOT / "data" / "gold" / "stock_batch.json"
GRADES = (-3, -2, -1, 0, 1, 2, 3)
QUOTA = {-3: 5, -2: 7, -1: 7, 0: 7, 1: 7, 2: 7, 3: 5}   # per 45; scaled to --n
SHOW = {"news": 12, "filings": 4, "social": 10}        # items shown per channel, highest engine weight first
TEXT_MAX = 1500
log = logging.getLogger("stock_batch")


def extract_snapshots(ref: str, dest: Path) -> Path:
    """`snapshots/` of a git ref into dest (one `git archive` call); returns dest/snapshots."""
    tar = subprocess.run(["git", "archive", "--format=tar", ref, "snapshots"], cwd=config.ROOT,
                         capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(dest, filter="data")
    return dest / "snapshots"


def latest_grades(snap_dir: Path) -> dict[str, dict]:
    """ticker -> last snapshot row that has a composite grade."""
    out = {}
    for p in sorted(snap_dir.glob("*.jsonl")):
        last = None
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (row.get("composite") or {}).get("grade") is not None:
                last = row
        if last:
            out[p.stem] = last
    return out


def pick(latest: dict[str, dict], n: int, rng: random.Random) -> list[str]:
    """Stratified by snapshot grade (QUOTA scaled to n); a short grade's places go to the rest at random."""
    by_g: dict[int, list[str]] = {g: [] for g in GRADES}
    for t, row in sorted(latest.items()):
        by_g[int(row["composite"]["grade"])].append(t)
    chosen: list[str] = []
    for g in GRADES:
        k = round(n * QUOTA[g] / sum(QUOTA.values()))
        chosen += rng.sample(by_g[g], min(k, len(by_g[g])))
    rest = [t for t in sorted(latest) if t not in chosen]
    chosen += rng.sample(rest, max(0, min(n - len(chosen), len(rest))))
    return chosen[:n]


def shown_items(items, as_of: datetime) -> list[dict]:
    """The items the engine weighs most, per channel; only display fields (no model label or score)."""
    from sentiment.aggregate import item_weight
    out = []
    for ch, cap in SHOW.items():
        ws = [(item_weight(si, as_of), si) for si in items if si.item.channel == ch]
        ws = sorted([x for x in ws if x[0] > 0], key=lambda x: -x[0])[:cap]
        for w, si in sorted(ws, key=lambda x: x[1].item.published_at, reverse=True):
            it = si.item
            out.append({"id": it.id, "channel": ch, "source": it.source, "publisher": it.publisher,
                        "title": it.title, "text": (it.text or "")[:TEXT_MAX], "url": it.url,
                        "published_at": it.published_at.isoformat(),
                        "age_h": round((as_of - it.published_at).total_seconds() / 3600, 1)})
    return out


def _ch(c) -> dict:
    return {"grade": c.grade, "score": c.score, "z": c.z, "n_items": c.n_items}


def build(n: int, seed: int, ref: str, out: Path, use_llm: bool) -> int:
    from sentiment.pipeline import Engine
    from sentiment.snapshots import history

    with tempfile.TemporaryDirectory() as tmp:
        snap_dir = extract_snapshots(ref, Path(tmp))
        latest = latest_grades(snap_dir)
        tickers = pick(latest, n, random.Random(seed))
        log.info("picked %d of %d tickers from %s", len(tickers), len(latest), ref)
        engine = Engine(use_llm=use_llm)
        batch = {"id": datetime.now(timezone.utc).strftime("%Y%m%d"), "ref": ref, "seed": seed, "stocks": []}
        for i, t in enumerate(tickers, 1):
            now = datetime.now(timezone.utc)
            try:
                snap, items = engine.analyze(t, history=history(t, now, root=snap_dir), as_of=now)
            except Exception:  # noqa: BLE001 - one bad ticker never stops the batch
                log.exception("analyze %s failed", t)
                continue
            c = snap.composite
            batch["stocks"].append({
                "ticker": t, "company": snap.company, "as_of": now.isoformat(),
                "model_version": snap.model_version, "snapshot_grade": latest[t]["composite"]["grade"],
                "engine": {**_ch(c), "bounds": c.bounds,
                           "channels": {k: _ch(v) for k, v in snap.channels.items()}},
                "items": shown_items(items, now)})
            batch["built_at"] = datetime.now(timezone.utc).isoformat()
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp_out = out.with_suffix(".tmp")
            tmp_out.write_text(json.dumps(batch, ensure_ascii=False, default=str), encoding="utf-8")
            tmp_out.replace(out)   # saved after every ticker, so a stopped run still leaves a usable batch
            log.info("[%d/%d] %s grade=%s (snapshot %s) shown=%d", i, len(tickers), t, c.grade,
                     latest[t]["composite"]["grade"], len(batch["stocks"][-1]["items"]))
    return len(batch["stocks"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ref", default="origin/data")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    config.load_env()
    if a.out.exists():
        print(f"{a.out} exists; move it away first (grades refer to its batch id)", file=sys.stderr)
        return 2
    built = build(a.n, a.seed, a.ref, a.out, use_llm=not a.no_llm)
    print(f"{built} stocks -> {a.out}")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
