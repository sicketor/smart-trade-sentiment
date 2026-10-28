"""JSONL storage shared by the corpus, teacher and gold steps."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from sentiment import config
from sentiment.types import Company, RawItem

CORPUS = config.ROOT / "data" / "corpus" / "items.jsonl"
TEACHER = config.ROOT / "data" / "teacher" / "labels.jsonl"
GOLD = config.ROOT / "data" / "gold" / "labels.jsonl"

# One split rule for every labeled item id (sha1 % 100), shared by student and meta-model:
#   teacher: < 65 student train, 65..69 student val, >= META_RESERVED never seen by the student
#            (70..89 meta train, >= 90 meta hold-out) — no stacking leakage through st_finsent.
#   gold:    scored by eval.crossfit (5 folds by bucket % 5) on buckets >= 65 plus the gold ids the
#            student was built without (models/st-finsent/gold_ids.json); the final meta-model
#            trains on all gold. Teacher rows of gold ids never train the meta-model.
META_RESERVED = 70


def bucket(item_id: str) -> int:
    import hashlib
    return int(hashlib.sha1(item_id.encode("utf-8")).hexdigest(), 16) % 100


def item_to_json(it: RawItem) -> dict:
    d = asdict(it)
    d["published_at"] = it.published_at.isoformat()
    return d


def item_from_json(d: dict) -> RawItem:
    d = dict(d)
    d["published_at"] = datetime.fromisoformat(d["published_at"])
    d.setdefault("meta", {})
    return RawItem(**d)


def company_to_json(c: Company) -> dict:
    return asdict(c)


def company_from_json(d: dict) -> Company:
    return Company(**d)


def read_jsonl(path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def latest_teacher(rows: Iterable[dict]) -> list[dict]:
    """One teacher row per (id, ticker): the highest prompt version `pv` wins (rows without it are v1);
    among equal versions the first row wins, so a re-run never overwrites a label."""
    best: dict = {}
    for t in rows:
        if "id" not in t:
            continue
        key = (t["id"], t.get("ticker"))
        if key not in best or t.get("pv", 1) > best[key].get("pv", 1):
            best[key] = t
    return list(best.values())


def append_jsonl(path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
