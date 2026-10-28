"""Append-only per-ticker snapshot log (data/snapshots/<TICKER>.jsonl) and daily-series reader."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config

SNAP_DIR = config.ROOT / "data" / "snapshots"
_KEYS = ("raw", "score", "z", "grade", "n_items", "n_effective", "agreement")
_CHANNELS = ("news", "filings", "social")


def compact(d: dict) -> dict:
    comp = d.get("composite") or {}
    return {
        "ticker": d.get("ticker"),
        "as_of": d.get("as_of"),
        "model_version": d.get("model_version"),
        "flags": d.get("flags"),
        "attention_z": d.get("attention_z"),
        "composite": {k: comp.get(k) for k in _KEYS},
        "channels": {ch: {k: (v or {}).get(k) for k in _KEYS} for ch, v in (d.get("channels") or {}).items()},
    }


def record(d: dict, root: Path = SNAP_DIR) -> Path:
    path = Path(root) / f"{d['ticker']}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(compact(d), ensure_ascii=False, default=str) + "\n")
    return path


def _parse(s) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s))
    except ValueError:
        return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def history(ticker: str, as_of: datetime, days: int = 90, root: Path = SNAP_DIR) -> dict[str, list[float]]:
    path = Path(root) / f"{ticker.upper()}.jsonl"
    if not path.exists():
        return {}
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    as_of = as_of.astimezone(timezone.utc)
    lo = as_of - timedelta(days=days)
    best: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            t = _parse(row.get("as_of"))
            if t is None or t < lo or t.date() >= as_of.date():
                continue
            cur = best.get(t.date())
            if cur is None or t >= cur[0]:
                best[t.date()] = (t, row)
    rows = [best[k][1] for k in sorted(best)]
    out: dict[str, list[float]] = {}
    for ch in _CHANNELS:
        if not any(ch in (r.get("channels") or {}) for r in rows):
            continue
        out[ch] = [float(((r.get("channels") or {}).get(ch) or {}).get("n_items") or 0) for r in rows]
        raw = [((r.get("channels") or {}).get(ch) or {}).get("raw") for r in rows]
        out[f"raw_{ch}"] = [float(x) for x in raw if x is not None]
    out["raw_composite"] = [float(x) for r in rows if (x := (r.get("composite") or {}).get("raw")) is not None]
    return {k: v for k, v in out.items() if v}
