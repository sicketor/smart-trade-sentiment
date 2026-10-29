"""Writes engine results to the HUB Supabase DB (PostgREST over httpx): history rows, latest snapshot, universe."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from pathlib import Path

import httpx

from . import config
from .snapshots import compact

log = logging.getLogger(__name__)

_IGNORE_DUP = "resolution=ignore-duplicates,return=minimal"
_TICKER = re.compile(r"^[A-Z0-9.\-]{1,10}$")   # same check as the pro_sentiment_* tables
RUNS = ("main", "refresh", "manual", "ondemand", "list")


class StoreError(Exception):
    pass


def canonical(ticker: str) -> str:
    return ticker.strip().upper().replace("-", ".")


def clean_json(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_json(v) for v in x]
    return x


def _grade(g) -> int | None:
    """DB check: smallint -3..3 or null."""
    return g if isinstance(g, int) and not isinstance(g, bool) and -3 <= g <= 3 else None


def _model_version(d: dict) -> str:
    return str(d.get("model_version") or "unknown")[:200]


def valid_row(row: dict) -> bool:
    """Rows the DB checks would reject; one of them fails a whole PostgREST batch."""
    return isinstance(row.get("ticker"), str) and bool(_TICKER.match(row["ticker"])) and row.get("run") in RUNS


def _valid(rows: list[dict]) -> list[dict]:
    keep = [r for r in rows if valid_row(r)]
    for r in rows:
        if not valid_row(r):
            log.warning("skipping a row the DB would reject: ticker=%r run=%r", r.get("ticker"), r.get("run"))
    return keep


def snapshot_row(d: dict, run: str) -> dict:
    comp = d.get("composite") or {}
    r = d.get("run") or run
    return clean_json({
        "ticker": canonical(str(d["ticker"])), "as_of": d["as_of"], "run": r,
        "model_version": _model_version(d),
        "grade": _grade(comp.get("grade")), "score": comp.get("score"), "z": comp.get("z"),
        "attention_z": d.get("attention_z"),
        "data": compact({**d, "run": r}),
    })


def latest_row(d: dict, run: str) -> dict:
    comp = d.get("composite") or {}
    payload = {k: v for k, v in d.items() if k not in ("items", "elapsed_s", "timings_s")}
    payload["run"] = run
    return clean_json({
        "ticker": canonical(str(d["ticker"])), "company": d.get("company"), "as_of": d["as_of"], "run": run,
        "model_version": _model_version(d),
        "grade": _grade(comp.get("grade")), "score": comp.get("score"), "payload": payload,
    })


def _chunks(rows: list, n: int):
    n = max(1, n)
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


class Store:
    def __init__(self, url: str, key: str, client: httpx.Client | None = None):
        self._base = url.rstrip("/") + "/rest/v1"
        self._headers = {"apikey": key, "Authorization": "Bearer " + key, "Content-Type": "application/json"}
        self._client = client or httpx.Client(timeout=config.SUPABASE_TIMEOUT)

    def __repr__(self) -> str:
        return f"Store({self._base})"

    @classmethod
    def from_env(cls) -> "Store | None":
        env = config.ROOT / ".env"
        if env.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env, override=False)
            except ImportError:
                pass
        url = (os.environ.get("SUPABASE_URL") or "").strip()
        key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        if not url or not key:
            log.info("Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY); skipping the DB write")
            return None
        return cls(url, key)

    def _post(self, path: str, body, prefer: str | None = None) -> httpx.Response:
        headers = dict(self._headers)
        if prefer:
            headers["Prefer"] = prefer
        for attempt in (0, 1):
            try:
                r = self._client.post(self._base + path, json=body, headers=headers)
            except httpx.TransportError as e:
                if attempt == 0:
                    time.sleep(2)
                    continue
                raise StoreError(f"transport error {path}: {type(e).__name__}") from None
            except (httpx.HTTPError, httpx.InvalidURL) as e:   # e.g. a malformed SUPABASE_URL
                raise StoreError(f"request error {path}: {type(e).__name__}") from None
            if r.status_code >= 500 and attempt == 0:
                time.sleep(2)
                continue
            if r.status_code >= 400:
                raise StoreError(f"{r.status_code} {path}: {r.text[:300]}")
            return r
        raise StoreError(f"request failed {path}")  # unreachable

    def _rpc_int(self, path: str, body) -> int:
        r = self._post(path, body)
        try:
            return int(r.json())
        except (ValueError, TypeError):
            raise StoreError(f"bad rpc response {path}: {r.text[:100]}") from None

    def _insert_snapshots(self, rows: list[dict]) -> list[str]:
        """Every chunk is tried; returns the errors instead of stopping at the first one."""
        errors = []
        for chunk in _chunks(rows, config.SUPABASE_BATCH):
            try:
                self._post("/pro_sentiment_snapshots?on_conflict=ticker,as_of,run", chunk, _IGNORE_DUP)
            except StoreError as e:
                errors.append(str(e))
        return errors

    def write_results(self, results: list[dict], run: str) -> dict:
        snaps, latest = [], []
        for d in results:
            if not isinstance(d, dict) or not d.get("ticker") or not d.get("as_of"):
                log.warning("skipping a result without ticker/as_of (%s)",
                            d.get("ticker") if isinstance(d, dict) else None)
                continue
            snaps.append(snapshot_row(d, run))
            latest.append(latest_row(d, run))
        snaps, latest = _valid(snaps), _valid(latest)
        n_latest = 0
        errors = self._insert_snapshots(snaps)   # a failed history write must not leave latest stale
        for chunk in _chunks(latest, config.SUPABASE_BATCH):
            try:
                n_latest += self._rpc_int("/rpc/pro_sentiment_upsert_latest", {"rows": chunk})
            except StoreError as e:
                errors.append(str(e))
        if errors:
            raise StoreError("; ".join(errors))
        return {"snapshots": len(snaps), "latest": n_latest}

    def set_universe(self, tickers: list[str]) -> int:
        clean = sorted({canonical(t) for t in tickers if t and t.strip()})
        bad = [t for t in clean if not _TICKER.match(t)]
        if bad:   # the RPC replaces the whole universe, so a partial list is never sent
            raise StoreError(f"invalid universe tickers {bad[:5]}; refusing to replace it")
        if not clean:
            raise StoreError("universe is empty; refusing to replace it")
        return self._rpc_int("/rpc/pro_sentiment_set_universe", {"p_tickers": clean})

    def import_history(self, snap_dir: Path) -> int:
        rows = []
        for f in sorted(Path(snap_dir).glob("*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(d, dict) or not d.get("ticker") or not d.get("as_of"):
                        continue
                    rows.append(snapshot_row(d, "manual"))
        rows = _valid(rows)
        errors = self._insert_snapshots(rows)
        if errors:
            raise StoreError("; ".join(errors))
        return len(rows)

    def close(self) -> None:
        self._client.close()
