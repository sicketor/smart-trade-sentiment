"""Writes engine results to the HUB Supabase DB (PostgREST over httpx): history rows, latest snapshot, universe."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from . import config
from .snapshots import compact

log = logging.getLogger(__name__)

_IGNORE_DUP = "resolution=ignore-duplicates,return=minimal"
_TICKER = re.compile(r"^[A-Z0-9.\-]{1,10}$")   # same check as the pro_sentiment_* tables
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
RUNS = ("main", "refresh", "manual", "ondemand", "list")


class StoreError(Exception):
    pass


def canonical(ticker: str) -> str:
    return ticker.strip().upper().replace("-", ".")


def parse_requests(spec: str) -> list[tuple[str, str]]:
    """'AAPL=uuid,BRK-B=uuid' -> [(canonical ticker, lowercase uuid)]; ValueError on any violation."""
    pairs, tickers, ids = [], set(), set()
    for part in (spec or "").split(","):
        left, sep, right = part.strip().partition("=")
        if not sep or "=" in right:
            raise ValueError("each request must look like TICKER=request_id")
        t, i = canonical(left), right.strip().lower()
        if not _TICKER.match(t):
            raise ValueError(f"invalid ticker {t[:12]!r}")
        if not _UUID.match(i):
            raise ValueError(f"invalid request id for {t}")
        if t in tickers or i in ids:
            raise ValueError("duplicate ticker or request id")
        tickers.add(t)
        ids.add(i)
        pairs.append((t, i))
    if not 1 <= len(pairs) <= 5:
        raise ValueError("1 to 5 requests are allowed")
    return pairs


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
        self._headers = {"apikey": key, "Content-Type": "application/json"}
        if not key.startswith("sb_"):   # legacy service_role JWT; new sb_secret_ keys go in apikey only (not a JWT)
            self._headers["Authorization"] = "Bearer " + key
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

    def _send(self, method: str, path: str, body=None, prefer: str | None = None, params=None) -> httpx.Response:
        headers = dict(self._headers)
        if prefer:
            headers["Prefer"] = prefer
        kw = {"params": params} if params else {}
        if method != "GET":
            kw["json"] = body
        for attempt in (0, 1):
            try:
                r = self._client.request(method, self._base + path, headers=headers, **kw)
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

    def _post(self, path: str, body, prefer: str | None = None) -> httpx.Response:
        return self._send("POST", path, body, prefer)

    def _patch(self, path: str, body, params=None) -> httpx.Response:
        return self._send("PATCH", path, body, "return=minimal", params)

    def _get(self, path: str, params=None) -> list:
        r = self._send("GET", path, params=params)
        try:
            data = r.json()
        except ValueError:
            raise StoreError(f"bad response {path}") from None
        if not isinstance(data, list):
            raise StoreError(f"bad response {path}")
        return data

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

    def mark_requests(self, ids: list[str], status: str, error: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if status == "running":
            allowed, body = "queued", {"status": "running", "started_at": now}
        elif status == "done":   # a result arriving after the 20-min timeout still closes the request
            allowed, body = "queued,running,failed", {"status": "done", "finished_at": now, "error": None}
        elif status == "failed":
            allowed = "queued,running"
            body = {"status": "failed", "finished_at": now, "error": (error or "analysis failed")[:500]}
        else:
            raise ValueError(f"unknown request status {status!r}")
        if not ids:
            return
        flt = {"id": "in.(" + ",".join(ids) + ")", "status": "in.(" + allowed + ")"}
        if status == "running":
            flt["status"] = "eq.queued"
        self._patch("/pro_sentiment_requests", body, flt)

    def _all(self, path: str, params: list) -> list:
        """Every row, page by page: PostgREST caps a response at 1000 rows."""
        rows, page = [], 1000
        while True:
            got = self._get(path, params + [("limit", str(page)), ("offset", str(len(rows)))])
            rows.extend(got)
            if len(got) < page:
                return rows

    def pick_list(self, cap: int, candidates: list[str] | None = None,
                  fresh_minutes: int = 30) -> tuple[list[str], int]:
        """Watch-list tickers for the hourly refresh (SPEC-he.md §10.2 row D): the union of all lists (or
        `candidates`), minus the universe (the daily runs cover it) and tickers analyzed in the last `fresh_minutes`.
        Oldest result first (never analyzed = oldest), at most `cap`. Returns (picked, how many were due)."""
        if candidates is None:
            candidates = [r.get("ticker") for r in self._all("/pro_sentiment_lists",
                                                             [("select", "ticker"), ("order", "ticker,user_id")])
                          if isinstance(r, dict)]
        tickers = {canonical(str(t)) for t in candidates if t and str(t).strip()}
        bad = sorted(t for t in tickers if not _TICKER.match(t))
        if bad:
            log.warning("skipping invalid list tickers %s", bad[:5])
            tickers -= set(bad)
        if not tickers:
            return [], 0
        universe = {canonical(str(r.get("ticker") or "")) for r in
                    self._all("/pro_sentiment_universe", [("select", "ticker"), ("order", "ticker")])
                    if isinstance(r, dict)}
        due = sorted(tickers - universe)
        last: dict[str, str] = {}
        for chunk in _chunks(due, 100):   # keeps the URL short
            for r in self._get("/pro_sentiment_latest", [
                ("select", "ticker,as_of"),
                ("ticker", "in.(" + ",".join(f'"{t}"' for t in chunk) + ")"),
            ]):
                if isinstance(r, dict) and r.get("ticker") and r.get("as_of"):
                    last[canonical(str(r["ticker"]))] = str(r["as_of"])
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=fresh_minutes)

        def when(t: str) -> datetime:
            try:
                d = datetime.fromisoformat(last[t].replace("Z", "+00:00"))
                return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                return datetime.min.replace(tzinfo=timezone.utc)

        due = sorted((t for t in due if when(t) <= cutoff), key=lambda t: (when(t), t))
        return due[:max(0, cap)], len(due)

    def pull_history(self, tickers: list[str], out_dir: Path, days: int = 90) -> int:
        tickers = sorted({canonical(t) for t in tickers})
        bad = [t for t in tickers if not _TICKER.match(t)]
        if bad:
            raise StoreError(f"invalid tickers {bad[:5]}")
        if not tickers:
            return 0
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._get("/pro_sentiment_snapshots", [
            ("select", "ticker,data"),
            ("ticker", "in.(" + ",".join(f'"{t}"' for t in tickers) + ")"),
            ("as_of", "gte." + since),
            # newest first: PostgREST caps a response (max_rows, 1000 by default), so a cut drops the oldest days
            ("order", "as_of.desc"),
        ])
        if isinstance(rows, list) and len(rows) >= 1000:
            log.warning("history pull hit the 1000-row cap; the oldest days are missing")
        by: dict[str, list[str]] = {}
        for row in reversed(rows):
            if not isinstance(row, dict) or not isinstance(row.get("data"), dict):
                continue
            t = canonical(str(row.get("ticker") or ""))
            if t in tickers:
                by.setdefault(t, []).append(json.dumps(row["data"], ensure_ascii=False))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for t, lines in by.items():
            (out_dir / f"{t}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n += len(lines)
        return n

    def close(self) -> None:
        self._client.close()
