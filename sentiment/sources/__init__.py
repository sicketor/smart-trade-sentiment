"""Source adapters. Each module exposes fetch(company, http, since) and a pure parse(payload, ...)."""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from ..http import Http
from ..types import Company, RawItem
from . import (finviz, google_news, hn, mastodon, nasdaq, reddit, sec, seekingalpha, stocktwits,
               yahoo_rss, yahoo_search)

SOURCES: dict[str, Callable] = {
    "google_news": google_news.fetch,
    "yahoo_rss": yahoo_rss.fetch,
    "yahoo_search": yahoo_search.fetch,
    "finviz": finviz.fetch,
    "nasdaq": nasdaq.fetch,
    "seekingalpha": seekingalpha.fetch,
    "sec": sec.fetch,
    "stocktwits": stocktwits.fetch,
    "reddit": reddit.fetch,
    "mastodon": mastodon.fetch,
    "hn": hn.fetch,
}


def _short_error(e: Exception) -> str:
    msg = str(e).strip() or type(e).__name__
    return msg[:120]


def collect(company: Company, http: Http, since: datetime,
            only: list[str] | None = None) -> tuple[list[RawItem], list[str], dict[str, str]]:
    """-> (items deduped by id, ok source ids, failed {source: short error}). Never raises."""
    names = [n for n in SOURCES if only is None or n in only]
    ok: list[str] = []
    failed: dict[str, str] = {}
    results: dict[str, list[RawItem]] = {}

    def run(name: str) -> None:
        try:
            results[name] = SOURCES[name](company, http, since)
        except Exception as e:  # adapters must never take the run down
            failed[name] = _short_error(e)

    if names:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run, names))
    seen: set[str] = set()
    items: list[RawItem] = []
    for name in names:
        if name in results:
            ok.append(name)
            for it in results[name]:
                if it.id not in seen:
                    seen.add(it.id)
                    items.append(it)
    return items, ok, failed
