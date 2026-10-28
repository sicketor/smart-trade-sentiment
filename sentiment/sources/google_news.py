"""Google News RSS search (news)."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from ..http import Http
from ..types import Company, RawItem
from ._util import entry_dt, feed_entries, finish, item_id, strip_html

SOURCE = "google_news"
URL = "https://news.google.com/rss/search"


def parse(payload: str, company: Company, since: datetime) -> list[RawItem]:
    out = []
    for e in feed_entries(payload):
        title = strip_html(e.get("title"))
        url = e.get("link") or ""
        dt = entry_dt(e)
        if not title or not dt:
            continue
        publisher = ""
        src = e.get("source")
        if src:
            publisher = (src.get("title") or "").strip()
        if " - " in title:
            head, tail = title.rsplit(" - ", 1)
            if head and (not publisher or tail.strip() == publisher):
                title, publisher = head.strip(), tail.strip()
        out.append(RawItem(item_id(SOURCE, url, title), SOURCE, "news", publisher or "Google News",
                           title, "", url, dt, company.ticker))
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    days = max(1, math.ceil((datetime.now(timezone.utc) - since).total_seconds() / 86400))
    q = f'"{company.ticker}" OR "{company.short_name}" stock when:{days}d'
    text = http.get_bytes(URL, params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    return parse(text, company, since)
