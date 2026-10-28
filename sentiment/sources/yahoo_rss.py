"""Yahoo Finance headline RSS (news)."""
from __future__ import annotations

from datetime import datetime

from ..http import Http
from ..types import Company, RawItem
from ._util import entry_dt, feed_entries, finish, item_id, strip_html

SOURCE = "yahoo_rss"
URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"


def parse(payload: str, company: Company, since: datetime) -> list[RawItem]:
    out = []
    for e in feed_entries(payload):
        title = strip_html(e.get("title"))
        url = e.get("link") or ""
        dt = entry_dt(e)
        if not title or not dt:
            continue
        out.append(RawItem(item_id(SOURCE, url, title), SOURCE, "news", "Yahoo Finance", title,
                           strip_html(e.get("description") or e.get("summary")), url, dt, company.ticker))
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    text = http.get_bytes(URL, params={"s": company.ticker.replace(".", "-"), "region": "US", "lang": "en-US"})
    return parse(text, company, since)
