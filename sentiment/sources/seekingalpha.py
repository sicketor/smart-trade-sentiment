"""Seeking Alpha combined symbol feed (news)."""
from __future__ import annotations

from datetime import datetime

from ..http import Http
from ..types import Company, RawItem
from ._util import entry_dt, feed_entries, finish, item_id, strip_html

SOURCE = "seekingalpha"
URL = "https://seekingalpha.com/api/sa/combined/{t}.xml"


def parse(payload: str, company: Company, since: datetime) -> list[RawItem]:
    out = []
    for e in feed_entries(payload):
        title = strip_html(e.get("title"))
        url = e.get("link") or ""
        dt = entry_dt(e)
        if not title or not dt:
            continue
        out.append(RawItem(item_id(SOURCE, url, title), SOURCE, "news", "Seeking Alpha", title,
                           strip_html(e.get("description") or e.get("summary")), url, dt, company.ticker))
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    return parse(http.get_bytes(URL.format(t=company.ticker)), company, since)
