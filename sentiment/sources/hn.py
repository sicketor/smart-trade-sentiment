"""Hacker News stories via Algolia (social)."""
from __future__ import annotations

from datetime import datetime

from ..http import Http
from ..types import Company, RawItem
from ._util import finish, item_id, strip_html, utc_from_epoch

SOURCE = "hn"
URL = "https://hn.algolia.com/api/v1/search_by_date"


def parse(payload: dict, company: Company, since: datetime) -> list[RawItem]:
    out = []
    for h in (payload or {}).get("hits") or []:
        title = strip_html(h.get("title"))
        ts = h.get("created_at_i")
        oid = h.get("objectID")
        if not title or ts is None:
            continue
        url = h.get("url") or (f"https://news.ycombinator.com/item?id={oid}" if oid else "")
        meta = {}
        if h.get("points") is not None:
            meta["likes"] = h["points"]
        out.append(RawItem(item_id(SOURCE, url, title), SOURCE, "social", "Hacker News", title, "", url,
                           utc_from_epoch(ts), company.ticker, meta))
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    data = http.get_json(URL, params={"query": f'"{company.short_name}"', "tags": "story",
                                      "hitsPerPage": 30})
    return parse(data, company, since)
