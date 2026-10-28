"""Yahoo Finance search endpoint, news part (news)."""
from __future__ import annotations

from datetime import datetime

from ..http import Http
from ..types import Company, RawItem
from ._util import finish, item_id, strip_html, utc_from_epoch

SOURCE = "yahoo_search"
URL = "https://query1.finance.yahoo.com/v1/finance/search"


def parse(payload: dict, company: Company, since: datetime) -> list[RawItem]:
    out = []
    for n in (payload or {}).get("news") or []:
        title = strip_html(n.get("title"))
        url = n.get("link") or ""
        ts = n.get("providerPublishTime")
        if not title or ts is None:
            continue
        meta = {}
        if n.get("relatedTickers"):
            meta["symbols"] = list(n["relatedTickers"])
        out.append(RawItem(item_id(SOURCE, url, title), SOURCE, "news", n.get("publisher") or "Yahoo Finance",
                           title, "", url, utc_from_epoch(ts), company.ticker, meta))
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    data = http.get_json(URL, params={"q": company.ticker.replace(".", "-"), "newsCount": 25, "quotesCount": 0})
    return parse(data, company, since)
