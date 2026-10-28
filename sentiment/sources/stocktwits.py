"""StockTwits symbol stream (social)."""
from __future__ import annotations

from datetime import datetime

from ..http import Http
from ..types import Company, RawItem
from ._util import account_age_days, finish, item_id, parse_iso, strip_html

SOURCE = "stocktwits"
URL = "https://api.stocktwits.com/api/2/streams/symbol/{t}.json"
MAX_EXTRA_PAGES = 2


def parse(payload: dict, company: Company, since: datetime) -> list[RawItem]:
    out = []
    for m in (payload or {}).get("messages") or []:
        body = strip_html(m.get("body"))
        user = m.get("user") or {}
        username = user.get("username")
        mid = m.get("id")
        dt = parse_iso(m.get("created_at"))
        if not body or not username or mid is None or dt is None:
            continue
        sent = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        meta = {
            "user_label": sent or None,
            "followers": user.get("followers"),
            "account_age_days": account_age_days(user.get("join_date")),
            "symbols": [s["symbol"] for s in m.get("symbols") or [] if s.get("symbol")],
        }
        likes = (m.get("likes") or {}).get("total")
        if likes is not None:
            meta["likes"] = likes
        url = f"https://stocktwits.com/{username}/message/{mid}"
        out.append(RawItem(item_id(SOURCE, url, body), SOURCE, "social", "@" + username, body[:140], body,
                           url, dt, company.ticker, meta))
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    url = URL.format(t=company.ticker)
    items: list[RawItem] = []
    params = None
    for _ in range(1 + MAX_EXTRA_PAGES):
        data = http.get_json(url, params=params)
        msgs = data.get("messages") or []
        if not msgs:
            break
        items.extend(parse(data, company, since))
        ids = [m["id"] for m in msgs if m.get("id") is not None]
        dates = [d for d in (parse_iso(m.get("created_at")) for m in msgs) if d]
        if not ids or not dates or min(dates) < since:
            break
        params = {"max": min(ids)}
    return finish(items, since)
