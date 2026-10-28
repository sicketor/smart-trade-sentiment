"""Mastodon hashtag timeline (social)."""
from __future__ import annotations

from datetime import datetime

from ..http import Http
from ..types import Company, RawItem
from ._util import account_age_days, finish, item_id, parse_iso, strip_html

SOURCE = "mastodon"
URL = "https://mastodon.social/api/v1/timelines/tag/{t}"


def parse(payload: list, company: Company, since: datetime) -> list[RawItem]:
    out = []
    for s in payload or []:
        text = strip_html(s.get("content"))
        acc = s.get("account") or {}
        acct = acc.get("acct")
        dt = parse_iso(s.get("created_at"))
        if not text or not acct or dt is None:
            continue
        url = s.get("url") or s.get("uri") or ""
        meta = {"followers": acc.get("followers_count"),
                "account_age_days": account_age_days(acc.get("created_at"))}
        out.append(RawItem(item_id(SOURCE, url, text), SOURCE, "social", "@" + acct, text[:140], text, url,
                           dt, company.ticker, meta))
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    data = http.get_json(URL.format(t=company.ticker.replace(".", "")), params={"limit": 40})
    return parse(data, company, since)
