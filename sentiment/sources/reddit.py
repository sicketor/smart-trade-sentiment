"""Reddit subreddit search RSS (social)."""
from __future__ import annotations

import re
from datetime import datetime

from .. import config
from ..http import Http
from ..types import Company, RawItem
from ._util import entry_dt, feed_entries, finish, item_id, strip_html

SOURCE = "reddit"
URL = "https://www.reddit.com/r/{sub}/search.rss"
_SUB = re.compile(r"/r/([^/]+)/")


def parse(payload: str, company: Company, since: datetime, sub: str | None = None) -> list[RawItem]:
    out = []
    for e in feed_entries(payload):
        title = strip_html(e.get("title"))
        url = e.get("link") or ""
        dt = entry_dt(e)
        if not title or not dt:
            continue
        content = e.get("content")
        raw = content[0].get("value") if content else (e.get("summary") or "")
        author = (e.get("author") or "").strip()
        s = sub or (_SUB.search(url).group(1) if _SUB.search(url) else "reddit")
        meta = {}
        out.append(RawItem(item_id(SOURCE, url, title), SOURCE, "social", f"r/{s}", title, strip_html(raw),
                           url, dt, company.ticker, meta))
        if author:
            out[-1].meta["author"] = author
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    name = company.short_name
    q = f'"{name}" OR "${company.ticker}"' if company.ambiguous else f'{company.ticker} OR "{name}"'
    items: list[RawItem] = []
    errors = []
    for sub in config.REDDIT_SUBS:
        try:
            text = http.get_bytes(URL.format(sub=sub),
                                 params={"q": q, "restrict_sr": 1, "sort": "new", "t": "week"})
            items.extend(parse(text, company, since, sub))
        except Exception as e:
            errors.append(f"{sub}: {e}")
    if errors and len(errors) == len(config.REDDIT_SUBS):
        raise RuntimeError("all subs failed: " + errors[0])
    return finish(items, since)
