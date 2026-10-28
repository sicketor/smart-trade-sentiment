"""Finviz quote-page news table (news). Times are US Eastern."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from ..http import Http
from ..types import Company, RawItem
from ._util import finish, item_id, strip_html

SOURCE = "finviz"
URL = "https://finviz.com/quote.ashx"
NY = ZoneInfo("America/New_York")
_DATE = re.compile(
    r"^(?:(?P<d>[A-Za-z]{3}-\d{1,2}-\d{2,4})|(?P<w>Today|Yesterday))?\s*(?P<t>\d{1,2}:\d{2}\s*[AP]M)$", re.I)


def parse(payload: str, company: Company, since: datetime) -> list[RawItem]:
    tree = HTMLParser(payload)
    now_ny = datetime.now(NY)
    last_date = now_ny.date()
    out = []
    for tr in tree.css("table#news-table tr"):
        tds = tr.css("td")
        a = tr.css_first("a.tab-link-news")
        if len(tds) < 2 or a is None:
            continue
        m = _DATE.match(" ".join(tds[0].text().split()))
        if not m:
            continue
        try:
            if m.group("d"):
                fmt = "%b-%d-%y" if len(m.group("d").rsplit("-", 1)[1]) == 2 else "%b-%d-%Y"
                last_date = datetime.strptime(m.group("d"), fmt).date()
            elif m.group("w"):
                last_date = now_ny.date() - timedelta(days=1 if m.group("w").lower() == "yesterday" else 0)
            t = datetime.strptime(m.group("t").replace(" ", "").upper(), "%I:%M%p").time()
        except ValueError:
            continue
        dt = datetime.combine(last_date, t, tzinfo=NY).astimezone(timezone.utc)
        title = strip_html(a.text())
        url = a.attributes.get("href") or ""
        if not title or not url:
            continue
        pub = tr.css_first("div.news-link-right span")
        publisher = strip_html(pub.text()).strip("() ") if pub is not None else ""
        out.append(RawItem(item_id(SOURCE, url, title), SOURCE, "news", publisher or "Finviz", title, "",
                           url, dt, company.ticker))
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    return parse(http.get_text(URL, params={"t": company.ticker.replace(".", "-"), "p": "d"}), company, since)
