"""Context series for the report: StockTwits bull %, Wikipedia views, prices (never raises)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from ..http import Http
from ..types import Company

WIKI_UA = "SmartTradeSentiment/0.1 (research@smartradeil.co.il)"


def stocktwits_bull_pct(data: dict) -> list[dict]:
    rows = []
    for d in (data or {}).get("data") or []:
        ts, bull = d.get("timestamp"), d.get("bullish")
        if ts and bull is not None:
            rows.append({"date": str(ts)[:10], "bull": bull})
    rows.sort(key=lambda r: r["date"])
    return rows


def wiki_views(data: dict) -> list[dict]:
    rows = []
    for d in (data or {}).get("items") or []:
        ts, v = str(d.get("timestamp", "")), d.get("views")
        if len(ts) >= 8 and v is not None:
            rows.append({"date": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}", "views": v})
    return rows


def prices(data: dict) -> list[dict]:
    res = ((data or {}).get("chart") or {}).get("result") or []
    if not res:
        return []
    r = res[0]
    closes = (((r.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
    rows = []
    for ts, c in zip(r.get("timestamp") or [], closes):
        if c is None:
            continue
        rows.append({"date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                     "close": round(c, 4)})
    return rows


def _wiki(company: Company, http: Http) -> list[dict]:
    s = http.get_json("https://en.wikipedia.org/w/api.php", ua=WIKI_UA, params={
        "action": "query", "list": "search", "srsearch": f"{company.short_name} company",
        "format": "json", "srlimit": 1})
    hits = (s.get("query") or {}).get("search") or []
    if not hits:
        raise ValueError("no wiki article")
    title = quote(hits[0]["title"].replace(" ", "_"), safe="")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
           f"{title}/daily/{start:%Y%m%d}00/{end:%Y%m%d}00")
    return wiki_views(http.get_json(url, ua=WIKI_UA))


def _chart(symbol: str, http: Http) -> list[dict]:
    data = http.get_json(f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
                         params={"range": "3mo", "interval": "1d"})
    return prices(data)


def fetch_context(company: Company, http: Http) -> dict:
    def safe(fn):
        try:
            return fn() or None
        except Exception:
            return None

    t = company.ticker
    return {
        "stocktwits_bull_pct": safe(lambda: stocktwits_bull_pct(
            http.get_json(f"https://api.stocktwits.com/api/2/symbols/{t}/sentiment.json"))),
        "wiki_views": safe(lambda: _wiki(company, http)),
        "prices": safe(lambda: _chart(t, http)),
        "spy_prices": safe(lambda: _chart("SPY", http)),
    }
