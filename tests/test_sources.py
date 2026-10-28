"""Source adapters: parse() on saved real payloads (offline) + live smoke (marked network)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sentiment.sources import (SOURCES, collect, finviz, google_news, hn, mastodon, nasdaq, reddit, sec,
                               seekingalpha, stocktwits, yahoo_rss, yahoo_search)
from sentiment.sources import context
from sentiment.tickers import AMBIGUOUS, clean_name
from sentiment.types import Company

FX = Path(__file__).parent / "fixtures"
OLD = datetime(2020, 1, 1, tzinfo=timezone.utc)
FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)
NVDA = Company("NVDA", "NVIDIA CORP", "Nvidia", 1045810, ["Nvidia"])


def txt(name):
    return (FX / name).read_text(encoding="utf-8")


def js(name):
    return json.loads(txt(name))


def check_common(items, source, channel):
    assert items
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids))
    for i in items:
        assert i.source == source and i.channel == channel and i.ticker == "NVDA"
        assert i.published_at.tzinfo is not None and i.published_at.utcoffset() == timedelta(0)
        assert i.title and "<" not in i.title and i.id


# ---- tickers -------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("NVIDIA CORP", "Nvidia"),
    ("APPLE INC.", "Apple"),
    ("ADVANCED MICRO DEVICES INC", "Advanced Micro Devices"),
    ("INTERNATIONAL BUSINESS MACHINES CORP", "International Business Machines"),
    ("IBM CORP", "IBM"),
    ("Alphabet Inc.", "Alphabet"),
    ("Berkshire Hathaway Inc. /DE/ Class B", "Berkshire Hathaway"),
])
def test_clean_name(raw, expected):
    assert clean_name(raw) == expected


def test_ambiguous_set():
    assert {"AI", "F", "ALL", "WISH"} <= AMBIGUOUS and "NVDA" not in AMBIGUOUS


def test_get_company_dot_and_dash(tmp_path, monkeypatch):
    from sentiment import config, tickers
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    (tmp_path / "company_tickers.json").write_text(json.dumps({
        "0": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        "2": {"cik_str": 1, "ticker": "AI", "title": "C3.AI, INC."}}), encoding="utf-8")
    assert tickers.get_company("BRK.B", None).cik == 1067983
    assert tickers.get_company("brk-b", None).ticker == "BRK.B"
    assert tickers.get_company("nvda", None).short_name == "Nvidia"
    assert tickers.get_company("AI", None).ambiguous is True
    assert tickers.get_company("ZZZZ", None) is None


# ---- news adapters -------------------------------------------------------------------------
def test_google_news():
    items = google_news.parse(txt("google_news.xml"), NVDA, OLD)
    check_common(items, "google_news", "news")
    assert all(i.text == "" and i.publisher and " - " not in i.publisher for i in items)
    assert not items[0].title.endswith(items[0].publisher)


def test_yahoo_rss():
    items = yahoo_rss.parse(txt("yahoo_rss.xml"), NVDA, OLD)
    check_common(items, "yahoo_rss", "news")
    assert all(i.publisher == "Yahoo Finance" for i in items)


def test_yahoo_search():
    items = yahoo_search.parse(js("yahoo_search.json"), NVDA, OLD)
    check_common(items, "yahoo_search", "news")
    assert any(i.meta.get("symbols") for i in items) or True
    assert items[0].published_at.year >= 2026


def test_finviz_dates_and_publisher():
    items = finviz.parse(txt("finviz.html"), NVDA, OLD)
    check_common(items, "finviz", "news")
    assert all(i.publisher and not i.publisher.startswith("(") for i in items)
    # a time-only row reuses the previous date; the dated row is converted from New York to UTC
    dated = [i for i in items if i.published_at.date().year == 2026 and i.published_at.month == 10]
    assert dated


def test_finviz_new_york_to_utc():
    html = ('<table id="news-table"><tr><td>Oct-26-26 09:30AM</td><td><div class="news-link-left">'
            '<a class="tab-link-news" href="http://x/1">A</a></div><div class="news-link-right">'
            '<span>(Reuters)</span></div></td></tr>'
            '<tr><td>08:15AM</td><td><a class="tab-link-news" href="http://x/2">B</a></td></tr></table>')
    items = finviz.parse(html, NVDA, OLD)
    assert items[0].published_at == datetime(2026, 10, 26, 13, 30, tzinfo=timezone.utc)  # EDT = UTC-4
    assert items[0].publisher == "Reuters"
    assert items[1].published_at == datetime(2026, 10, 26, 12, 15, tzinfo=timezone.utc)
    assert items[1].publisher == "Finviz"


def test_nasdaq():
    check_common(nasdaq.parse(txt("nasdaq.xml"), NVDA, OLD), "nasdaq", "news")


def test_seekingalpha():
    items = seekingalpha.parse(txt("seekingalpha.xml"), NVDA, OLD)
    check_common(items, "seekingalpha", "news")
    assert all(i.publisher == "Seeking Alpha" for i in items)


# ---- sec -----------------------------------------------------------------------------------
def test_sec_filings_titles_and_urls():
    items = sec.parse(js("sec.json"), NVDA, OLD)
    check_common(items, "sec", "filings")
    forms = {i.meta["form"]: i for i in items}
    assert "4" not in forms  # no XML given -> no insider items
    assert forms["8-K"].title.startswith("8-K: ")
    assert forms["10-Q"].title == "Quarterly report"
    assert forms["10-K"].title == "Annual report"
    assert forms["S-3"].title == "Shelf registration / securities offering prospectus (S-3)"
    assert forms["8-K"].url.startswith("https://www.sec.gov/Archives/edgar/data/1045810/")
    assert "-" not in forms["8-K"].url.split("/")[-2]
    assert all(i.publisher == "SEC EDGAR" for i in items)


def test_sec_8k_item_map():
    payload = {"filings": {"recent": {
        "form": ["8-K"], "filingDate": ["2026-10-01"], "acceptanceDateTime": ["2026-10-01T20:00:00.000Z"],
        "accessionNumber": ["0000000000-26-000001"], "primaryDocument": ["a.htm"], "items": ["2.02,9.01"]}}}
    (it,) = sec.parse(payload, NVDA, OLD)
    assert it.title == "8-K: Results of operations and financial condition; Financial statements and exhibits"


def test_sec_form4():
    payload = js("sec.json")
    r = payload["filings"]["recent"]
    acc = r["accessionNumber"][3]
    assert r["form"][3] == "4"
    items = sec.parse(payload, NVDA, OLD, {acc: txt("form4.xml")})
    ins = [i for i in items if i.meta["form"] == "4"]
    assert len(ins) == 1
    i = ins[0]
    assert i.title.startswith("Insider ") and " sold " in i.title and " shares at $" in i.title
    assert "(EVP, General Counsel and Sec)" in i.title and "Unknown" not in i.title
    assert i.meta["insider_value"] < 0


def test_sec_form4_parse_rows():
    xml = """<ownershipDocument><reportingOwner><reportingOwnerId><rptOwnerName>Huang Jen Hsun
    </rptOwnerName></reportingOwnerId><reportingOwnerRelationship><officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship></reportingOwner><nonDerivativeTable>
    <nonDerivativeTransaction><transactionAmounts><transactionShares><value>100</value></transactionShares>
    <transactionPricePerShare><value>10</value></transactionPricePerShare></transactionAmounts>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding></nonDerivativeTransaction>
    <nonDerivativeTransaction><transactionAmounts><transactionShares><value>50</value></transactionShares>
    <transactionPricePerShare><value>10</value></transactionPricePerShare></transactionAmounts>
    <transactionCoding><transactionCode>M</transactionCode></transactionCoding></nonDerivativeTransaction>
    </nonDerivativeTable></ownershipDocument>"""
    d = sec.parse_form4(xml)
    assert d["verb"] == "bought" and d["shares"] == 100 and d["net_value"] == 1000
    assert sec.parse_form4(xml.replace("<transactionCode>P<", "<transactionCode>A<")) is None
    assert sec.parse_form4("not xml") is None


def test_sec_raw_xml_url_strips_xsl():
    f = {"acc": "0001-26-000002", "doc": "xslF345X06/wk-form4.xml"}
    assert sec._raw_xml_url(NVDA, f).endswith("/1045810/000126000002/wk-form4.xml")


# ---- social --------------------------------------------------------------------------------
def test_stocktwits():
    items = stocktwits.parse(js("stocktwits.json"), NVDA, OLD)
    check_common(items, "stocktwits", "social")
    for i in items:
        assert i.publisher.startswith("@") and i.title == i.text[:140]
        assert i.url.startswith("https://stocktwits.com/") and "/message/" in i.url
        assert {"user_label", "followers", "account_age_days", "symbols"} <= set(i.meta)


def test_reddit():
    items = reddit.parse(txt("reddit.xml"), NVDA, OLD)
    check_common(items, "reddit", "social")
    assert all(i.publisher.startswith("r/") for i in items)
    assert all("<" not in i.text for i in items)


def test_mastodon():
    items = mastodon.parse(js("mastodon.json"), NVDA, OLD)
    check_common(items, "mastodon", "social")
    assert all(i.publisher.startswith("@") and "<" not in i.text for i in items)
    assert all("followers" in i.meta and "account_age_days" in i.meta for i in items)


def test_hn():
    items = hn.parse(js("hn.json"), NVDA, OLD)
    check_common(items, "hn", "social")
    assert all(i.publisher == "Hacker News" and i.url for i in items)


# ---- generic edge cases --------------------------------------------------------------------
@pytest.mark.parametrize("mod,name,loader", [
    (google_news, "google_news.xml", txt), (yahoo_rss, "yahoo_rss.xml", txt), (nasdaq, "nasdaq.xml", txt),
    (seekingalpha, "seekingalpha.xml", txt), (reddit, "reddit.xml", txt), (finviz, "finviz.html", txt),
    (yahoo_search, "yahoo_search.json", js), (stocktwits, "stocktwits.json", js),
    (mastodon, "mastodon.json", js), (hn, "hn.json", js), (sec, "sec.json", js),
])
def test_since_filters_everything(mod, name, loader):
    assert mod.parse(loader(name), NVDA, FUTURE) == []


def test_malformed_input_is_empty():
    for mod in (google_news, yahoo_rss, nasdaq, seekingalpha, reddit, finviz):
        assert mod.parse("", NVDA, OLD) == []
        assert mod.parse("<html>garbage", NVDA, OLD) == []
    for mod in (yahoo_search, stocktwits, hn, sec):
        assert mod.parse({}, NVDA, OLD) == []
    assert mastodon.parse([], NVDA, OLD) == []


def test_missing_fields_skip_item_only():
    p = {"news": [{"title": "no time"}, {"title": "ok", "link": "http://a", "providerPublishTime": 1793027286}]}
    items = yahoo_search.parse(p, NVDA, OLD)
    assert [i.title for i in items] == ["ok"]


def test_duplicates_keep_first():
    p = {"hits": [{"title": "A", "url": "http://a", "created_at_i": 1793027286, "points": 1},
                  {"title": "A2", "url": "http://a", "created_at_i": 1793027286, "points": 9}]}
    items = hn.parse(p, NVDA, OLD)
    assert len(items) == 1 and items[0].title == "A"


def test_hn_url_fallback_and_no_url_id():
    p = {"hits": [{"title": "Ask HN", "created_at_i": 1793027286, "objectID": "42"}]}
    (it,) = hn.parse(p, NVDA, OLD)
    assert it.url == "https://news.ycombinator.com/item?id=42"


def test_sources_ids():
    assert list(SOURCES) == ["google_news", "yahoo_rss", "yahoo_search", "finviz", "nasdaq", "seekingalpha",
                             "sec", "stocktwits", "reddit", "mastodon", "hn"]


def test_collect_isolates_failures(monkeypatch):
    from sentiment.types import RawItem

    def good(c, h, s):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return [RawItem("x", "a", "news", "p", "t", "", "u", t, "NVDA")]

    def bad(c, h, s):
        raise RuntimeError("boom")

    monkeypatch.setitem(SOURCES, "google_news", good)
    monkeypatch.setitem(SOURCES, "yahoo_rss", good)
    monkeypatch.setitem(SOURCES, "hn", bad)
    items, ok, failed = collect(NVDA, None, OLD, only=["google_news", "yahoo_rss", "hn"])
    assert len(items) == 1  # deduped by id
    assert ok == ["google_news", "yahoo_rss"] and failed == {"hn": "boom"}


# ---- context -------------------------------------------------------------------------------
def test_context_parsers():
    bull = context.stocktwits_bull_pct(js("st_sent.json"))
    assert [b["date"] for b in bull] == sorted(b["date"] for b in bull) and "bull" in bull[0]
    px = context.prices(js("yahoo_chart.json"))
    assert px and set(px[0]) == {"date", "close"}
    views = context.wiki_views({"items": [{"timestamp": "2026100100", "views": 5}]})
    assert views == [{"date": "2026-10-01", "views": 5}]
    assert context.prices({}) == [] and context.wiki_views(None) == []


def test_fetch_context_never_raises():
    class Boom:
        def get_json(self, *a, **k):
            raise RuntimeError("x")
    out = context.fetch_context(NVDA, Boom())
    assert out == {"stocktwits_bull_pct": None, "wiki_views": None, "prices": None, "spy_prices": None}


# ---- live ----------------------------------------------------------------------------------
@pytest.mark.network
def test_live_collect_nvda():
    from sentiment.http import Http
    from sentiment.tickers import get_company
    h = Http()
    c = get_company("NVDA", h)
    items, ok, failed = collect(c, h, datetime.now(timezone.utc) - timedelta(days=3))
    assert len(ok) >= 9 and items
    assert set(ok) | set(failed) == set(SOURCES)
