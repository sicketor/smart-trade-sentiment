"""Ticker relevance and other-ticker detection (SPEC §6.3)."""
from __future__ import annotations

import re

from .types import Company, RawItem

_CASHTAG = re.compile(r"\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b")
_LISTICLE = re.compile(
    r"\b(\d+|top|best|worst)\s+(\w+\s+){0,3}(stocks|shares|picks|names|reasons|things)\b", re.I
)
_MARKET_REPORT = re.compile(r"\bmarket\b.{0,60}\b(report|forecast|size|share|outlook|analysis)\b", re.I)
_AGG_SOURCES = {"yahoo_rss", "finviz", "nasdaq", "seekingalpha"}
# "Verizon (NYSE:VZ, XETRA:BAC)": a symbol on a non-US exchange is often another company (Amcor ASX:AMC,
# Merck KGaA XTRA:MRK). It never counts as a ticker mention; a dual listing still matches by company name.
_FOREIGN = re.compile(
    r"\b(?:TSXV?|TSE|CSE|NEO|LSE|AIM|XETRA|XTRA|ETR|FRA|FWB|BIT|BME|EPA|ENXT\w*|AMS|EBR|SWX|SIX|VTX|HLSE|HEL|STO|"
    r"OM|CPH|OSE|OB|WSE|WBAG|VIE|ASX|NZX|NZSE|HKEX|HKG|SEHK|SHA|SHE|SSE|SZSE|TYO|KRX|KOSDAQ|NSE|BSE|TASE|TLV|"
    r"JSE|BMV|BOVESPA|BVMF|SGX|IDX|SET|KLSE|IST|ATSE)\s*:\s*[A-Z0-9][A-Z0-9.\-/]*")


def _cashtags(s: str) -> set[str]:
    return {m.upper() for m in _CASHTAG.findall(s)}


def _variants(ticker: str) -> set[str]:
    return {ticker, ticker.replace("-", "."), ticker.replace(".", "-")}


def _mentions(text: str, company: Company) -> bool:
    t = company.ticker
    text = _FOREIGN.sub(" ", text)
    if t in _cashtags(text):
        return True
    if re.search(rf"\((?:NASDAQ|NYSE|NYSEARCA|AMEX|NYSE American)\s*:\s*{re.escape(t)}\)", text, re.I):
        return True
    if not company.ambiguous and re.search(rf"\b{re.escape(t)}\b", text):
        return True
    names = [company.short_name, *company.aliases, company.name]
    for n in names:
        if n and len(n) >= 3 and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.I):
            return True
    return False


def other_tickers(item: RawItem, company: Company) -> set[str]:
    full = item.title + " " + item.text
    syms = {str(x).upper() for x in item.meta.get("symbols", [])}
    return (_cashtags(full) | syms) - _variants(company.ticker)


def relevance(item: RawItem, company: Company) -> float:
    if item.source == "sec":   # fetched by the company's own CIK; titles never name it
        return 1.0
    full = item.title + " " + item.text
    others = other_tickers(item, company)
    if not _mentions(full, company):
        if item.source == "stocktwits":
            return 0.8 if len(others) <= 1 else 0.5
        if item.source in _AGG_SOURCES:
            return 0.5
        return 0.0
    in_title = _mentions(item.title, company)
    base = 1.0 if in_title else 0.6
    if len(others) >= 3:
        base = min(base, 0.3)
    elif len(others) >= 1 and not in_title:
        base *= 0.8
    if _LISTICLE.search(item.title):
        base = min(base, 0.4)
    if _MARKET_REPORT.search(item.title):   # "ASIC Market Global Report ... Nvidia, AMD ..."
        base = min(base, 0.3)
    return round(base, 3)
