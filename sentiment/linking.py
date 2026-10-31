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


# "BX Oct 2026 135.000 put (BX261030P00135000)", "ZS 261030 380.00C (ZS261030C380000)": a quote page, no news in it
_OPTION_QUOTE = re.compile(r"\b[A-Z]{1,6}\d{6}[CP]\d{5,8}\b")
_MARKET_WORD = (r"(?i:stocks?|shares?|Q[1-4]|earnings|maintain(?:s|ed)|reiterated|initiated|upgraded|downgraded|"
                r"price target|DCF|SWOT|falls after|rises after|gears up)\b")


# Another company whose name contains ours.
_OTHER_NAME = re.compile(r"Portland General Electric|Omega Flex|Shoe Carnival|Lucid Diagnostics", re.I)

# What proves that an everyday-word name ("Target", "Block") is the company.
_WEAK_SUFFIX = r"(?:Holdings|Group|Inc|Corp|Corporation|Ltd)\b"
_WEAK_VERB = (r"(?i:reports?|announces?|falls?|rises?|drops?|sinks?|slides?|surges?|jumps?|climbs?|beats?|misses|"
              r"pauses?|skipped|raises?|cuts|lowers?|posts?|bulls|investors|director|ceo|cfo)\b")
_NOT_COMPANY = r"(?!\s+(?i:market|exchange|futures)\b)"
# Also an index or an exchange: "Dow rises 200 points", "Nasdaq stocks": verbs and plurals prove nothing here.
_INDEX_NAMES = {"DOW", "Nasdaq", "MSCI"}


def _bare(t: str) -> str:
    """The symbol as a word of its own: not DRX.L, T-Mobile, O'Reilly, P/E or AT&T; "GM's" still counts."""
    return rf"(?<![\w.\-'’/&]){re.escape(t)}(?![\w\-/&]|\.\w|['’](?!s\b))"


def _in_context(text: str, t: str) -> bool:
    bare = _bare(t)
    return bool(re.search(rf"{bare}(?:['’]s|:)?\s+(?:vs\b|or\s+[A-Z]{{2,5}}\b|{_MARKET_WORD})", text)
                or re.search(rf"\bvs\.?\s+{bare}", text))


def _weak_name_proof(text: str, short: str) -> bool:
    """ "Target Stock", "Lucid Group", "Snap Sinks 4%", "Costco vs. Target"; never "price target" or "stock news"."""
    name = rf"(?<![\w\-])(?<![Pp]rice\s)(?:{re.escape(short)}|{re.escape(short.title())})"
    if short in _INDEX_NAMES:
        return bool(re.search(rf"{name}(?:['’]s)?\s+(?:{_WEAK_SUFFIX}|(?i:stock|earnings|Q[1-4])\b{_NOT_COMPANY})", text))
    proof = rf"(?:{_MARKET_WORD}|{_WEAK_SUFFIX}|{_WEAK_VERB}|vs\b){_NOT_COMPANY}"
    return bool(re.search(rf"{name}(?:['’]s)?\s+{proof}", text) or re.search(rf"\bvs\.?\s+{name}", text))


def _cashtags(s: str) -> set[str]:
    return {m.upper() for m in _CASHTAG.findall(s)}


def _variants(ticker: str) -> set[str]:
    return {ticker, ticker.replace("-", "."), ticker.replace(".", "-")}


def _mentions(text: str, company: Company) -> bool:
    t = company.ticker
    text = _OTHER_NAME.sub(" ", _FOREIGN.sub(" ", text))
    if t in _cashtags(text):
        return True
    if re.search(rf"\((?:NASDAQ|NYSE|NYSEARCA|AMEX|NYSE American)\s*:\s*{re.escape(t)}\)", text, re.I):
        return True
    if re.search(rf"\({re.escape(t)}\s*:\s*(?:NASDAQ|NYSE)\)", text, re.I):   # "Colgate gains (CL:NYSE)"
        return True
    if not company.ambiguous and re.search(_bare(t), text):
        return True
    if company.contextual and _in_context(text, t):
        return True
    first = re.split(r"[\s\-]", company.short_name)[0]
    if (len(first) >= 3 and re.search(rf"\({re.escape(t)}\)", text)
            and re.search(rf"(?<!\w){re.escape(first)}(?!\w)", text, re.I)):
        return True   # "Deere (DE)", "Bunge Limited (BG)", but not "BAWAG Group AG (BG)"
    if company.weak_name and _weak_name_proof(text, company.short_name):
        return True
    names = [*([] if company.weak_name else [company.short_name]), *company.aliases, company.name]
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
    if _OPTION_QUOTE.search(item.title):
        return 0.0
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
