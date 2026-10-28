"""SEC company lookup (ticker -> Company) and name cleaning."""
from __future__ import annotations

import json
import re
import time

from . import config
from .http import Http
from .types import Company

SEC_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_TTL = 7 * 86400

ALIASES: dict[str, list[str]] = {
    "GOOGL": ["Google", "Alphabet"], "GOOG": ["Google", "Alphabet"],
    "META": ["Meta", "Facebook"], "AMD": ["AMD"], "IBM": ["IBM"], "BRK.B": ["Berkshire"],
    "BRK.A": ["Berkshire"], "TSLA": ["Tesla"], "MSFT": ["Microsoft"], "AMZN": ["Amazon"],
    "NVDA": ["Nvidia"], "AAPL": ["Apple"], "NFLX": ["Netflix"], "JPM": ["JPMorgan", "JP Morgan"],
    "BAC": ["Bank of America"], "GS": ["Goldman Sachs"], "WMT": ["Walmart"], "KO": ["Coca-Cola"],
    "DIS": ["Disney"], "INTC": ["Intel"], "PLTR": ["Palantir"], "AVGO": ["Broadcom"], "TSM": ["TSMC"],
}

AMBIGUOUS = {
    "A", "AI", "ALL", "ARE", "BE", "BIG", "CAN", "CAR", "CASH", "COST", "EAT", "F", "FOR", "FUN",
    "GO", "HAS", "HE", "IT", "KEY", "LOW", "MAN", "NOW", "ON", "ONE", "OPEN", "OUT", "PLAY", "REAL",
    "RUN", "SAVE", "SEE", "SO", "TWO", "U", "UP", "V", "WELL", "X", "YOU", "ANY", "DAY", "NEW",
    "TRUE", "BEST", "LIFE", "LOVE", "PEAK", "RARE", "SAFE", "WISH",
}

_SUFFIX = re.compile(
    r"(?:\b(?:inc|corp|corporation|com|co|company|ltd|limited|holdings|holding|group|plc|n\.v|s\.a|se|ag|"
    r"llc|l\.p|lp|common stock)\b\.?|/[A-Z]{2}/|,|\bclass [abc]\b)",
    re.I,
)


def clean_name(sec_title: str) -> str:
    s = _SUFFIX.sub(" ", sec_title)
    s = re.sub(r"\s+", " ", s).strip(" .-")
    out = []
    for w in s.split(" "):
        if w.isupper() and len(w) > 4:
            w = w.title()
        out.append(w)
    return " ".join(out)


def _norm(t: str) -> str:
    return t.upper().replace("-", ".")


def load_companies(http: Http) -> dict[str, Company]:
    path = config.CACHE_DIR / "company_tickers.json"
    data = None
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = None
    if data is None:
        data = http.get_json(SEC_URL, ua=config.SEC_UA)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    out: dict[str, Company] = {}
    for row in data.values():
        t = _norm(str(row["ticker"]))
        if t in out:
            continue
        name = row["title"]
        short = clean_name(name)
        aliases = list(ALIASES.get(t, []))
        out[t] = Company(ticker=t, name=name, short_name=short, cik=int(row["cik_str"]),
                         aliases=aliases, ambiguous=t in AMBIGUOUS)
    return out


def get_company(ticker: str, http: Http) -> Company | None:
    return load_companies(http).get(_norm(ticker.strip()))
