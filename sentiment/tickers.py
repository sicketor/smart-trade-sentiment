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
    "C": ["Citi"], "GE": ["GE Aerospace"], "PEG": ["PSEG"], "EL": ["Estée Lauder", "Estee Lauder"],
    "MP": ["MP Materials"], "PH": ["Parker Hannifin"], "PG": ["Procter And Gamble", "P&G"], "CL": ["Colgate"],
    "DD": ["DuPont"], "BG": ["Bunge"], "BR": ["Broadridge"], "TT": ["Trane"], "MU": ["Micron"], "VZ": ["Verizon"],
    "COO": ["CooperCompanies"], "KDP": ["Keurig"], "USB": ["U.S. Bancorp"], "GEHC": ["GE HealthCare"],
    "MSTR": ["MicroStrategy"], "XYZ": ["Block Inc"], "DAL": ["Delta Air"], "MPC": ["Marathon Petroleum"],
    "CCI": ["Crown Castle"], "APD": ["Air Products"], "DRI": ["Darden"], "NSC": ["Norfolk Southern"],
    "MLM": ["Martin Marietta"], "FIX": ["Comfort Systems"], "AMP": ["Ameriprise"], "ATO": ["Atmos"],
    "ICE": ["Intercontinental Exchange", "NYSE Parent", "ICE (ICE)", "ICE First Look"],
}

AMBIGUOUS = {
    "A", "AI", "ALL", "ARE", "BE", "BIG", "CAN", "CAR", "CASH", "COST", "EAT", "F", "FOR", "FUN",
    "GO", "HAS", "HE", "IT", "KEY", "LOW", "MAN", "NOW", "ON", "ONE", "OPEN", "OUT", "PLAY", "REAL",
    "RUN", "SAVE", "SEE", "SO", "TWO", "U", "UP", "V", "WELL", "X", "YOU", "ANY", "DAY", "NEW",
    "TRUE", "BEST", "LIFE", "LOVE", "PEAK", "RARE", "SAFE", "WISH",
    "C", "D", "J", "L", "O", "P", "Q", "T", "W",   # initials and share classes: "Arthur J.", "Class C"
}

# Also abbreviations of other things ("MS" the disease, "PM" the hour, "HD Hyundai", "ICE" the agency), yet real
# headlines use them bare ("MA Stock Drops"): measured 31/10 on live items, see linking._in_context.
CONTEXTUAL = {
    "BA", "BB", "BG", "BR", "CB", "CF", "CI", "CL", "DD", "DE", "ED", "EL", "ES", "EW", "FE", "GE", "GL", "GS", "HD",
    "IP", "IR", "MA", "MO", "MP", "MS", "PG", "PH", "PM", "RF", "RL", "SW", "TT", "WM",
    "DOC", "FAST", "ICE", "NUE", "PEG",
    "AFL", "AME", "AMP", "APD", "ATO", "BALL", "CCI", "CMS", "CNC", "COO", "COP", "CRM", "DAL", "DOW", "DRI", "ECHO",
    "ELV", "ESS", "ETN", "FIS", "FIVE", "FIX", "FLEX", "FOX", "GIS", "HAL", "HST", "IEX", "KDP", "MET", "MLM", "MPC",
    "MSI", "MTB", "NSC", "NWS", "PGR", "PNW", "PSA", "REG", "SNAP", "SRE", "STT", "TXT", "USB", "VST", "WAT", "WEC",
}

# Cleaned names that are everyday words ("price target", "stock news"): the name alone proves nothing.
WEAK_NAMES = {
    "Affirm", "BALL", "Block", "Booking", "Carnival", "Celsius", "Coherent", "DOW", "Dover", "Everest", "FLEX", "Fox",
    "Lucid", "MSCI", "NEWS", "Nasdaq", "Progressive", "Snap", "Strategy", "Target", "Toast", "Upstart", "Waters",
}

_SUFFIX = re.compile(
    r"(?:\b(?:inc|corp|corporation|com|co|company|ltd|limited|holdings|holding|group|plc|n\.v|s\.a|se|ag|"
    r"sa|llc|l\.p|lp|common stock)\b\.?|/[A-Z]{2}/|,|\bclass [abc]\b)",
    re.I,
)


def clean_name(sec_title: str) -> str:
    s = re.sub(r"\s[/\\]\s?\w+[/\\]?\s*$", " ", sec_title)   # "MP Materials Corp. / DE", "US Bancorp \DE\"
    s = _SUFFIX.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" .-&")
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
                         aliases=aliases, ambiguous=t in AMBIGUOUS or t in CONTEXTUAL, contextual=t in CONTEXTUAL,
                         weak_name=short in WEAK_NAMES)
    return out


def get_company(ticker: str, http: Http) -> Company | None:
    return load_companies(http).get(_norm(ticker.strip()))
