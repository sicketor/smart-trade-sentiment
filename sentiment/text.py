"""Text normalization and splitting helpers (pure functions, no I/O)."""
from __future__ import annotations

import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_DATELINE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z .,/'&-]{1,40}?\s+)?\((?:Reuters|AP|AFP|Bloomberg|Dow Jones|MarketWatch|CNBC|UPI|Benzinga|PRNewswire|Business Wire|GlobeNewswire|Xinhua)\)\s*[-–—:]\s*"
)
_HANDLE = re.compile(r"(?<![\w])@\w+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)

_ABBREV = {
    "inc", "corp", "co", "u.s", "mr", "dr", "vs", "e.g", "i.e", "ms", "mrs", "ltd", "st",
    "jr", "sr", "no", "u.k", "etc", "prof", "sen", "rep", "gov", "gen", "lt", "col",
}
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+")
_STOP = {"the", "a", "an", "and", "of", "to", "in", "on", "for", "at", "by", "with", "from"}


def normalize(s: str) -> str:
    s = html.unescape(s or "")
    s = _TAG.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = _WS.sub(" ", s).strip()
    s = _DATELINE.sub("", s, count=1)
    return s.strip()


def for_social_model(s: str) -> str:
    s = _URL.sub("http", s)
    return _HANDLE.sub("@user", s)


def split_sentences(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(s):
        end = m.end()
        if end >= len(s):
            break
        nxt = s[end]
        if not (nxt.isupper() or nxt.isdigit() or nxt in "\"'(["):
            continue
        punct = m.group(0).strip()
        if punct.startswith("."):
            word = s[start:m.start()].split(" ")[-1] if m.start() > start else ""
            word = word.lstrip("(\"'[")
            if word.lower() in _ABBREV and len(punct.rstrip("\"')]")) == 1:
                continue
            if len(word) == 1 and word.isupper():
                continue
        out.append(s[start:end].strip())
        start = end
    tail = s[start:].strip()
    if tail:
        out.append(tail)
    return out


def title_key(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return " ".join(w for w in s.split() if w not in _STOP)
