"""Lexicon, catalyst, numeric and structure features (SPEC §2.3, §5.3). Pure text processing."""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache

from . import config
from .text import normalize
from .types import Company, RawItem

_I = re.I

EVENT_PATTERNS: dict[str, re.Pattern] = {k: re.compile(v, _I) for k, v in {
    "EARNINGS_BEAT": r"(beat|beats|tops|topped|surpass\w*|exceed\w*)\b.{0,40}\b(estimates?|expectations|consensus|forecasts?|eps|revenue|profit)|record (revenue|profit|quarter)",
    "EARNINGS_MISS": r"(miss|misses|missed|falls? short|below)\b.{0,40}\b(estimates?|expectations|consensus|forecasts?|eps|revenue)",
    "GUIDANCE_UP": r"\b(raise[sd]?|boost\w*|lift\w*|hike[sd]?|increase[sd]?|up[sp]?)\s+(its\s+)?(full[- ]year\s+|annual\s+|fy\s?\d*\s+|q[1-4]\s+)?(guidance|outlook|forecast|view|targets?)",
    "GUIDANCE_DOWN": r"\b(cut\w*|lower\w*|slash\w*|reduce[sd]?|withdr[ae]w\w*|suspend\w*|trim\w*)\s+(its\s+)?(full[- ]year\s+|annual\s+|fy\s?\d*\s+|q[1-4]\s+)?(guidance|outlook|forecast|view|targets?)|(weak|soft|disappointing)\s+(guidance|outlook|forecast)",
    "ANALYST_UPGRADE": r"\bupgrade[sd]?\b|raised? to (buy|outperform|overweight)|initiates?.{0,30}(buy|outperform|overweight)",
    "ANALYST_DOWNGRADE": r"\bdowngrade[sd]?\b|cut to (sell|underperform|underweight|neutral|hold)|initiates?.{0,30}(sell|underperform|underweight)",
    "PRICE_TARGET": r"price target|\bPT\b",
    "OFFERING_DILUTION": r"(secondary|public|follow-on|registered direct|at-the-market|atm)\s+offering|shelf registration|424b[0-9]|\bS-3\b|dilut\w+|convertible (notes|senior notes) offering|going concern",
    "BUYBACK_DIVIDEND": r"buyback|repurchase|dividend (hike|increase|raise)|raises? (its )?dividend|special dividend",
    "M_AND_A": r"acquir\w+|merger|to buy .{0,30} for \$|takeover|buyout|tender offer|definitive agreement",
    "FDA_CLINICAL": r"\bFDA\b|clinical (trial|hold)|phase (1|2|3|i|ii|iii)\b|primary endpoint|complete response letter|\bCRL\b|breakthrough therapy|fast track",
    "LEGAL_REGULATORY": r"lawsuit|sued|class action|subpoena|\bSEC\b (probe|investigation|charges)|\bDOJ\b|antitrust|investigation|probe|fine[sd]?\b|settle\w*|indict\w*|recall",
    "INSIDER_TRADE": r"insider|form 4|director .{0,20}(bought|sold|buys|sells)|ceo .{0,20}(bought|sold|buys|sells|sale)",
    "PRODUCT_CONTRACT": r"(wins?|awarded|secures?|signs?|lands?)\s.{0,30}(contract|deal|order|partnership)|partnership with|launch(es|ed)?\b",
    "MANAGEMENT_CHANGE": r"(ceo|cfo|coo|chief executive|president)\s.{0,30}(resign\w*|step\w* down|depart\w*|appoint\w*|named|exit\w*|retire\w*)|names? new (ceo|cfo)",
    "MACRO_SECTOR": r"\b(fed|federal reserve|inflation|cpi|tariff\w*|rates?|recession|jobs report|treasury yields?)\b",
    "SHORT_REPORT": r"short[- ]seller|short report|hindenburg|muddy waters|citron|grizzly research|spruce point|wolfpack",
}.items()}
EVENT_TAGS = list(EVENT_PATTERNS)

_PT_UP = re.compile(r"\b(raise[sd]?|lift\w*|boost\w*|hike[sd]?|up[sp]?) .{0,20}price target| price target .{0,20}(raised|increased|lifted)", _I)
_PT_DOWN = re.compile(r"\b(cut\w*|lower\w*|slash\w*|reduce[sd]?|trim\w*) .{0,20}price target|price target .{0,20}(cut|lowered|reduced|slashed)", _I)
_PCT_UP = re.compile(r"\b(?:up|rises?|rose|jumps?|jumped|soars?|soared|gains?|gained|surges?|surged|climbs?|climbed|rall(?:y|ies|ied)|higher by) .{0,15}?(\d+(?:\.\d+)?)\s?%", _I)
_PCT_DOWN = re.compile(r"\b(?:down|falls?|fell|drops?|dropped|slides?|slid|plunges?|plunged|sinks?|sank|tumbles?|tumbled|declines?|declined|slumps?|slumped|lower by) .{0,15}?(\d+(?:\.\d+)?)\s?%", _I)
_CONTRAST = re.compile(r"\b(but|however|despite|although|though|nevertheless|yet)\b", _I)
_FORWARD = re.compile(r"guidance|outlook|forecast|expects?|sees|projects?|next quarter|full[- ]year|fy\d{2}|going forward", _I)
_WORD = re.compile(r"[a-z][a-z'-]*")
_ACRONYMS = {"CEO", "CFO", "EPS", "FDA", "SEC", "AI", "US", "ETF", "IPO", "GDP", "CPI"}
_BULL = "🚀📈🐂💎🙌🔥"
_BEAR = "📉🐻💀🩸"
_EMOJI = _BULL + _BEAR

FEATURE_KEYS = (
    ["lm_pos", "lm_neg", "lm_unc", "lm_lit", "lm_strong", "lm_weak",
     "vader_compound", "vader_pos", "vader_neg"]
    + [f"evt_{t}" for t in EVENT_TAGS]
    + ["evt_count", "eps_dir", "pt_dir", "pct_move", "insider_value",
       "has_contrast", "after_contrast_vader", "forward_looking", "is_question", "exclam",
       "caps_ratio", "emoji_bull", "emoji_bear", "title_len", "has_body",
       "is_news", "is_filings", "is_social", "user_label", "author_weight"]
)


@lru_cache(maxsize=1)
def _lm():
    import pysentiment2 as ps
    lm = ps.LM()
    return lm, set(lm._posset), set(lm._negset)


@lru_cache(maxsize=1)
def _extra() -> dict[str, set[str]]:
    d = json.loads((config.LEXICON_DIR / "lm_extra.json").read_text(encoding="utf-8"))
    return {k: set(v) for k, v in d.items()}


@lru_cache(maxsize=1)
def _trading() -> dict[str, float]:
    return json.loads((config.LEXICON_DIR / "trading_lexicon.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    a.lexicon.update(_trading())
    return a


def _vader_scores(text: str) -> dict[str, float]:
    spaced = "".join(f" {c} " if c in _EMOJI else c for c in text)
    return _vader().polarity_scores(spaced)


def _span(m: re.Match) -> str:
    s = " ".join(m.group(0).split())
    return s if len(s) <= 60 else s[:57] + "..."


def extract(item: RawItem, company: Company) -> tuple[dict[str, float], list[str], list[str]]:
    title = item.title or ""
    text = normalize(title + ". " + (item.text or ""))
    lower = text.lower()
    f: dict[str, float] = {k: 0.0 for k in FEATURE_KEYS}

    # --- Loughran-McDonald ---
    words = _WORD.findall(lower)
    n = max(len(words), 1)
    lm, pos, neg = _lm()
    toks = lm.tokenize(text)
    ex = _extra()
    f["lm_pos"] = sum(t in pos for t in toks) / n
    f["lm_neg"] = sum(t in neg for t in toks) / n
    f["lm_unc"] = sum(w in ex["uncertainty"] for w in words) / n
    f["lm_lit"] = sum(w in ex["litigious"] for w in words) / n
    f["lm_strong"] = sum(w in ex["strong_modal"] for w in words) / n
    f["lm_weak"] = sum(w in ex["weak_modal"] for w in words) / n

    # --- VADER ---
    v = _vader_scores(text)
    f["vader_compound"], f["vader_pos"], f["vader_neg"] = v["compound"], v["pos"], v["neg"]

    # --- events ---
    events: list[str] = []
    spans: list[str] = []
    for tag, pat in EVENT_PATTERNS.items():
        m = pat.search(text)
        if m:
            events.append(tag)
            f[f"evt_{tag}"] = 1.0
            spans.append(_span(m))
    f["evt_count"] = float(len(events))
    if not events:
        events = ["OTHER"]

    # --- numeric ---
    beat, miss = "EARNINGS_BEAT" in events, "EARNINGS_MISS" in events
    f["eps_dir"] = 1.0 if beat and not miss else -1.0 if miss and not beat else 0.0
    up, down = _PT_UP.search(text), _PT_DOWN.search(text)
    f["pt_dir"] = 1.0 if up else -1.0 if down else 0.0
    mu, md = _PCT_UP.search(text), _PCT_DOWN.search(text)
    pct = 0.0
    if mu and (not md or mu.start() <= md.start()):
        pct = float(mu.group(1))
    elif md:
        pct = -float(md.group(1))
    f["pct_move"] = max(-30.0, min(30.0, pct)) / 30.0
    try:
        iv = float(item.meta.get("insider_value", 0) or 0)
    except (TypeError, ValueError):
        iv = 0.0
    f["insider_value"] = math.copysign(min(1.0, math.log10(1 + abs(iv)) / 7), iv) if iv else 0.0

    # --- structure ---
    c = _CONTRAST.search(text)
    f["has_contrast"] = 1.0 if c else 0.0
    if c:
        f["after_contrast_vader"] = _vader_scores(text[c.end():])["compound"]
    f["forward_looking"] = min(len(_FORWARD.findall(text)), 3) / 3
    f["is_question"] = 1.0 if title.strip().endswith("?") else 0.0
    f["exclam"] = min(text.count("!"), 5) / 5
    long_words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 3]
    skip = _ACRONYMS | {company.ticker.upper()}
    caps = [w for w in long_words if w.isupper() and w not in skip]
    f["caps_ratio"] = len(caps) / len(long_words) if long_words else 0.0
    f["emoji_bull"] = min(sum(text.count(e) for e in _BULL), 5) / 5
    f["emoji_bear"] = min(sum(text.count(e) for e in _BEAR), 5) / 5
    f["title_len"] = min(len(title.split()), 30) / 30
    f["has_body"] = 1.0 if (item.text or "").strip() else 0.0

    # --- source ---
    f["is_news"] = 1.0 if item.channel == "news" else 0.0
    f["is_filings"] = 1.0 if item.channel == "filings" else 0.0
    f["is_social"] = 1.0 if item.channel == "social" else 0.0
    label = str(item.meta.get("user_label") or "").lower()
    f["user_label"] = 1.0 if label == "bullish" else -1.0 if label == "bearish" else 0.0
    fol = item.meta.get("followers")
    if fol:
        try:
            f["author_weight"] = min(1.0, math.log10(1 + max(float(fol), 0)) / 5)
        except (TypeError, ValueError):
            pass

    # --- drivers ---
    trading = _trading()
    found = {}
    for w in _WORD.findall(lower):
        if trading.get(w):
            found[w] = trading[w]
    for e in _EMOJI:
        if e in text and trading.get(e):
            found[e] = trading[e]
    strongest = sorted(found, key=lambda w: -abs(found[w]))
    drivers = list(dict.fromkeys(spans + strongest))[:3]
    return f, events, drivers
