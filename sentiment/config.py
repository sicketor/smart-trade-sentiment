"""Tunable constants. Starting values from SPEC-he.md; phase 2 recalibrates them."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("SENTIMENT_CACHE", ROOT / ".cache"))
MODELS_DIR = ROOT / "models"
LEXICON_DIR = ROOT / "data" / "lexicons"

# Browser UA for sites that block bots; SEC needs "Name email" (secret SEC_USER_AGENT in CI).
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
SEC_UA = os.environ.get("SEC_USER_AGENT", "SmartTrade Sentiment research@smartradeil.co.il")

# Minimum seconds between two requests to the same host.
HOST_MIN_INTERVAL = {
    "www.reddit.com": 3.0,
    "finviz.com": 2.0,
    "news.google.com": 1.0,
    "data.sec.gov": 0.15,
    "www.sec.gov": 0.15,
    "api.marketaux.com": 2.5,   # free plan returns 429 after ~30 fast requests
}
DEFAULT_HOST_INTERVAL = 0.5
HTTP_TIMEOUT = 15.0

LOOKBACK_DAYS = 7           # collect items up to this age
REDDIT_SUBS = ("wallstreetbets", "stocks", "investing", "options", "StockMarket")

# SPEC §8
HALF_LIFE_HOURS = {"news": 24.0, "filings": 72.0, "social": 6.0}
SHRINK_K = {"news": 3.0, "filings": 2.0, "social": 5.0}
CHANNEL_WEIGHTS = {"news": 0.45, "filings": 0.25, "social": 0.30}
MIN_EFFECTIVE_ITEMS = 1.0   # below this a channel shows "low data"
MIN_ITEMS = 3               # fewer items with weight > 0 -> "low data"
# Score normalization: cross-section stats from the backfill, blended with the ticker's own history.
AGG_NORM_PATH = ROOT / "sentiment" / "aggregate_norm.json"
HIST_PRIOR_DAYS = 60        # history weight = n / (n + HIST_PRIOR_DAYS)
HIST_MIN_DAYS = 5           # fewer history days -> cross-section stats only
SD_FLOOR_FRAC = 0.25        # sd never below this fraction of the cross-section sd
SOURCE_WEIGHT = {
    "sec": 1.0,
    "tier1": 0.9,      # Reuters, Bloomberg, WSJ, AP, CNBC, Dow Jones, Barron's, FT
    "financial": 0.7,  # MarketWatch, Yahoo, Nasdaq, Seeking Alpha news, Benzinga, Investopedia
    "blog": 0.5,       # Motley Fool, 24/7 Wall St, Zacks, InvestorPlace, SA contributor pieces
    "social": 0.4,
}
TIER1_PUBLISHERS = ("reuters", "bloomberg", "wall street journal", "wsj", "associated press",
                    "cnbc", "dow jones", "barron", "financial times", "ft.com")
FINANCIAL_PUBLISHERS = ("marketwatch", "yahoo", "nasdaq", "seeking alpha", "benzinga",
                        "investopedia", "investor's business daily", "thestreet", "fortune", "forbes")

# SPEC §2.2 grade thresholds on the -100..+100 scale (lower bounds of +1, +2, +3).
GRADE_THRESHOLDS = (10.0, 30.0, 60.0)
MIXED_AGREEMENT = 40.0
IN_FOCUS_Z = 2.0
HYPE_Z = 3.0

DEDUP_FUZZ = 90             # rapidfuzz token_set_ratio on titles
DEDUP_COSINE = 0.80         # MiniLM same-event threshold

# Gemini (SPEC §5.0): free tier, used only when GEMINI_API_KEY is set.
GEMINI_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
GEMINI_BATCH = 25
# OpenAI-compatible chat providers for the teacher prompt (sentiment.llm.OpenAICompatLabeler)
LLM_PROVIDERS = {
    "deepseek": {"base": "https://api.deepseek.com", "model": "deepseek-chat", "key_env": "DEEPSEEK_API_KEY"},
    "groq": {"base": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b", "key_env": "GROQ_API_KEY"},
}
GEMINI_MAX_ITEMS_PER_TICKER = 40
