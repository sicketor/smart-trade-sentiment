"""Data contracts shared by every module. See SPEC-he.md §2 and §3."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Channel = Literal["news", "filings", "social"]
Stance = Literal["bullish", "bearish", "none"]

# Value of each of the 5 item classes (-2..+2) on the -1..+1 score scale.
CLASS_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


@dataclass
class Company:
    ticker: str          # upper case, e.g. "NVDA"
    name: str            # SEC title, e.g. "NVIDIA CORP"
    short_name: str      # cleaned, e.g. "Nvidia"
    cik: int
    aliases: list[str] = field(default_factory=list)  # extra names ("Google" for GOOGL)
    ambiguous: bool = False  # ticker is a common word (AI, ALL, F, ON...): needs "$" or name
    contextual: bool = False  # ambiguous, but a bare ticker counts next to market words ("MA Stock", "vs. HD")
    weak_name: bool = False  # short_name is an everyday word ("Target", "NEWS"): alone it does not count


@dataclass
class RawItem:
    """One text item from a source, before any scoring."""
    id: str                  # sha1 of canonical url, or of source+text when no url
    source: str              # adapter id: "google_news", "yahoo_rss", "finviz", "nasdaq", "seekingalpha",
                             # "yahoo_search", "sec", "stocktwits", "reddit", "mastodon", "hn"
    channel: Channel
    publisher: str           # e.g. "Reuters", "r/wallstreetbets", "@user"
    title: str               # headline; for social posts the first 140 chars of the body
    text: str                # body/summary; may be "" for headline-only feeds
    url: str
    published_at: datetime   # timezone-aware UTC
    ticker: str              # the ticker it was collected for
    meta: dict = field(default_factory=dict)
    # meta keys used downstream (all optional):
    #   "user_label": "Bullish"|"Bearish"   (StockTwits self-label)
    #   "followers": int, "account_age_days": int, "likes": int
    #   "form": "8-K"|"4"|"10-Q"            (SEC)
    #   "symbols": list[str]                (other tickers tagged on the item)


@dataclass
class ModelOutput:
    """One model's opinion on one item, mapped to (neg, neu, pos)."""
    model: str
    probs: tuple[float, float, float]   # neg, neu, pos; sums to 1


@dataclass
class ScoredItem:
    item: RawItem
    relevance: float                    # 0..1 (SPEC §6.3)
    cluster_id: str                     # same-event cluster (SPEC §6.2)
    novelty: float                      # 1.0 first in cluster, then 1/k^1.5
    outputs: list[ModelOutput]          # every model that ran on it
    features: dict[str, float]          # lexicon / structure / catalyst features (SPEC §5.3)
    events: list[str]                   # catalyst tags (SPEC §2.3)
    probs5: tuple[float, float, float, float, float]  # calibrated, classes -2..+2
    score: float                        # sum(p_k * CLASS_VALUES[k]), -1..+1
    cls: int                            # argmax class, -2..+2
    confidence: float                   # max(probs5)
    stance: Stance
    llm: dict | None = None             # Gemini answer when used: {"cls","stance","event","reason"}
    drivers: list[str] = field(default_factory=list)  # short phrases that drove the score


@dataclass
class ChannelSnapshot:
    channel: Channel
    score: float | None          # display -100..+100: normalized when a norm exists, else == raw; None when coverage too low
    grade: int | None            # -3..+3 (percentile cuts of z with a norm, fixed thresholds without)
    agreement: float | None      # 0..100 over the non-neutral items
    n_items: int
    n_effective: float           # sum of weights
    momentum: float | None       # raw_24h - mean of daily raws over the window (raw scale)
    bull_ratio: float | None     # social only: bullish / (bullish + bearish) stance, 0..1
    raw: float | None = None     # shrunk non-neutral mean -100..+100 before normalization
    z: float | None = None       # (raw - mu) / sd against the blended history / cross-section stats
    pos_ratio: float | None = None   # weight share of positive among non-neutral items (>= 3 of them)
    bounds: list[float] | None = None  # the 6 display scores where the grade steps up (-3|-2|...|+3), for the gauge


@dataclass
class TickerSnapshot:
    ticker: str
    company: str
    as_of: datetime
    composite: ChannelSnapshot   # channel field = "news" is ignored for composite
    channels: dict[str, ChannelSnapshot]
    attention_z: float | None
    flags: list[str]             # "mixed", "in_focus", "hype", "low_data"
    top_bullish: list[ScoredItem]
    top_bearish: list[ScoredItem]
    events: dict[str, int]       # catalyst tag -> count in window
    context: dict                # stocktwits_bull_pct history, wiki_views, price...
    sources_ok: list[str]
    sources_failed: dict[str, str]   # source -> error message
    model_version: str
