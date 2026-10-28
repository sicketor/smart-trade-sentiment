"""Ticker-level aggregation (SPEC §2.2, §8): channel scores with recency decay and Bayesian
shrinkage, 7 grades, agreement, momentum, bull ratio, attention and flags."""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

from . import config
from .types import ChannelSnapshot, ScoredItem

CHANNELS = ("news", "filings", "social")
DAY = timedelta(days=1)


def source_weight(si: ScoredItem) -> float:
    it = si.item
    if it.source == "sec":
        return config.SOURCE_WEIGHT["sec"]
    if it.channel == "social":
        return config.SOURCE_WEIGHT["social"]
    pub = it.publisher.lower()
    if any(p in pub for p in config.TIER1_PUBLISHERS):
        return config.SOURCE_WEIGHT["tier1"]
    if any(p in pub for p in config.FINANCIAL_PUBLISHERS):
        return config.SOURCE_WEIGHT["financial"]
    return config.SOURCE_WEIGHT["blog"]


def author_weight(si: ScoredItem) -> float:
    """Social only: 0.5..1 by log-followers; accounts younger than 30 days count half."""
    if si.item.channel != "social":
        return 1.0
    w = 0.5 + 0.5 * si.features.get("author_weight", 0.0)
    age = si.item.meta.get("account_age_days")
    if age is not None and age < 30:
        w *= 0.5
    return w


def item_weight(si: ScoredItem, as_of: datetime) -> float:
    """w_i = source · relevance · novelty · confidence · author · 2^(−Δt/halflife)."""
    hours = max(0.0, (as_of - si.item.published_at).total_seconds() / 3600)
    decay = 2 ** (-hours / config.HALF_LIFE_HOURS[si.item.channel])
    return (source_weight(si) * si.relevance * si.novelty * si.confidence
            * author_weight(si) * decay)


def grade(score: float | None) -> int | None:
    if score is None:
        return None
    t1, t2, t3 = config.GRADE_THRESHOLDS
    a = abs(score)
    g = 3 if a >= t3 else 2 if a >= t2 else 1 if a >= t1 else 0
    return g if score >= 0 else -g


def _weighted(items: list[ScoredItem], as_of: datetime, channel: str) -> tuple[float | None, float, float | None]:
    """(shrunk score -100..100 or None, Σw, agreement 0..100 or None)."""
    if not items:
        return None, 0.0, None
    w = np.array([item_weight(si, as_of) for si in items])
    s = np.array([si.score for si in items])
    sw = float(w.sum())
    if sw <= 1e-9:
        return None, sw, None
    score = 100.0 * float((w * s).sum()) / (sw + config.SHRINK_K[channel])
    mean = float((w * s).sum() / sw)
    std = math.sqrt(float((w * (s - mean) ** 2).sum() / sw))
    agreement = max(0.0, min(100.0, 100.0 * (1.0 - std)))
    return score, sw, agreement


def channel_snapshot(channel: str, items: list[ScoredItem], as_of: datetime) -> ChannelSnapshot:
    score, n_eff, agreement = _weighted(items, as_of, channel)
    if n_eff < config.MIN_EFFECTIVE_ITEMS:
        score, agreement = None, None
    momentum = None
    if score is not None:
        # score of the last 24h minus the mean of the daily scores of the window (days with data).
        recent = [si for si in items if as_of - si.item.published_at <= DAY]
        s24, _, _ = _weighted(recent, as_of, channel)
        daily = []
        days = max(1, math.ceil(max((as_of - si.item.published_at for si in items), default=DAY) / DAY))
        for d in range(days):
            day_items = [si for si in items
                         if d * DAY <= max(as_of - si.item.published_at, timedelta(0)) < (d + 1) * DAY]
            if day_items:   # each day judged at its own end, so older days are not decayed away
                ds, _, _ = _weighted(day_items, as_of - d * DAY, channel)
                if ds is not None:
                    daily.append(ds)
        if s24 is not None and len(daily) >= 2:
            momentum = round(s24 - float(np.mean(daily)), 1)
    bull_ratio = None
    if channel == "social":
        bulls = sum(si.stance == "bullish" for si in items)
        bears = sum(si.stance == "bearish" for si in items)
        if bulls + bears >= 3:
            bull_ratio = round(bulls / (bulls + bears), 3)
    return ChannelSnapshot(
        channel=channel,
        score=None if score is None else round(score, 1),
        grade=grade(score),
        agreement=None if agreement is None else round(agreement, 1),
        n_items=len(items),
        n_effective=round(n_eff, 3),
        momentum=momentum,
        bull_ratio=bull_ratio,
    )


def attention_z(daily: list[float] | None) -> float | None:
    """Latest day vs the earlier days (oldest -> newest), Poisson floor on the spread.
    Fed by Wikipedia page views or by our own snapshot history, never by the collected items:
    most sources return only their latest N items, so item counts always peak in the last 24h."""
    if not daily or len(daily) < 8:
        return None
    prior, last = daily[:-1], daily[-1]
    mu = float(np.mean(prior))
    sd = max(float(np.std(prior)), math.sqrt(max(mu, 1.0)))
    return round((last - mu) / sd, 2)


def composite(channels: dict[str, ChannelSnapshot], hype: bool) -> ChannelSnapshot:
    alpha = dict(config.CHANNEL_WEIGHTS)
    if hype:
        alpha["social"] *= 0.5
    have = [c for c in CHANNELS if channels[c].score is not None]
    total = sum(alpha[c] for c in have)

    def mix(attr: str) -> float | None:
        vals = [(alpha[c], getattr(channels[c], attr)) for c in have if getattr(channels[c], attr) is not None]
        if not vals:
            return None
        return round(sum(a * v for a, v in vals) / sum(a for a, _ in vals), 1)

    score = round(sum(alpha[c] * channels[c].score for c in have) / total, 1) if have else None
    return ChannelSnapshot(
        channel="news",   # ignored for the composite
        score=score,
        grade=grade(score),
        agreement=mix("agreement"),
        n_items=sum(ch.n_items for ch in channels.values()),
        n_effective=round(sum(ch.n_effective for ch in channels.values()), 3),
        momentum=mix("momentum"),
        bull_ratio=channels["social"].bull_ratio,
    )


def summarize(items: list[ScoredItem], as_of: datetime, context: dict | None = None,
              history: dict[str, list[float]] | None = None) -> dict:
    """Everything TickerSnapshot needs except identity, context and sources.
    history: daily item counts per channel from stored snapshots, oldest -> newest (phase 3)."""
    by_ch = {c: [si for si in items if si.item.channel == c] for c in CHANNELS}
    channels = {c: channel_snapshot(c, by_ch[c], as_of) for c in CHANNELS}
    views = (context or {}).get("wiki_views") or []
    att = attention_z([float(v["views"]) for v in views])
    history = history or {}
    social_z = attention_z(history.get("social"))
    news_z = attention_z(history.get("news"))
    hype = social_z is not None and social_z >= config.HYPE_Z and (news_z is None or news_z < 1.0)
    comp = composite(channels, hype)

    flags = []
    if comp.score is None:
        flags.append("low_data")
    if comp.agreement is not None and comp.agreement < config.MIXED_AGREEMENT:
        flags.append("mixed")
    if att is not None and att >= config.IN_FOCUS_Z:
        flags.append("in_focus")
    if hype:
        flags.append("hype")

    # Top lists only from items clearly about the ticker (aggregator fallbacks are 0.5).
    ranked = sorted((si for si in items if si.relevance >= 0.6), key=lambda si: item_weight(si, as_of) * si.score)
    top_bear = [si for si in ranked[:5] if si.score < -0.1]
    top_bull = [si for si in reversed(ranked[-5:]) if si.score > 0.1]

    events: Counter[str] = Counter()
    seen: set[str] = set()
    for si in items:
        if si.cluster_id in seen:
            continue
        seen.add(si.cluster_id)
        events.update(t for t in si.events if t != "OTHER")

    return {"composite": comp, "channels": channels, "attention_z": att, "flags": flags,
            "top_bullish": top_bull, "top_bearish": top_bear, "events": dict(events.most_common())}
