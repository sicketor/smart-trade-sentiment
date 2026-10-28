"""Ticker-level aggregation (SPEC §2.2, §8): channel scores with recency decay and Bayesian
shrinkage over the non-neutral items, z normalization (own history blended with cross-section
stats) with percentile grades, agreement, momentum, bull/positive ratios, attention and flags."""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import config
from .types import ChannelSnapshot, ScoredItem

log = logging.getLogger(__name__)

CHANNELS = ("news", "filings", "social")
DAY = timedelta(days=1)
CUT_PERCENTILES = (5, 15, 35, 65, 85, 95)

# Headlines that only report the price move ("shares rise 5% today") carry no new information:
# the move is the market's reaction, so they are dropped unless a catalyst tag explains it.
_MOVE_VERB = (r"rise[sn]?|rose|jump(s|ed)?|soar(s|ed)?|surge[sd]?|climb(s|ed)?|rall(y|ies|ied)|gain(s|ed)?"
              r"|fall(s|en)?|fell|drop(s|ped)?|slide[sd]?|slid|sink(s)?|sank|tumble[sd]?|plunge[sd]?"
              r"|slump(s|ed)?|dip(s|ped)?|(trade|trading|move|moving) (higher|lower)|(is|are) (up|down)")
_MOVE_WHEN = (r"\d+(\.\d+)?\s?%|\btoday\b|\bthis week\b|\bon (monday|tuesday|wednesday|thursday|friday)\b"
              r"|\bpremarket\b|\bafter[- ]hours\b")
PRICE_MOVE_RE = re.compile(
    rf"\b(shares?|stocks?)\b.{{0,40}}\b({_MOVE_VERB})\b.{{0,30}}({_MOVE_WHEN})"
    r"|\bwhy\b.{0,60}\b(stock|shares)\b.{0,30}\b(is|are|was|were|has|have)\b.{0,20}"
    r"\b(up|down|higher|lower|soar\w*|sink\w*|fall\w*|ris\w*|jump\w*|plung\w*|surg\w*|tumbl\w*|crash\w*"
    r"|rall\w*|slid\w*|climb\w*)\b"
    r"|\b(stock|shares)\b.{0,15}\b(is|are)\s+(up|down)\b",
    re.IGNORECASE,
)

_norm_cache: dict[Path, dict] = {}


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


def price_move_only(si: ScoredItem) -> bool:
    """News headline that only reports the stock's move, with no catalyst tag behind it."""
    if si.item.channel != "news":
        return False
    if any(e != "OTHER" for e in si.events):
        return False
    return PRICE_MOVE_RE.search(si.item.title or "") is not None


def item_weight(si: ScoredItem, as_of: datetime) -> float:
    """w_i = source · relevance² · novelty² · confidence · author · 2^(−Δt/halflife); 0 for price-move-only."""
    if price_move_only(si):
        return 0.0
    hours = max(0.0, (as_of - si.item.published_at).total_seconds() / 3600)
    decay = 2 ** (-hours / config.HALF_LIFE_HOURS[si.item.channel])
    return (source_weight(si) * si.relevance ** 2 * si.novelty ** 2 * si.confidence
            * author_weight(si) * decay)


def grade(score: float | None) -> int | None:
    """Fixed-threshold grade on the raw -100..+100 scale (used when no norm is available)."""
    if score is None:
        return None
    t1, t2, t3 = config.GRADE_THRESHOLDS
    a = abs(score)
    g = 3 if a >= t3 else 2 if a >= t2 else 1 if a >= t1 else 0
    return g if score >= 0 else -g


def grade_from_cuts(z: float, cuts: list[float]) -> int:
    """-3..+3 from the 5/15/35/65/85/95 percentile cuts of the cross-sectional z."""
    for i, c in enumerate(cuts):
        if z < c:
            return i - 3
    return 3


def percentile_cuts(zs: list[float]) -> list[float]:
    """Grade cuts for the norm file (used by the backfill script)."""
    if not zs:
        raise ValueError("percentile_cuts needs at least one value")
    return [float(v) for v in np.percentile(np.asarray(zs, dtype=float), CUT_PERCENTILES)]


def load_norm(path: Path | None = None) -> dict | None:
    """Cross-section stats written by the backfill; None when missing or unreadable (old behavior).
    A loaded file is cached by path; a missing one is re-checked on the next call."""
    p = Path(path) if path is not None else config.AGG_NORM_PATH
    if p in _norm_cache:
        return _norm_cache[p]
    if not p.is_file():
        return None
    try:
        norm = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("aggregate norm unreadable (%s): %s", p, e)
        return None
    if not isinstance(norm, dict) or not isinstance(norm.get("groups"), dict):
        log.warning("aggregate norm has no groups: %s", p)
        return None
    _norm_cache[p] = norm
    return norm


def _valid_stats(stats: dict | None) -> bool:
    if not stats:
        return False
    try:
        sd = float(stats["sd"])
        float(stats["mu"])
        cuts = stats["cuts"]
    except (KeyError, TypeError, ValueError):
        return False
    return sd > 0 and isinstance(cuts, list) and len(cuts) == len(CUT_PERCENTILES)


def normalize(raw: float | None, stats: dict | None,
              hist: list[float] | None) -> tuple[float | None, float | None, int | None]:
    """(display -100..100, z, grade). The ticker's own history (previous days) blends into the
    cross-section mu/sd with weight n/(n+HIST_PRIOR_DAYS) once it has HIST_MIN_DAYS days.
    Without valid stats the raw score is shown with the fixed-threshold grades."""
    if raw is None:
        return None, None, None
    if not _valid_stats(stats):
        return round(raw, 1), None, grade(raw)
    mu_c, sd_c = float(stats["mu"]), float(stats["sd"])
    h = [float(x) for x in hist or [] if x is not None]
    n = len(h)
    lam = n / (n + config.HIST_PRIOR_DAYS) if n >= config.HIST_MIN_DAYS else 0.0
    if lam > 0:
        mu = lam * float(np.mean(h)) + (1 - lam) * mu_c
        var = lam * float(np.var(h)) + (1 - lam) * sd_c ** 2
    else:
        mu, var = mu_c, sd_c ** 2
    sd = max(math.sqrt(var), config.SD_FLOOR_FRAC * sd_c)
    z = (raw - mu) / sd
    cuts = [float(c) for c in stats["cuts"]]
    return round(100.0 * math.tanh(z / 2), 1), round(z, 3), grade_from_cuts(z, cuts)


def _weighted(items: list[ScoredItem], as_of: datetime,
              channel: str) -> tuple[float | None, float, float | None, float | None]:
    """(shrunk raw -100..100 or None, Σw, agreement 0..100 or None, positive ratio or None).
    Coverage counts every item; the score averages only the non-neutral ones."""
    if not items:
        return None, 0.0, None, None
    w = np.array([item_weight(si, as_of) for si in items])
    n_eff = float(w.sum())
    active = w > 0
    if int(active.sum()) < config.MIN_ITEMS or n_eff < config.MIN_EFFECTIVE_ITEMS:
        return None, n_eff, None, None
    cls = np.array([si.cls for si in items])
    nn = active & (cls != 0)
    if not nn.any():
        return 0.0, n_eff, 100.0, None
    wn = w[nn]
    s = np.array([si.score for si in items])[nn]
    swn = float(wn.sum())
    raw = 100.0 * float((wn * s).sum()) / (swn + config.SHRINK_K[channel])
    mean = float((wn * s).sum() / swn)
    std = math.sqrt(float((wn * (s - mean) ** 2).sum() / swn))
    agreement = max(0.0, min(100.0, 100.0 * (1.0 - std)))
    pos_ratio = round(float(wn[cls[nn] > 0].sum()) / swn, 3) if int(nn.sum()) >= 3 else None
    return raw, n_eff, agreement, pos_ratio


def channel_snapshot(channel: str, items: list[ScoredItem], as_of: datetime,
                     stats: dict | None = None, hist: list[float] | None = None) -> ChannelSnapshot:
    raw, n_eff, agreement, pos_ratio = _weighted(items, as_of, channel)
    momentum = None
    if raw is not None:
        # raw of the last 24h minus the mean of the daily raws of the window (days with data).
        recent = [si for si in items if as_of - si.item.published_at <= DAY]
        s24 = _weighted(recent, as_of, channel)[0]
        daily = []
        days = max(1, math.ceil(max((as_of - si.item.published_at for si in items), default=DAY) / DAY))
        for d in range(days):
            day_items = [si for si in items
                         if d * DAY <= max(as_of - si.item.published_at, timedelta(0)) < (d + 1) * DAY]
            if day_items:   # each day judged at its own end, so older days are not decayed away
                ds = _weighted(day_items, as_of - d * DAY, channel)[0]
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
    display, z, g = normalize(raw, stats, hist)
    return ChannelSnapshot(
        channel=channel,
        score=display,
        grade=g,
        agreement=None if agreement is None else round(agreement, 1),
        n_items=len(items),
        n_effective=round(n_eff, 3),
        momentum=momentum,
        bull_ratio=bull_ratio,
        raw=None if raw is None else round(raw, 1),
        z=z,
        pos_ratio=pos_ratio,
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


def composite(channels: dict[str, ChannelSnapshot], hype: bool,
              stats: dict | None = None, hist: list[float] | None = None) -> ChannelSnapshot:
    """Alpha-weighted mix of the channel RAW scores, then normalized like a channel."""
    alpha = dict(config.CHANNEL_WEIGHTS)
    if hype:
        alpha["social"] *= 0.5
    have = [c for c in CHANNELS if channels[c].raw is not None]

    def mix(attr: str, digits: int = 1) -> float | None:
        vals = [(alpha[c], getattr(channels[c], attr)) for c in have if getattr(channels[c], attr) is not None]
        if not vals:
            return None
        return round(sum(a * v for a, v in vals) / sum(a for a, _ in vals), digits)

    raw = mix("raw")
    display, z, g = normalize(raw, stats, hist)
    return ChannelSnapshot(
        channel="news",   # ignored for the composite
        score=display,
        grade=g,
        agreement=mix("agreement"),
        n_items=sum(ch.n_items for ch in channels.values()),
        n_effective=round(sum(ch.n_effective for ch in channels.values()), 3),
        momentum=mix("momentum"),
        bull_ratio=channels["social"].bull_ratio,
        raw=raw,
        z=z,
        pos_ratio=mix("pos_ratio", 3),
    )


def _count_z(history: dict[str, list[float]], channel: str, today: int) -> float | None:
    """Attention z of today's item count vs the previous days in the snapshot log."""
    prior = history.get(channel)
    if prior is None:
        return None
    return attention_z([float(x) for x in prior if x is not None] + [float(today)])


def summarize(items: list[ScoredItem], as_of: datetime, context: dict | None = None,
              history: dict[str, list[float]] | None = None, norm: dict | None = None) -> dict:
    """Everything TickerSnapshot needs except identity, context and sources.
    history (snapshot log, previous days only, oldest -> newest): "news"/"filings"/"social" daily
    item counts and "raw_<channel>"/"raw_composite" daily raw scores. norm: see load_norm."""
    history = history or {}
    groups = (norm or {}).get("groups") or {}
    by_ch = {c: [si for si in items if si.item.channel == c] for c in CHANNELS}
    channels = {c: channel_snapshot(c, by_ch[c], as_of, groups.get(c), history.get(f"raw_{c}"))
                for c in CHANNELS}
    views = (context or {}).get("wiki_views") or []
    att = attention_z([float(v["views"]) for v in views])
    social_z = _count_z(history, "social", channels["social"].n_items)
    news_z = _count_z(history, "news", channels["news"].n_items)
    hype = social_z is not None and social_z >= config.HYPE_Z and (news_z is None or news_z < 1.0)
    comp = composite(channels, hype, groups.get("composite"), history.get("raw_composite"))

    flags = []
    if comp.score is None:
        flags.append("low_data")
    if comp.agreement is not None and comp.agreement < config.MIXED_AGREEMENT:
        flags.append("mixed")
    if att is not None and att >= config.IN_FOCUS_Z:
        flags.append("in_focus")
    if hype:
        flags.append("hype")

    # Top lists only from items clearly about the ticker (aggregator fallbacks are 0.5);
    # price-move-only headlines (weight 0) are left out.
    ranked = sorted((si for si in items if si.relevance >= 0.6 and not price_move_only(si)),
                    key=lambda si: item_weight(si, as_of) * si.score)
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
