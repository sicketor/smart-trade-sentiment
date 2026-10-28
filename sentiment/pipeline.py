"""End-to-end analysis of one or more tickers (SPEC §4):
collect -> normalize -> relevance -> dedup -> features -> models -> meta (+Gemini) -> aggregate."""
from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from . import __version__, config
from .aggregate import item_weight, load_norm, summarize
from .dedup import cluster
from .features import extract
from .http import Http
from .linking import relevance
from .meta import Meta
from .sources import collect
from .sources.context import fetch_context
from .text import normalize
from .tickers import get_company
from .types import Company, RawItem, ScoredItem, TickerSnapshot

log = logging.getLogger(__name__)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NOT_RELEVANT_FACTOR = 0.3   # Gemini says "not about this ticker" -> keep, but mostly ignore


class Engine:
    """Holds the heavy objects (HTTP client, models, embedder) across tickers."""

    def __init__(self, use_llm: bool = True, device: str = "auto"):
        from .models.runner import ModelRunner
        self.http = Http()
        self.runner = ModelRunner(device=device)
        self.meta = Meta()
        self.llm = None
        if use_llm:
            from .llm import GeminiLabeler
            labeler = GeminiLabeler()
            self.llm = labeler if labeler.available else None
        self._embedder = None
        self.norm = load_norm()
        if self.norm is not None and self.norm.get("meta_version") != self.meta.version:
            # raw scores shift with the meta model, so a norm built on another one would mis-grade
            log.warning("aggregate norm was built on %s, meta is %s: normalization off until the backfill reruns",
                        self.norm.get("meta_version"), self.meta.version)
            self.norm = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL, device=self.runner.device)
        return self._embedder.encode(texts, normalize_embeddings=True, batch_size=64)

    @property
    def model_version(self) -> str:
        v = f"{__version__}/{self.meta.version}" + ("+gemini" if self.llm else "")
        if self.norm is not None:
            v += f"+norm{self.norm.get('version', '?')}"
        return v

    def analyze(self, ticker: str, days: int = config.LOOKBACK_DAYS, only: list[str] | None = None,
                history: dict[str, list[float]] | None = None,
                as_of: datetime | None = None) -> tuple[TickerSnapshot, list[ScoredItem]]:
        """history: the ticker's previous-day snapshot log (sentiment.snapshots.history), read at the same as_of."""
        company = get_company(ticker, self.http)
        if company is None:
            raise ValueError(f"unknown US ticker: {ticker}")
        as_of = as_of or datetime.now(timezone.utc)
        raw, ok, failed = collect(company, self.http, as_of - timedelta(days=days), only=only)
        items = self._score(company, raw)
        try:
            context = fetch_context(company, self.http)
        except Exception as e:   # context is optional decoration
            log.warning("context failed for %s: %s", ticker, e)
            context = {}
        s = summarize(items, as_of, context, history=history, norm=self.norm)
        return TickerSnapshot(
            ticker=company.ticker, company=company.short_name, as_of=as_of,
            composite=s["composite"], channels=s["channels"], attention_z=s["attention_z"],
            flags=s["flags"], top_bullish=s["top_bullish"], top_bearish=s["top_bearish"],
            events=s["events"], context=context, sources_ok=ok, sources_failed=failed,
            model_version=self.model_version,
        ), items

    def _score(self, company: Company, raw: list[RawItem]) -> list[ScoredItem]:
        clean = [dataclasses.replace(it, title=normalize(it.title), text=normalize(it.text)) for it in raw]
        rel = {it.id: relevance(it, company) for it in clean}
        clean = [it for it in clean if rel[it.id] > 0 and it.title]
        if not clean:
            return []
        clusters = cluster(clean, self.embed)
        feats = {it.id: extract(it, company) for it in clean}
        outputs: dict[str, list] = {}
        for ch in ("news", "filings", "social"):
            group = [it for it in clean if it.channel == ch]
            if group:
                outputs.update(self.runner.score_items(group, company, ch))

        def build(it: RawItem, llm: dict | None) -> ScoredItem:
            f, ev, drivers = feats[it.id]
            if llm and llm.get("event") and llm["event"] != "OTHER" and llm["event"] not in ev:
                ev = [e for e in ev if e != "OTHER"] + [llm["event"]]
            r = self.meta.predict(it.channel, outputs.get(it.id, []), f, ev, llm)
            relevance_ = rel[it.id] * (LLM_NOT_RELEVANT_FACTOR if llm and not llm.get("relevant", True) else 1.0)
            cid, nov = clusters.get(it.id, (it.id[:12], 1.0))
            return ScoredItem(item=it, relevance=round(relevance_, 3), cluster_id=cid, novelty=round(nov, 4),
                              outputs=outputs.get(it.id, []), features=f, events=ev, probs5=r.probs5,
                              score=r.score, cls=r.cls, confidence=r.confidence, stance=r.stance,
                              llm=llm, drivers=drivers)

        scored = [build(it, None) for it in clean]
        if self.llm is not None:
            scored = self._with_llm(company, scored, build)
        return scored

    def _with_llm(self, company: Company, scored: list[ScoredItem], build) -> list[ScoredItem]:
        """Send the items that move the score most (weight x |score|, boosted by model disagreement)
        to Gemini, then rescore them with its answer."""
        as_of = datetime.now(timezone.utc)

        def priority(si: ScoredItem) -> float:
            dis = float(np.std([o.probs[2] - o.probs[0] for o in si.outputs])) if len(si.outputs) > 1 else 0.0
            return item_weight(si, as_of) * (0.2 + abs(si.score)) * (1.0 + 2.0 * dis)

        firsts = [si for si in scored if si.novelty >= 1.0]   # one per event cluster
        chosen = sorted(firsts, key=priority, reverse=True)[: config.GEMINI_MAX_ITEMS_PER_TICKER]
        try:
            answers = self.llm.label(company, [si.item for si in chosen])
        except Exception as e:
            log.warning("gemini failed: %s", type(e).__name__)
            return scored
        return [build(si.item, answers[si.item.id]) if si.item.id in answers else si for si in scored]


def snapshot_to_dict(snap: TickerSnapshot, items: list[ScoredItem] | None = None) -> dict:
    def ch(c) -> dict:
        return dataclasses.asdict(c)

    d = {
        "ticker": snap.ticker, "company": snap.company, "as_of": snap.as_of.isoformat(),
        "composite": {k: v for k, v in ch(snap.composite).items() if k != "channel"},
        "channels": {k: ch(v) for k, v in snap.channels.items()},
        "attention_z": snap.attention_z, "flags": snap.flags, "events": snap.events,
        "top_bullish": [item_to_dict(si) for si in snap.top_bullish],
        "top_bearish": [item_to_dict(si) for si in snap.top_bearish],
        "context": snap.context, "sources_ok": snap.sources_ok, "sources_failed": snap.sources_failed,
        "model_version": snap.model_version,
    }
    if items is not None:
        d["items"] = [item_to_dict(si, full=True) for si in
                      sorted(items, key=lambda s: s.item.published_at, reverse=True)]
    return d


def item_to_dict(si: ScoredItem, full: bool = False) -> dict:
    it = si.item
    d = {
        "id": it.id, "source": it.source, "channel": it.channel, "publisher": it.publisher,
        "title": it.title, "url": it.url, "published_at": it.published_at.isoformat(),
        "cls": si.cls, "score": si.score, "confidence": si.confidence, "relevance": si.relevance,
        "stance": si.stance, "events": si.events, "drivers": si.drivers,
        "reason": si.llm.get("reason") if si.llm else None,
    }
    if full:
        d |= {"probs5": si.probs5, "novelty": si.novelty, "cluster_id": si.cluster_id,
              "models": {o.model: o.probs for o in si.outputs}}
    return d
