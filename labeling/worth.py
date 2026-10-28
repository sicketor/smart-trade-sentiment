"""'Worth labeling' model: learns from Tom's own skip / not-relevant decisions which candidates are likely
to be wasted time, so the labeling queue can prefer the useful ones. Pure module: the embedder is injected."""
from __future__ import annotations

from typing import Callable

import numpy as np

from sentiment.text import title_key
from train import io as tio

MIN_PER_CLASS = 50


def _ticker(row: dict) -> str:
    it = row.get("item") or {}
    return it.get("ticker") or (row.get("company") or {}).get("ticker", "") or ""


def text_key(row: dict) -> tuple[str, str]:
    it = row.get("item") or {}
    body = it.get("text", "") if it.get("channel") == "social" else it.get("title", "")
    k = title_key(body or "")
    if not k:   # Hebrew / emoji-only text: fall back to the raw text, then to the id
        k = " ".join((body or "").lower().split()) or str(it.get("id", ""))
    return (_ticker(row), k)


def is_waste(g: dict) -> bool:
    return bool(g.get("skipped")) or g.get("relevant", True) is False


class WorthModel:
    def __init__(self, embed: Callable[[list[str]], np.ndarray]):
        self.embed = embed
        self.ready = False
        self.n_train = 0
        self._clf = None
        self._cache: dict[tuple, np.ndarray] = {}
        self._teacher: dict = {}

    def _features(self, rows: list[dict]) -> np.ndarray:
        ck = lambda r: (r["item"]["id"], _ticker(r))
        todo = [r for r in rows if ck(r) not in self._cache]
        if todo:
            texts = []
            for r in todo:
                it = r["item"]
                texts.append(f"{_ticker(r)}: {it.get('title', '')}. {(it.get('text') or '')[:300]}")
            vecs = np.asarray(self.embed(texts), dtype=float)
            for r, v in zip(todo, vecs):
                self._cache[ck(r)] = v
        emb = np.vstack([self._cache[ck(r)] for r in rows])
        extras = []
        for r in rows:
            it = r["item"]
            t = self._teacher.get((it["id"], _ticker(r)))
            gem_off = 1.0 if t is not None and t.get("relevant", True) is False else 0.0
            gem_zero = 1.0 if t is not None and t.get("cls") == 0 else 0.0
            n_words = len(((it.get("title") or "") + " " + (it.get("text") or "")).split())
            extras.append([gem_off, gem_zero, 1.0 if t is not None else 0.0, float(r.get("relevance", 0) or 0),
                           1.0 if it.get("channel") == "social" else 0.0, min(n_words, 80) / 80])
        return np.hstack([emb, np.asarray(extras, dtype=float)])

    def fit(self, corpus_rows, gold_rows, teacher_rows) -> bool:
        from sklearn.linear_model import LogisticRegression

        self.ready = False
        self._teacher = {(t["id"], t.get("ticker")): t for t in tio.latest_teacher(teacher_rows or [])}
        latest: dict[str, dict] = {}
        for g in gold_rows:
            if "review" in g or "id" not in g:
                continue
            latest[g["id"]] = g
        corpus: dict[tuple, dict] = {}
        for r in corpus_rows:
            it = r.get("item") or {}
            corpus.setdefault((it.get("id"), it.get("ticker") or (r.get("company") or {}).get("ticker", "")), r)
        rows, y = [], []
        for iid, g in latest.items():
            r = corpus.get((iid, g.get("ticker")))
            if r is None:
                continue
            rows.append(r)
            y.append(1 if is_waste(g) else 0)
        y = np.asarray(y)
        if int((y == 1).sum()) < MIN_PER_CLASS or int((y == 0).sum()) < MIN_PER_CLASS:
            return False
        X = self._features(rows)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        clf.fit(X, y)
        self._clf = clf
        self.n_train = len(rows)
        self.ready = True
        return True

    def score(self, rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.zeros(0)
        if not self.ready or self._clf is None:
            return np.ones(len(rows))
        return 1.0 - self._clf.predict_proba(self._features(rows))[:, 1]
