"""kNN few-shot pool for the Gemini v3 prompt: per batch of items, gold examples that resemble them.

The pool is written by `python -m train.fewshot --pool` (gold train buckets only).
"""
from __future__ import annotations

import json

import numpy as np

from . import config

POOL = config.ROOT / "sentiment" / "fewshot_pool.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TEXT = 220
NEAR_DUP = 0.95
CHANNELS = ("news", "social")
CLASSES = (-2, -1, 0, 1, 2)


def example_text(item: dict) -> str:
    title, text = (item.get("title") or "").strip(), (item.get("text") or "").strip()
    if not title or (text and text.startswith(title[:40])):
        s = text or title
    else:
        s = f"{title} | {text}" if text else title
    s = " ".join(s.split())
    return s if len(s) <= MAX_TEXT else s[: MAX_TEXT - 3] + "..."


class ShotPool:
    def __init__(self, entries: list[dict], embed=None):
        self.entries = list(entries)
        self._embed = embed
        self._model = None
        self._emb: np.ndarray | None = None

    @classmethod
    def load(cls, path=POOL, embed=None) -> "ShotPool | None":
        try:
            entries = json.loads(open(path, encoding="utf-8").read())
        except (OSError, ValueError):
            return None
        if not isinstance(entries, list) or not entries:
            return None
        if not all(isinstance(e, dict) and {"id", "ticker", "channel", "text", "cls"} <= e.keys() for e in entries):
            return None
        return cls(entries, embed=embed)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._embed is not None:
            return np.asarray(self._embed(texts))
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model.encode(texts, normalize_embeddings=True, batch_size=64)

    def select(self, items: list, k_per_item: int = 2, cap: int = 16) -> list[dict]:
        if self._emb is None:
            self._emb = self._encode([e["text"] for e in self.entries])
        chosen: list[int] = []
        if items:
            item_emb = self._encode([example_text({"title": it.title, "text": it.text}) for it in items])
            sims = np.asarray(item_emb) @ self._emb.T
            for it, row in zip(items, sims):
                if len(chosen) >= cap:
                    break
                taken = 0
                for j in np.argsort(-row, kind="stable"):
                    j = int(j)
                    if taken >= k_per_item or len(chosen) >= cap:
                        break
                    if self.entries[j]["id"] == it.id or row[j] > NEAR_DUP or j in chosen:
                        continue
                    chosen.append(j)
                    taken += 1
        item_ids = {it.id for it in items}
        covered = {(self.entries[j]["channel"], self.entries[j]["cls"]) for j in chosen}
        anchors: list[int] = []
        for ch in CHANNELS:
            for c in CLASSES:
                if (ch, c) in covered:
                    continue
                for j, e in enumerate(self.entries):
                    if e["channel"] == ch and e["cls"] == c and e["id"] not in item_ids:
                        anchors.append(j)
                        break
        return [self.entries[j] for j in chosen + anchors]

    def block(self, items: list) -> str:
        exs = self.select(items)
        if not exs:
            return ""
        lines = [f'- [{e["channel"]}] target {e["ticker"]}: {json.dumps(e["text"], ensure_ascii=False)} -> cls {e["cls"]}'
                 for e in exs]
        return ("Examples labeled by our analyst, chosen to resemble these items (apply the same standard):\n"
                + "\n".join(lines) + "\n")
