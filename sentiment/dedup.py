"""Duplicate / same-event clustering (SPEC §6.2)."""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
from rapidfuzz import fuzz

from . import config
from .text import title_key
from .types import RawItem


def cluster(
    items: list[RawItem],
    embed: Callable[[list[str]], np.ndarray] | None,
) -> dict[str, tuple[str, float]]:
    """Map item.id -> (cluster_id, novelty). Items in different channels never share a cluster."""
    by_channel: dict[str, list[RawItem]] = {}
    for it in items:
        by_channel.setdefault(it.channel, []).append(it)
    for lst in by_channel.values():
        lst.sort(key=lambda i: (i.published_at, i.id))

    # one embed() call for all non-social titles
    vecs: dict[str, np.ndarray] = {}
    if embed is not None:
        emb_items = [i for ch, lst in by_channel.items() if ch != "social" for i in lst]
        if emb_items:
            arr = np.asarray(embed([i.title for i in emb_items]), dtype=float)
            vecs = {i.id: arr[n] for n, i in enumerate(emb_items)}

    result: dict[str, tuple[str, float]] = {}
    for ch, lst in by_channel.items():
        clusters: list[dict] = []
        for it in lst:
            key = title_key(it.title)
            nwords = len(key.split())
            target = None
            if key:
                for c in clusters:
                    if c["key"] == key:
                        target = c
                        break
            if target is None and nwords >= 4:
                best = -1.0
                for c in clusters:
                    if len(c["key"].split()) < 4:
                        continue
                    r = fuzz.token_set_ratio(key, c["key"])
                    if r >= config.DEDUP_FUZZ and r > best:
                        best, target = r, c
            if target is None and ch != "social" and it.id in vecs:
                best = -1.0
                for c in clusters:
                    if c["vec"] is None:
                        continue
                    sim = float(np.dot(vecs[it.id], c["vec"]))
                    if sim >= config.DEDUP_COSINE and sim > best:
                        best, target = sim, c
            if target is None:
                cid = hashlib.sha1(it.id.encode()).hexdigest()[:12]
                clusters.append({"id": cid, "key": key, "vec": vecs.get(it.id), "k": 1})
                result[it.id] = (cid, 1.0)
            else:
                target["k"] += 1
                result[it.id] = (target["id"], 1.0 / target["k"] ** 1.5)
    return result
