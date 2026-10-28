"""Five public financial-sentiment datasets in one format (SPEC-he.md §9)."""
from __future__ import annotations

import ast
import hashlib
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

DATASETS: tuple[str, ...] = ("tfns", "sentfin", "finentity", "fiqa", "fpb")
_STR_LABEL = {"negative": -1, "neutral": 0, "positive": 1, "bearish": -1, "bullish": 1}


@dataclass(frozen=True)
class Example:
    id: str
    dataset: str
    text: str
    target: str | None
    channel: str          # "news" | "social"
    label3: int           # -1, 0, 1
    label5: int | None    # -2..2, only for fiqa


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def fiqa_label5(s: float) -> int:
    if s <= -0.5:
        return -2
    if s < -0.1:
        return -1
    if s <= 0.1:
        return 0
    if s < 0.5:
        return 1
    return 2


def _hf(path: str, split: str):
    from datasets import load_dataset
    return load_dataset(path, split=split)


def _lab(v, names=None) -> int:
    if isinstance(v, str):
        return _STR_LABEL[v.strip().lower()]
    return _STR_LABEL[names[v].lower()]


def _tfns(split):
    ds = _hf("zeroshot/twitter-financial-news-sentiment", "train" if split == "train" else "validation")
    m = {0: -1, 1: 1, 2: 0}
    return [Example(f"tfns:{split}:{i}", "tfns", r["text"], None, "news", m[int(r["label"])], None)
            for i, r in enumerate(ds)]


def _sentfin(split):
    ds = _hf("tasksource/sentfin", "train" if split == "train" else "test")
    return [Example(f"sentfin:{split}:{i}", "sentfin", r["sentence1"], r["sentence2"], "news",
                    _lab(r["labels"]), None) for i, r in enumerate(ds)]


def _finentity(split):
    ds = _hf("yixuantt/FinEntity", "train")
    out = []
    for i, r in enumerate(ds):
        is_test = int(hashlib.sha1(str(i).encode()).hexdigest(), 16) % 5 == 0
        if is_test != (split == "test"):
            continue
        ann = r["annotations"]
        if isinstance(ann, str):
            ann = ast.literal_eval(ann)
        for j, a in enumerate(ann or []):
            lab = a.get("label", a.get("tag"))
            out.append(Example(f"finentity:{split}:{i}:{j}", "finentity", r["content"], a["value"],
                               "news", _lab(lab), None))
    return out


def _fiqa(split):
    splits = ["train", "validation"] if split == "train" else ["test"]
    out = []
    for sp in splits:
        for i, r in enumerate(_hf("pauri32/fiqa-2018", sp)):
            l5 = fiqa_label5(float(r["sentiment_score"]))
            out.append(Example(f"fiqa:{sp}:{i}", "fiqa", r["sentence"], r["target"],
                               "social" if r["format"] == "post" else "news",
                               (l5 > 0) - (l5 < 0), l5))
    return out


def _fpb(split):
    splits = ["train", "validation"] if split == "train" else ["test"]
    out = []
    for sp in splits:
        ds = _hf("atrost/financial_phrasebank", sp)
        names = ds.features["label"].names
        out += [Example(f"fpb:{sp}:{i}", "fpb", r["sentence"], None, "news", _lab(r["label"], names), None)
                for i, r in enumerate(ds)]
    return out


_LOADERS = {"tfns": _tfns, "sentfin": _sentfin, "finentity": _finentity, "fiqa": _fiqa, "fpb": _fpb}


def load(name: str, split: Literal["train", "test"], limit: int | None = None, seed: int = 0) -> list[Example]:
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    data = _LOADERS[name](split)
    if limit is not None and limit < len(data):
        data = random.Random(seed).sample(data, limit)
    return data


@lru_cache(maxsize=1)
def test_texts() -> set[str]:
    return {normalize(e.text) for n in DATASETS for e in load(n, "test")}
