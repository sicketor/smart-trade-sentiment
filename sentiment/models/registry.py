"""Catalog of the off-the-shelf models. Label order verified against each config.id2label."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .. import config

NEWS = frozenset({"news", "filings"})
ALL = frozenset({"news", "filings", "social"})
SOCIAL = frozenset({"social"})


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    channels: frozenset[str]
    kind: Literal["cls", "nli"]
    labels: tuple[str, ...]  # per class index, each "neg" | "neu" | "pos"
    input: Literal["plain", "social", "pair", "target"]   # target: ("TICKER | Company", text)
    max_len: int = 256
    version: str = ""        # part of the cache key (local models: VERSION file, else weights mtime)
    cache_id: str = ""       # cache-key identity instead of hf_id (local models: path-independent)


SPECS: dict[str, ModelSpec] = {s.name: s for s in (
    ModelSpec("finbert", "ProsusAI/finbert", NEWS, "cls", ("pos", "neg", "neu"), "plain"),
    ModelSpec("fin_distilroberta", "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
              NEWS, "cls", ("neg", "neu", "pos"), "plain"),
    ModelSpec("finbert_tone", "yiyanghkust/finbert-tone", NEWS, "cls", ("neu", "pos", "neg"), "plain"),
    ModelSpec("absa", "yangheng/deberta-v3-base-absa-v1.1", ALL, "cls", ("neg", "neu", "pos"), "pair"),
    ModelSpec("nli", "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33", ALL, "nli", (), "pair"),
    ModelSpec("fintwit", "StephanAkkerman/FinTwitBERT-sentiment", SOCIAL, "cls",
              ("neu", "pos", "neg"), "social"),
    ModelSpec("stocktwits_rb", "zhayunduo/roberta-base-stocktwits-finetuned", SOCIAL, "cls",
              ("neg", "pos"), "social"),
    ModelSpec("twitter_rb", "cardiffnlp/twitter-roberta-base-sentiment-latest", SOCIAL, "cls",
              ("neg", "neu", "pos"), "social"),
)}


STUDENT_DIR = "st-finsent"            # torch weights (*.safetensors)
STUDENT_ONNX_DIR = "st-finsent-onnx"  # the same student exported to model.onnx (train/export.py)


def student_paths() -> tuple[Path | None, Path | None]:
    """(torch dir, onnx dir) of the local student; each None when its weights are missing."""
    tdir, odir = config.MODELS_DIR / STUDENT_DIR, config.MODELS_DIR / STUDENT_ONNX_DIR
    has_torch = tdir.is_dir() and any(tdir.glob("*.safetensors"))
    return (tdir if has_torch else None), (odir if (odir / "model.onnx").is_file() else None)


def _version(path: Path, weights: Path) -> str:
    vfile = path / "VERSION"
    if vfile.is_file() and (v := vfile.read_text(encoding="utf-8").strip()):
        return v
    return str(int(weights.stat().st_mtime))


def _student() -> ModelSpec | None:
    """ST-FinSent (SPEC §5.0), our 5-class target-aware student; torch weights first, else its ONNX export."""
    tdir, odir = student_paths()
    if tdir is not None:
        path, weights = tdir, sorted(tdir.glob("*.safetensors"))[0]
    elif odir is not None:
        path, weights = odir, odir / "model.onnx"
    else:
        return None
    return ModelSpec("st_finsent", str(path), ALL, "cls", ("neg", "neg", "neu", "pos", "pos"), "target",
                     version=_version(path, weights), cache_id=STUDENT_DIR)


if (_st := _student()) is not None:
    SPECS[_st.name] = _st


def specs_for(channel: str) -> list[ModelSpec]:
    return [s for s in SPECS.values() if channel in s.channels]
