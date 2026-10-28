"""Item-level combiner: model outputs + features (+ Gemini) -> calibrated 5-class probabilities.

SPEC §5.4. Until phase 2 trains a meta-model (models/meta_<group>.joblib), a transparent
heuristic is used: a weighted average of the models' (neg, neu, pos), nudged by hard evidence
(catalyst tags, numbers, StockTwits self-labels) and by Gemini when it answered, then spread
over 5 classes by intensity. Snapshots record which one ran in `model_version`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .types import CLASS_VALUES, ModelOutput

# Trust per model inside a channel group (heuristic only; the learned model replaces this).
MODEL_TRUST = {
    # Set from the public benchmark (docs/benchmark.md, FPB excluded for leakage): NLI is weakest.
    "text": {"st_finsent": 2.0, "fin_distilroberta": 1.3, "absa": 1.0, "finbert": 1.0,
             "finbert_tone": 0.8, "nli": 0.4},
    "social": {"st_finsent": 2.0, "fintwit": 1.2, "absa": 1.0, "stocktwits_rb": 0.8,
               "twitter_rb": 0.6, "nli": 0.4},
}

# Direction a catalyst usually implies for the target. Only the heuristic uses it.
EVENT_PRIOR = {
    "EARNINGS_BEAT": 0.5, "EARNINGS_MISS": -0.5, "GUIDANCE_UP": 0.7, "GUIDANCE_DOWN": -0.7,
    "ANALYST_UPGRADE": 0.5, "ANALYST_DOWNGRADE": -0.5, "OFFERING_DILUTION": -0.6,
    "BUYBACK_DIVIDEND": 0.4, "SHORT_REPORT": -0.6,
}
STRONG_EVENTS = frozenset(EVENT_PRIOR) | {"M_AND_A", "FDA_CLINICAL", "LEGAL_REGULATORY"}

LLM_ONEHOT_SMOOTH = (0.70, 0.12, 0.03)  # mass at the class, at +-1 class, at +-2 classes
LLM_WEIGHT = 0.55


def group_of(channel: str) -> str:
    return "social" if channel == "social" else "text"


@dataclass
class MetaResult:
    probs5: tuple[float, float, float, float, float]
    score: float
    cls: int
    confidence: float
    stance: str
    disagreement: float        # std of the models' (pos - neg); high = send to Gemini


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, None)
    s = v.sum()
    return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


def _llm5(cls: int) -> np.ndarray:
    at, near, far = LLM_ONEHOT_SMOOTH
    out = np.zeros(5)
    for k in range(5):
        d = abs(k - (cls + 2))
        out[k] = at if d == 0 else near if d == 1 else far if d == 2 else 0.0
    return _norm(out)


def _to5(p3: np.ndarray, strength: float) -> np.ndarray:
    """Spread (neg, neu, pos) over 5 classes. More confident and more intense -> more mass at +-2."""
    neg, neu, pos = p3
    s_neg = neg * strength * neg
    s_pos = pos * strength * pos
    return np.array([s_neg, neg - s_neg, neu, pos - s_pos, s_pos])


def _evidence(features: dict[str, float], events: list[str]) -> float:
    e = sum(EVENT_PRIOR.get(t, 0.0) for t in events)
    e += 0.4 * features.get("pt_dir", 0.0) + 0.5 * features.get("pct_move", 0.0)
    e += 0.5 * features.get("insider_value", 0.0)
    return float(np.clip(e, -1.0, 1.0))


class Meta:
    """Loads learned meta-models when present, else uses the heuristic."""

    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.learned: dict[str, dict] = {}
        for group in ("text", "social"):
            path = models_dir / f"meta_{group}.joblib"
            if path.exists():
                import joblib
                self.learned[group] = joblib.load(path)  # {"model", "keys", "temperature", "version"}

    @property
    def version(self) -> str:
        if self.learned:
            return "meta-" + "+".join(f"{g}:{b.get('version', '?')}" for g, b in sorted(self.learned.items()))
        return "heuristic-v0"

    def predict(self, channel: str, outputs: list[ModelOutput], features: dict[str, float],
                events: list[str], llm: dict | None = None) -> MetaResult:
        group = group_of(channel)
        diffs = [o.probs[2] - o.probs[0] for o in outputs]
        disagreement = float(np.std(diffs)) if len(diffs) > 1 else 0.0
        if group in self.learned:
            p5 = self._learned(group, outputs, features)
        else:
            p5 = self._heuristic(group, outputs, features, events)
        if llm and llm.get("relevant", True):
            # Fixed blend: the learned model is trained on Gemini labels, so Gemini cannot be one
            # of its inputs (leakage). Phase 2 tunes LLM_WEIGHT on the gold set.
            p5 = _norm((1 - LLM_WEIGHT) * p5 + LLM_WEIGHT * _llm5(int(llm["cls"])))
        score = float(np.dot(p5, CLASS_VALUES))
        cls = int(np.argmax(p5)) - 2
        stance = self._stance(channel, features, llm, p5)
        return MetaResult(tuple(round(float(x), 4) for x in p5), round(score, 4), cls,
                          round(float(p5.max()), 4), stance, round(disagreement, 4))

    def _heuristic(self, group, outputs, features, events) -> np.ndarray:
        trust = MODEL_TRUST[group]
        acc, wsum = np.zeros(3), 0.0
        for o in outputs:
            w = trust.get(o.model, 0.5)
            acc += w * np.asarray(o.probs)
            wsum += w
        p3 = acc / wsum if wsum else np.array([0.0, 1.0, 0.0])

        mix, mix_w = p3 * 1.0, 1.0
        ev = _evidence(features, events)
        if ev:
            w = 0.5 * abs(ev)
            mix += w * (np.array([0, 1 - ev, ev]) if ev > 0 else np.array([-ev, 1 + ev, 0]))
            mix_w += w
        label = features.get("user_label", 0.0)
        if label:   # the author's own bullish/bearish tag on StockTwits
            mix += 0.6 * (np.array([0.05, 0.15, 0.8]) if label > 0 else np.array([0.8, 0.15, 0.05]))
            mix_w += 0.6
        p3 = _norm(mix / mix_w)

        strength = 0.3 + 0.3 * abs(features.get("vader_compound", 0.0))
        if STRONG_EVENTS.intersection(events):
            strength += 0.3
        return _norm(_to5(p3, min(strength, 1.0)))

    def _learned(self, group, outputs, features) -> np.ndarray:
        bundle = self.learned[group]
        x = np.array([[vector_value(k, outputs, features, None) for k in bundle["keys"]]])
        return learned_probs(bundle, x)[0]

    @staticmethod
    def _stance(channel, features, llm, p5) -> str:
        if channel != "social":
            return "none"
        label = features.get("user_label", 0.0)
        if label:
            return "bullish" if label > 0 else "bearish"
        if llm and llm.get("stance") in ("bullish", "bearish"):
            return llm["stance"]
        lean = (p5[3] + p5[4]) - (p5[0] + p5[1])
        if lean > 0.25:
            return "bullish"
        if lean < -0.25:
            return "bearish"
        return "none"


def learned_probs(bundle: dict, x: np.ndarray) -> np.ndarray:
    """(n, 5) calibrated probabilities. Classes the model never saw get probability 0."""
    model = bundle["model"]
    d = np.asarray(model.decision_function(x), dtype=float)
    if d.ndim == 1:   # binary: one logit for classes_[1]; softmax([0, d]) == sigmoid(d)
        d = np.column_stack([np.zeros_like(d), d])
    logits = d / bundle.get("temperature", 1.0)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    out = np.zeros((len(x), 5))
    for j, c in enumerate(model.classes_):
        out[:, int(c) + 2] = p[:, j]
    return out


def vector_value(key: str, outputs: list[ModelOutput], features: dict[str, float], llm: dict | None) -> float:
    """One input of the learned meta-model. Keys: "m:<model>:<neg|neu|pos>", "f:<feature>",
    "llm:cls", "llm:has". Missing model -> neutral (0, 1, 0)."""
    kind, _, rest = key.partition(":")
    if kind == "m":
        name, _, part = rest.rpartition(":")
        idx = ("neg", "neu", "pos").index(part)
        for o in outputs:
            if o.model == name:
                return float(o.probs[idx])
        return 1.0 if part == "neu" else 0.0
    if kind == "f":
        v = features.get(rest, 0.0)
        return float(v) if math.isfinite(v) else 0.0
    if key == "llm:cls":
        return float(llm["cls"]) / 2 if llm else 0.0
    if key == "llm:has":
        return 1.0 if llm else 0.0
    raise KeyError(key)
