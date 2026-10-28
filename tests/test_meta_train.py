"""Meta-model training on synthetic, learnable data (no models, no network)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from sentiment import config
from sentiment.features import FEATURE_KEYS
from sentiment.meta import Meta, learned_probs
from sentiment.types import ModelOutput, RawItem
from train import meta_train


def fake_examples(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    exs = []
    for i in range(n):
        cls = int(rng.integers(-2, 3))
        pos = np.clip(0.5 + 0.2 * cls + rng.normal(0, 0.08), 0.01, 0.98)
        neg = np.clip(0.5 - 0.2 * cls + rng.normal(0, 0.08), 0.01, 0.98)
        neu = max(0.01, 1 - abs(cls) * 0.4)
        s = pos + neg + neu
        outputs = [ModelOutput(m, (neg / s, neu / s, pos / s)) for m in ("finbert", "absa", "nli")]
        feats = {k: 0.0 for k in FEATURE_KEYS} | {"vader_compound": 0.3 * cls + rng.normal(0, 0.1)}
        iid = f"id{i}"
        item = RawItem(iid, "google_news", "news", "Reuters", "t", "", "", datetime.now(timezone.utc), "X")
        exs.append({"item": item, "outputs": outputs, "features": feats, "events": ["OTHER"],
                    "label": {"cls": cls, "relevant": True}, "origin": "teacher",
                    "split": "tholdout" if meta_train.bucket(iid) >= 90 else "train"})
    return exs


def test_target_cls_irrelevant_is_neutral():
    assert meta_train.target_cls({"cls": 2, "relevant": False}) == 0
    assert meta_train.target_cls({"cls": -1}) == -1


def test_train_group_learns_and_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    r = meta_train.train_group("text", fake_examples(1500), force=True)
    assert r["saved"] and r["eval_on"] == "teacher-holdout"
    assert r["metrics"]["learned"]["macro_f1_3"] > 0.8
    assert 0.25 <= r["temperature"] <= 5.0
    meta = Meta(models_dir=tmp_path)
    assert "text" in meta.learned and meta.version.startswith("meta-text:lr-")
    ex = fake_examples(1, seed=7)[0]
    res = meta.predict("news", ex["outputs"], ex["features"], ex["events"])
    assert sum(res.probs5) == pytest.approx(1, abs=1e-3)


def test_learned_probs_scatter_missing_class(tmp_path):
    from sklearn.linear_model import LogisticRegression
    x = np.array([[0.0], [1.0], [2.0], [3.0]] * 10)
    y = np.array([-1, 0, 1, 2] * 10)            # class -2 never seen
    p = learned_probs({"model": LogisticRegression().fit(x, y), "temperature": 1.0}, x[:4])
    assert p.shape == (4, 5) and np.allclose(p[:, 0], 0) and np.allclose(p.sum(1), 1)


def test_too_little_data_skips():
    r = meta_train.train_group("social", fake_examples(50), force=True)
    assert "skipped" in r


def test_learned_probs_binary_and_small_calib():
    from sklearn.linear_model import LogisticRegression
    x = np.array([[0.0], [0.1], [0.9], [1.0]] * 5)
    y = np.array([-1, -1, 1, 1] * 5)
    bundle = {"model": LogisticRegression().fit(x, y), "temperature": 1.0}
    p = learned_probs(bundle, x[:4])
    assert p.shape == (4, 5) and np.allclose(p.sum(axis=1), 1)
    assert p[0, 1] > 0.5 and p[3, 3] > 0.5 and p[:, [0, 2, 4]].sum() == 0
    assert meta_train.fit_temperature(bundle, x[:4], y[:4]) == 1.0   # under MIN_CALIB rows -> T = 1


def _crossfit_data():
    from test_crossfit import make
    return (make("g", 220, "gold") + make("t", 300, "teacher", seed=2)
            + make("h", 220, "gold", channel="social", seed=3) + make("s", 300, "teacher", channel="social", seed=4))


def test_force_text_saves_only_text(tmp_path, monkeypatch):
    labeled = _crossfit_data()
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(meta_train, "load_labeled", lambda: labeled)
    monkeypatch.setattr(meta_train, "featurize", lambda exs, device: None)
    monkeypatch.setattr(meta_train, "beats_heuristic", lambda cv: False)   # only --force can save
    assert meta_train.main(["--force", "text"]) == 0
    assert (tmp_path / "meta_text.joblib").exists() and not (tmp_path / "meta_social.joblib").exists()
    assert meta_train.forced_groups("all") == {"text", "social"} and meta_train.forced_groups(None) == set()


def test_crossfit_path_bundle(tmp_path, monkeypatch):
    from eval.crossfit import crossfit
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    exs = [e for e in _crossfit_data() if e["item"].channel == "news"]
    cv = crossfit({"text": [e for e in exs if e["origin"] == "gold"]},
                  [e for e in exs if e["origin"] == "teacher"], 5, n_boot=100)["text"]
    assert "fallback" not in cv
    r = meta_train.train_group("text", exs, force=False, cv=cv)
    assert r["eval_on"] == "gold-crossfit" and r["counts"]["train"] == len(exs)
    assert r["temperature"] == round(meta_train.fit_temperature_multi(cv["_parts"]), 4)
    assert 0.25 <= r["temperature"] <= 5.0
    assert r["saved"] == meta_train.beats_heuristic(cv) and r["crossfit"]["n"] == cv["n"]
    assert "macro_f1_3" in r["metrics"]["learned"] and not any(k.startswith("_") for k in r["crossfit"])
    if r["saved"]:
        assert "text" in Meta(models_dir=tmp_path).learned
