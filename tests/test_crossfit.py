"""Gold cross-fit on synthetic, learnable data (no models, no network)."""
import json
from datetime import datetime, timezone

import numpy as np

from eval import crossfit as CF
from eval.metrics import macro_f1
from sentiment.features import FEATURE_KEYS
from sentiment.types import ModelOutput, RawItem
from train.io import bucket


def make(prefix: str, n: int, origin: str, channel: str = "news", seed: int = 0) -> list[dict]:
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
        iid = f"{prefix}{i}"
        src = "stocktwits" if channel == "social" else "google_news"
        item = RawItem(iid, src, channel, "Reuters", "t", "", "", datetime.now(timezone.utc), "X")
        ex = {"item": item, "outputs": outputs, "features": feats, "events": ["OTHER"],
              "label": {"cls": cls, "relevant": True}, "origin": origin, "split": "train"}
        if origin == "gold":
            ex["teacher"] = {"cls": cls, "relevant": True}
        exs.append(ex)
    return exs


def ids(exs: list[dict]) -> set[str]:
    return {e["item"].id for e in exs}


def test_fold_splits_never_leak_test_ids():
    gold = make("g", 300, "gold")
    # teacher rows for every gold id (as Gemini labels them too) plus teacher-only rows
    teacher = make("g", 300, "teacher", seed=1) + make("t", 200, "teacher", seed=2)
    eligible, always = CF.split_gold(gold, 65)
    splits = CF.fold_splits(gold, teacher, 5, 65)
    seen_test: list[str] = []
    for s in splits:
        test_ids, calib_ids = ids(s["test"]), ids(s["calib"])
        train_teacher = {e["item"].id for e in s["train"] if e["origin"] == "teacher"}
        assert not test_ids & train_teacher                 # no teacher row of a test-fold gold id
        assert not test_ids & ids(s["train"]) and not calib_ids & ids(s["train"])
        assert not test_ids & calib_ids
        assert ids(always) <= ids(s["train"])
        assert all(bucket(i) % 5 == s["fold"] and bucket(i) >= 65 for i in test_ids)
        assert all(bucket(i) % 5 == (s["fold"] + 1) % 5 for i in calib_ids)
        seen_test += sorted(test_ids)
    assert sorted(seen_test) == sorted(ids(eligible))       # every eligible item scored exactly once


def test_f1_dir_matches_sklearn():
    rng = np.random.default_rng(3)
    for _ in range(20):
        y, c = rng.integers(-2, 3, 40), rng.integers(-2, 3, 40)
        assert CF._f1_dir(y, c) == macro_f1(np.sign(y), np.sign(c))
    assert CF._f1_dir(np.array([1, 1]), np.array([1, 1])) == macro_f1([1, 1], [1, 1])


def test_crossfit_reports_and_falls_back():
    gold = make("g", 260, "gold")
    teacher = make("t", 400, "teacher", seed=2) + make("s", 300, "teacher", channel="social", seed=4)
    small = make("h", 40, "gold", channel="social", seed=5)
    res = CF.crossfit({"text": gold, "social": small}, teacher, 5, n_boot=200)
    social = res["social"]
    assert "fallback" in social and social["n"] == len(CF.split_gold(small, 65)[0])
    r = res["text"]
    n = len(CF.split_gold(gold, 65)[0])
    assert r["n"] == n >= CF.MIN_ELIGIBLE and r["n_teacher"] == 400
    lc, h = r["models"]["learned_cv"], r["models"]["heuristic"]
    assert lc["n"] == h["n"] == n and lc["f1_dir"] > 0.8
    assert lc["f1_dir_ci"][0] <= lc["f1_dir"] <= lc["f1_dir_ci"][1]
    for k in ("rps", "log_loss", "brier", "ece_eqmass", "ece_floor"):
        assert k in lc and k in h
    assert {"finbert", "absa", "nli", "gemini"} <= set(r["models"])
    assert r["models"]["gemini"]["acc5"] == 1.0
    assert set(r["learned_vs_heuristic"]) == {"f1_dir", "rps", "mcnemar_dir_p"}
    assert sum(len(y) for _, _, y in r["_parts"]) == n
    assert set(r["_metrics"]) == {"learned", "heuristic"}
    pub = CF.public(res)
    assert not any(k.startswith("_") for g in pub.values() for k in g)


def test_student_gold_makes_low_bucket_eligible(tmp_path, monkeypatch):
    gold = make("g", 300, "gold")
    low = next(e["item"].id for e in gold if bucket(e["item"].id) < 65)
    eligible, always = CF.split_gold(gold, 65, {low})
    assert low in ids(eligible) and low not in ids(always)
    assert low not in ids(CF.split_gold(gold, 65)[0])
    assert low in ids(s for sp in CF.fold_splits(gold, [], 5, 65, {low}) for s in sp["test"])

    # CLI: file is read, and --no-student-gold ignores it
    f = tmp_path / "gold_ids.json"
    f.write_text(json.dumps([low]), encoding="utf-8")
    assert CF.load_student_gold(f) == {low}
    assert CF.load_student_gold(tmp_path / "nope.json") == set()
    seen = []
    monkeypatch.setattr(CF, "STUDENT_GOLD", f)
    monkeypatch.setattr(CF, "STAGE2", tmp_path / "no-stage2.json")   # the real models/ may hold a stage-2 student
    monkeypatch.setattr(CF, "load_labeled", lambda: gold)
    monkeypatch.setattr(CF, "featurize", lambda labeled, device: None)
    monkeypatch.setattr(CF, "crossfit", lambda by_group, teacher, folds, **kw: seen.append(kw["student_gold"]) or {})
    monkeypatch.setattr(CF.config, "ROOT", tmp_path)
    CF.main([])
    CF.main(["--no-student-gold"])
    assert seen == [{low}, set()]


def test_apply_student_oof_replaces_only_st_finsent(tmp_path):
    exs = make("g", 3, "gold")
    exs[0]["outputs"].append(ModelOutput("st_finsent", (1.0, 0.0, 0.0)))
    p = tmp_path / "oof.json"
    p.write_text(json.dumps({"ids": ["g0", "g1"], "tickers": ["X", "OTHER"],
                             "s2": [[0, 0, 0.2, 0.3, 0.5], [0.1, 0.1, 0.8, 0, 0]]}), encoding="utf-8")
    n = CF.apply_student_oof(exs, CF.load_student_oof(p))
    assert n == 2   # g1 matches on id when the ticker differs; g2 has no OOF row
    st = [o for o in exs[0]["outputs"] if o.model == "st_finsent"]
    assert len(st) == 1 and np.allclose(st[0].probs, (0.0, 0.2, 0.8))
    assert len(exs[0]["outputs"]) == 4
    assert np.allclose([o for o in exs[1]["outputs"] if o.model == "st_finsent"][0].probs, (0.2, 0.8, 0.0))
    assert not any(o.model == "st_finsent" for o in exs[2]["outputs"])


def test_prepare_student_only_for_stage2(tmp_path, monkeypatch):
    import pytest
    exs = make("g", 3, "gold")
    assert CF.prepare_student(exs, None, stage2=tmp_path / "none.json") is None
    s2 = tmp_path / "stage2.json"
    s2.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(CF, "STUDENT_OOF", tmp_path / "missing.json")
    with pytest.raises(SystemExit):
        CF.prepare_student(exs, None, stage2=s2)
    oof = tmp_path / "oof.json"
    oof.write_text(json.dumps({"ids": ["g0"], "tickers": ["X"], "s2": [[0, 0, 1, 0, 0]]}), encoding="utf-8")
    monkeypatch.setattr(CF, "STUDENT_OOF", oof)
    assert CF.prepare_student(exs, None, stage2=s2) == {"g0", "g1", "g2"}   # g1, g2 are newer than the student
    assert [o for o in exs[0]["outputs"] if o.model == "st_finsent"][0].probs == (0, 1, 0)
