import os
from datetime import datetime, timezone

import pytest

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentiment import config
from sentiment.models.registry import SPECS, specs_for
from sentiment.models.runner import ModelRunner
from sentiment.types import Company, RawItem

GOOD = "Company X beats estimates and raises full-year guidance"
BAD = "Shares plunge after the company slashes guidance"
MOON = "$TSLA to the moon 🚀 loading calls"


@pytest.fixture(scope="module")
def runner():
    return ModelRunner(cache=False)


def _argmax(p):
    return max(range(3), key=lambda i: p[i])


def test_registry():
    # st_finsent registers itself only when local trained weights exist
    base = lambda specs: {s.name for s in specs} - {"st_finsent"}
    assert len(base(SPECS.values())) == 8
    assert base(specs_for("social")) == {"absa", "nli", "fintwit", "stocktwits_rb", "twitter_rb"}
    assert base(specs_for("news")) == {"finbert", "fin_distilroberta", "finbert_tone", "absa", "nli"}


@pytest.mark.parametrize("name", ["finbert", "fin_distilroberta", "absa", "nli"])
def test_label_mapping_news(runner, name):
    p = runner.predict(name, [GOOD, BAD], ["Company X", "Company X"])
    if name == "nli":  # spec'd hypotheses give low entailment for both, so neu dominates; direction must still hold
        assert p[0][2] > p[0][0] and p[1][0] > p[1][2], p
        return
    assert _argmax(p[0]) == 2, (name, p[0])
    assert _argmax(p[1]) == 0, (name, p[1])
    assert abs(sum(p[0]) - 1) < 1e-4


@pytest.mark.parametrize("name", ["fintwit", "stocktwits_rb"])
def test_label_mapping_social(runner, name):
    neg, neu, pos = runner.predict(name, [MOON])[0]
    assert pos > neg
    if name == "stocktwits_rb":
        assert neu == 0.0


def test_empty_and_long_text(runner):
    assert runner.predict("finbert", []) == []
    p = runner.predict("finbert", ["good results " * 2000])
    assert len(p) == 1 and abs(sum(p[0]) - 1) < 1e-4


def test_nli_arithmetic(monkeypatch):
    r = ModelRunner(device="cpu", cache=False)

    def fake(spec, first, second):
        return [[0.9, 0.1] if "go up" in h else [0.05, 0.95] for h in second]
    monkeypatch.setattr(r, "_run", fake)
    neg, neu, pos = r.predict("nli", ["t"], ["Acme"])[0]
    assert (round(neg, 4), round(neu, 4), round(pos, 4)) == (0.0476, 0.0952, 0.8571)


def test_nli_neutral_floor(monkeypatch):
    r = ModelRunner(device="cpu", cache=False)
    monkeypatch.setattr(r, "_run", lambda s, f, h: [[0.9, 0.1]] * len(f))  # both 0.9 -> neu 0.1
    neg, neu, pos = r.predict("nli", ["t"], ["A"])[0]
    assert abs(neg - pos) < 1e-9 and abs(sum((neg, neu, pos)) - 1) < 1e-9


def test_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    r = ModelRunner(device="cpu", cache=True)
    calls = []

    def fake(spec, first, second):
        calls.append(len(first))
        return [[0.1, 0.2, 0.7]] * len(first)  # finbert: (pos, neg, neu)
    monkeypatch.setattr(r, "_run", fake)
    a = r.predict("finbert", ["hello"])
    b = r.predict("finbert", ["hello"])
    assert calls == [1] and a == b
    assert a[0] == pytest.approx((0.2, 0.7, 0.1))
    r.predict("finbert", ["hello", "new"])
    assert calls == [1, 1]


def test_two_class_mapping(monkeypatch):
    r = ModelRunner(device="cpu", cache=False)
    monkeypatch.setattr(r, "_run", lambda s, f, h: [[0.25, 0.75]] * len(f))
    assert r.predict("stocktwits_rb", ["x"])[0] == (0.25, 0.0, 0.75)


def test_score_items_skips_failed_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    r = ModelRunner(device="cpu", cache=False)

    def fake(spec, first, second):
        if spec.name == "absa":
            raise RuntimeError("model absa unavailable")
        n = len(SPECS[spec.name].labels) or 2
        return [[1.0 / n] * n] * len(first)
    monkeypatch.setattr(r, "_run", fake)
    co = Company("X", "X CORP", "Company X", 1)
    now = datetime.now(timezone.utc)
    items = [RawItem("a", "google_news", "news", "p", "T", "body " * 200, "u", now, "X")]
    out = r.score_items(items, co, "news")
    names = [o.model for o in out["a"]]
    assert "absa" not in names and "finbert" in names and "nli" in names
    assert all(len(o.probs) == 3 for o in out["a"])


def test_social_fallback():
    from sentiment.models.runner import _social_fallback
    assert _social_fallback("hi @bob see https://x.com/a") == "hi @user see http"


def test_speed_report(runner):
    """Informational: per-model load time and items/sec on a batch of 64 headlines."""
    import time
    texts = [f"Company X reports quarterly results number {i} ahead of expectations" for i in range(64)]
    for dev in ("auto", "cpu"):
        r = runner if dev == "auto" else ModelRunner(device="cpu", cache=False)
        for name in ("finbert", "absa", "nli", "fintwit"):
            t0 = time.time()
            r._load(SPECS[name])
            load = time.time() - t0
            t0 = time.time()
            r.predict(name, texts, ["Company X"] * 64)
            print(f"SPEED {r.device} {name}: load {load:.1f}s, {64 / (time.time() - t0):.0f} items/s")
