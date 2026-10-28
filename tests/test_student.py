import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import torch

from sentiment.types import Company, RawItem
from train import build_dataset as bd
from train import io as tio_real
from train.student import soft_loss


def test_smooth5_edges():
    assert bd.smooth5(-2) == pytest.approx([0.85, 0.15, 0, 0, 0])
    assert bd.smooth5(2) == pytest.approx([0, 0, 0, 0.15, 0.85])
    assert bd.smooth5(0) == pytest.approx([0, 0.075, 0.85, 0.075, 0])
    assert bd.smooth5(-1) == pytest.approx([0.075, 0.85, 0.075, 0, 0])
    for c in range(-2, 3):
        assert sum(bd.smooth5(c)) == pytest.approx(1)
    with pytest.raises(ValueError):
        bd.smooth5(3)


def test_map3():
    assert bd.map3(-1) == [0.35, 0.65, 0, 0, 0]
    assert bd.map3(0) == [0, 0, 1, 0, 0]
    assert bd.map3(1) == [0, 0, 0, 0.65, 0.35]


def test_norm_text_and_val_hash():
    assert bd.norm_text("  Hello   WORLD\n") == "hello world"
    for i in range(50):
        s = f"id{i}"
        assert bd.is_val(s) == (int(hashlib.sha1(s.encode()).hexdigest(), 16) % 20 == 0)


def test_soft_loss_hand_computed():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]])
    soft = torch.tensor([[0.2] * 5, [1.0, 0, 0, 0, 0]])
    w = torch.tensor([1.0, 3.0])
    import math
    l1 = math.log(5)
    l2 = -(2.0 - math.log(math.exp(2.0) + 4))
    assert soft_loss(logits, soft, w).item() == pytest.approx((l1 * 1 + l2 * 3) / 4, rel=1e-5)


# ---- build() with fake io / datasets ----
def _id(s):
    return hashlib.sha1(s.encode()).hexdigest()


def _crow(iid, title, text="", source="google_news", user_label=None, day=1):
    item = {"id": iid, "title": title, "text": text, "source": source, "ticker": "NVDA", "meta": {}}
    if user_label:
        item["meta"]["user_label"] = user_label
    return {"item": item, "company": {"short_name": "Nvidia"}, "day": day}


def _setup(monkeypatch, tmp_path, corpus, teacher, gold, public, test_set, buckets=None):
    def item_from_json(d):
        return RawItem(id=d["id"], source=d["source"], channel="news", publisher="p", title=d["title"],
                       text=d["text"], url="", published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                       ticker=d["ticker"], meta=d["meta"])

    def write(name, rows):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p

    c = write("corpus.jsonl", corpus)
    t = write("teacher.jsonl", teacher)
    g = write("gold.jsonl", gold)
    days = {r["item"]["id"]: r["day"] for r in corpus}

    def item_from_json2(d):
        it = item_from_json(d)
        it.published_at = datetime(2026, 1, days[d["id"]], tzinfo=timezone.utc)
        return it

    tio = SimpleNamespace(
        CORPUS=c, TEACHER=t, GOLD=g,
        read_jsonl=lambda p: (json.loads(x) for x in open(p, encoding="utf-8") if x.strip()),
        item_from_json=item_from_json2,
        company_from_json=lambda d: Company("NVDA", "NVIDIA CORP", d["short_name"], 1),
        bucket=lambda i: (buckets or {}).get(i, 0), META_RESERVED=70, latest_teacher=tio_real.latest_teacher,
    )
    ds = SimpleNamespace(DATASETS=tuple(public), load=lambda n, s, limit=None, seed=0: public[n],
                         test_texts=lambda: set(test_set))
    monkeypatch.setattr(bd, "tio", tio)
    monkeypatch.setattr(bd, "ds", ds)


def _ex(i, text, target="", label3=0, label5=0):
    return SimpleNamespace(id=i, text=text, target=target, label3=label3, label5=label5)


def _read(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def test_build_full(monkeypatch, tmp_path):
    vid = "v"
    corpus = [
        _crow("a", "Nvidia soars", "Big beat " * 100),          # teacher, cls 2
        _crow("c", "Irrelevant", ""),                            # teacher relevant False -> class 0
        _crow(vid, "Val item"),                                  # teacher bucket 66 -> val
        _crow("r", "Reserved item"),                             # teacher bucket 80 -> meta-model only
        _crow("g", "Gold item"),                                 # in gold -> dropped
        _crow("l", "Leaky headline"),                            # teacher but text in test set
        _crow("t", "Leaky title", "with a different body"),      # teacher, title in test set
        _crow("s1", "to the moon", source="stocktwits", user_label="Bullish", day=1),
        _crow("s2", "dump it", source="stocktwits", user_label="Bearish", day=5),
        _crow("s3", "no label", source="stocktwits"),
        _crow("a", "dup row ignored"),
    ]
    teacher = [
        {"id": "a", "ticker": "NVDA", "cls": 2, "relevant": True},
        {"id": "c", "ticker": "NVDA", "cls": -2, "relevant": False},
        {"id": vid, "ticker": "NVDA", "cls": 1, "relevant": True},
        {"id": "r", "ticker": "NVDA", "cls": 1, "relevant": True},
        {"id": "g", "ticker": "NVDA", "cls": 1, "relevant": True},
        {"id": "l", "ticker": "NVDA", "cls": 1, "relevant": True},
        {"id": "t", "ticker": "NVDA", "cls": 1, "relevant": True},
    ]
    gold = [{"id": "g", "skipped": False}]
    public = {
        "tfns": [_ex("1", "Stocks rally", label3=1), _ex("2", "TEST  Sentence", label3=1)],
        "fiqa": [_ex("3", "Tsla up", target="TSLA", label5=-2)],
    }
    _setup(monkeypatch, tmp_path, corpus, teacher, gold, public, {"test sentence", "leaky headline", "leaky title"},
           buckets={vid: 66, "r": 80})
    out = tmp_path / "out"
    stats = bd.build(out, max_stocktwits=1, public_weight=0.5, weak_path=tmp_path / "none.jsonl")
    train, val = _read(out / "train.jsonl"), _read(out / "val.jsonl")
    by = {r["id"]: r for r in train + val}

    # teacher val = buckets 65..69; one teacher row is too few, so public val rows top it up
    assert [r["id"] for r in val if r["source"] == "teacher"] == [vid]
    assert all(r["source"].startswith("public:") and bd.is_val(r["id"]) for r in val if r["source"] != "teacher")
    assert "r" not in by and stats["reserved_for_meta"] == 1
    assert "g" not in by and "l" not in by and "t" not in by and "tfns:2" not in by
    assert stats["dropped_leakage"] == 3 and stats["dropped_gold"] == 1
    assert by["a"]["target"] == "NVDA | Nvidia"
    assert by["a"]["text"].startswith("Nvidia soars. Big beat") and len(by["a"]["text"]) == len("Nvidia soars. ") + 600
    assert by["a"]["soft"] == pytest.approx([0, 0, 0, 0.15, 0.85]) and by["a"]["weight"] == 1.0
    assert by["c"]["soft"] == pytest.approx(bd.smooth5(0))
    assert by["c"]["text"] == "Irrelevant"
    # stocktwits: newest first, capped at 1, unlabeled skipped
    assert "s2" in by and "s1" not in by and "s3" not in by
    assert by["s2"]["soft"] == [0.25, 0.6, 0.15, 0, 0] and by["s2"]["weight"] == 0.3
    assert by["tfns:1"]["soft"] == [0, 0, 0, 0.65, 0.35] and by["tfns:1"]["weight"] == 0.5
    assert by["tfns:1"]["target"] == "" and by["tfns:1"]["source"] == "public:tfns"
    assert by["fiqa:3"]["target"] == "TSLA | TSLA"
    assert by["fiqa:3"]["soft"] == pytest.approx([0.85, 0.15, 0, 0, 0])
    assert stats["rows"]["teacher"] == 3


def test_build_no_teacher_val_from_public(monkeypatch, tmp_path, capsys):
    pub = [_ex(str(i), f"text number {i}", label3=0) for i in range(200)]
    _setup(monkeypatch, tmp_path, [], [], [], {"sentfin": pub}, set())
    out = tmp_path / "o"
    stats = bd.build(out, weak_path=tmp_path / "none.jsonl")
    train, val = _read(out / "train.jsonl"), _read(out / "val.jsonl")
    assert "WARNING" in capsys.readouterr().out
    assert len(train) + len(val) == 200 and 0 < len(val) < 40
    assert all(bd.is_val(r["id"]) for r in val) and not any(bd.is_val(r["id"]) for r in train)
    assert stats["n_val"] == len(val)


def _weak(tmp_path, rows):
    p = tmp_path / "weak.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _wrow(i, label, created="2026-01-01", ticker="NVDA", text=None):
    return {"id": i, "ticker": ticker, "text": text or f"weak text {i}", "label": label, "mid": 1, "user": "u",
            "created_at": created}


def test_build_weak_file_used(monkeypatch, tmp_path):
    corpus = [_crow("s1", "corpus st", source="stocktwits", user_label="Bullish"), _crow("a", "teacher item"),
              _crow("g", "Gold item")]
    teacher = [{"id": "a", "ticker": "NVDA", "cls": 1, "relevant": True}]
    _setup(monkeypatch, tmp_path, corpus, teacher, [{"id": "g", "skipped": True}, {"id": "g2", "skipped": False}],
           {}, set())
    weak = _weak(tmp_path, [
        _wrow("w1", 1, "2026-01-02"), _wrow("w2", -1, "2026-01-03"), _wrow("w2", -1, "2026-01-03"),
        _wrow("g", 1), _wrow("a", 1), _wrow("w3", 1, ticker="ZZZ"), _wrow("w4", 1, "2026-01-01")])
    out = tmp_path / "o"
    stats = bd.build(out, max_stocktwits=2, weak_path=weak, weak_weight=0.3)
    rows = {r["id"]: r for r in _read(out / "train.jsonl") + _read(out / "val.jsonl")}
    assert stats["rows"]["weak_stocktwits"] == 2 and "s1" not in rows
    assert set(rows) == {"a", "w1", "w2"}                  # w4 capped (oldest); g, a, ZZZ dropped
    assert rows["w2"]["soft"] == [0.25, 0.6, 0.15, 0, 0] and rows["w1"]["soft"] == [0, 0, 0.15, 0.6, 0.25]
    assert rows["w1"]["weight"] == 0.3 and rows["w1"]["target"] == "NVDA | Nvidia"
    assert rows["w1"]["source"] == "weak_stocktwits"
    assert stats["dropped_gold"] == 2                      # corpus gold row + weak gold row
    assert json.loads((out / "gold_ids.json").read_text()) == ["g", "g2"]


def test_build_corpus_fallback_without_weak_file(monkeypatch, tmp_path):
    corpus = [_crow("s1", "to the moon", source="stocktwits", user_label="Bullish")]
    _setup(monkeypatch, tmp_path, corpus, [], [], {}, set())
    stats = bd.build(tmp_path / "o", weak_path=tmp_path / "missing.jsonl")
    assert stats["rows"] == {"stocktwits": 1}
    assert json.loads((tmp_path / "o" / "gold_ids.json").read_text()) == []


def test_build_multi_ticker_teacher_company(monkeypatch, tmp_path):
    corpus = [_crow("m", "Both"), _crow("m", "Both")]
    corpus[1]["item"]["ticker"] = "AMD"
    corpus[1]["company"] = {"short_name": "AMD Inc"}
    teacher = [{"id": "m", "ticker": "NVDA", "cls": 1, "relevant": True},
               {"id": "m", "ticker": "AMD", "cls": -1, "relevant": True}]
    _setup(monkeypatch, tmp_path, corpus, teacher, [], {}, set())
    # fake company_from_json hard-codes ticker NVDA; only short_name matters for the target
    bd.build(tmp_path / "o", weak_path=tmp_path / "none.jsonl")
    got = {r["target"] for r in _read(tmp_path / "o" / "train.jsonl")}
    assert got == {"NVDA | Nvidia", "AMD | AMD Inc"}


def test_agreement_relations():
    assert [bd.agreement(a, b) for a, b in ((1, 1), (1, 2), (-2, -1), (0, 1), (-1, 0), (1, -1), (2, -2))] == \
        ["same", "near", "near", "neutral", "neutral", "opposite", "opposite"]
    soft, w, rel = bd.second_soft(2, None)
    assert soft == bd.smooth5(2) and w == 1.0 and rel == "none"
    soft, w, rel = bd.second_soft(1, -1)
    assert rel == "opposite" and w == bd.AGREE_WEIGHT["opposite"]
    assert soft == pytest.approx([(x + y) / 2 for x, y in zip(bd.smooth5(1), bd.smooth5(-1))])
    assert sum(soft) == pytest.approx(1.0)


def test_build_second_labeler_weights_teacher_rows(monkeypatch, tmp_path):
    corpus = [_crow("a", "A up"), _crow("b", "B down"), _crow("c", "C flat")]
    teacher = [{"id": i, "ticker": "NVDA", "cls": c, "relevant": True} for i, c in (("a", 2), ("b", -1), ("c", 0))]
    _setup(monkeypatch, tmp_path, corpus, teacher, [], {}, [])
    sec = tmp_path / "second.jsonl"
    sec.write_text("\n".join(json.dumps(r) for r in (
        {"id": "a", "ticker": "NVDA", "cls": 2, "relevant": True},
        {"id": "b", "ticker": "OTHER", "cls": 1, "relevant": True},     # id fallback, opposite
    )), encoding="utf-8")
    stats = bd.build(tmp_path / "out", weak_path=tmp_path / "none.jsonl", second=sec)
    rows = {r["id"]: r for r in _read(tmp_path / "out" / "train.jsonl") if r["source"] == "teacher"}
    assert rows["a"]["weight"] == 1.0 and rows["a"]["soft"] == bd.smooth5(2)
    assert rows["b"]["weight"] == bd.AGREE_WEIGHT["opposite"]
    assert rows["c"]["weight"] == 1.0 and rows["c"]["soft"] == bd.smooth5(0)   # no second label
    assert stats["agreement"] == {"same": 1, "opposite": 1, "none": 1}
