import sys
import types

from sentiment import config
from train import teacher
from train.io import read_jsonl


def _row(i, ticker, day, channel="news"):
    return {"item": {"id": i, "source": "hn", "channel": channel, "publisher": "p", "title": "t" + i, "text": "",
                     "url": "u", "published_at": f"2026-10-{day:02d}T00:00:00+00:00", "ticker": ticker, "meta": {}},
            "company": {"ticker": ticker, "name": ticker, "short_name": ticker, "cik": 1, "aliases": [],
                        "ambiguous": False},
            "relevance": 1.0, "cluster_id": "c", "novelty": 1.0, "collected_at": "x"}


class FakeLabeler:
    def __init__(self, exhaust_after=None, skip=()):
        self.exhausted = False
        self.calls = []
        self.exhaust_after, self.skip = exhaust_after, set(skip)

    def label(self, company, items):
        self.calls.append((company.ticker, [i.id for i in items]))
        if self.exhaust_after and len(self.calls) >= self.exhaust_after:
            self.exhausted = True
        return {i.id: {"cls": 1, "stance": "bullish", "event": "OTHER", "relevant": True, "reason": "r"}
                for i in items if i.id not in self.skip}


def test_pending_groups_excludes_labeled_dedups_and_orders():
    rows = [_row("a1", "A", 1), _row("a2", "A", 5), _row("a2", "A", 5), _row("b1", "B", 2),
            _row("b2", "B", 3, "social"), _row("a3", "A", 3)]
    p = teacher.pending(rows, {"a3"}, channels=("news",))
    assert [r["item"]["id"] for r in p["A"]] == ["a2", "a1"]
    assert [r["item"]["id"] for r in p["B"]] == ["b1"]
    assert "b2" not in [r["item"]["id"] for r in p["B"]]


def test_round_robin_sleep_and_missing_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_BATCH", 2)
    sleeps = []
    monkeypatch.setattr(teacher.time, "sleep", lambda s: sleeps.append(s))
    rows = [_row(f"a{k}", "A", k + 1) for k in range(4)] + [_row("b0", "B", 1)]
    lab = FakeLabeler(skip={"a2"})
    out = tmp_path / "t.jsonl"
    teacher.run(lab, rows, set(), max_items=100, min_interval=4.5, out_path=out)
    assert [c[0] for c in lab.calls] == ["A", "B", "A"]
    assert lab.calls[0][1] == ["a3", "a2"]          # newest first
    assert len(sleeps) == 2 and all(0 < s <= 4.5 for s in sleeps)
    written = list(read_jsonl(out))
    assert {r["id"] for r in written} == {"a3", "a1", "a0", "b0"} and "a2" not in {r["id"] for r in written}
    assert set(written[0]) == {"id", "ticker", "cls", "stance", "event", "relevant", "reason", "ts", "pv"}
    assert written[0]["pv"] == 1   # FakeLabeler has no prompt_version


def test_stops_when_exhausted(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "GEMINI_BATCH", 1)
    monkeypatch.setattr(teacher.time, "sleep", lambda s: None)
    rows = [_row(f"a{k}", "A", k + 1) for k in range(5)]
    lab = FakeLabeler(exhaust_after=2)
    teacher.run(lab, rows, set(), out_path=tmp_path / "t.jsonl")
    assert len(lab.calls) == 2
    assert "quota exhausted" in capsys.readouterr().out


def test_max_items(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_BATCH", 2)
    monkeypatch.setattr(teacher.time, "sleep", lambda s: None)
    rows = [_row(f"a{k}", "A", k + 1) for k in range(6)]
    lab = FakeLabeler()
    teacher.run(lab, rows, set(), max_items=3, out_path=tmp_path / "t.jsonl")
    assert sum(len(c[1]) for c in lab.calls) == 3


def test_missing_key_exits_2(monkeypatch, capsys):
    class NoKey:
        available = False
    monkeypatch.setitem(sys.modules, "sentiment.llm", types.SimpleNamespace(GeminiLabeler=NoKey))
    assert teacher.main([]) == 2
    assert "GEMINI_API_KEY missing" in capsys.readouterr().out


def test_relabel_only_writes_prompt_version(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher.time, "sleep", lambda s: None)
    lab = FakeLabeler()
    lab.prompt_version = 2
    out = tmp_path / "t.jsonl"
    teacher.run(lab, [_row("a", "AAA", 1), _row("b", "AAA", 2)], set(), min_interval=0, out_path=out, only={"b"})
    assert [(r["id"], r["pv"]) for r in read_jsonl(out)] == [("b", 2)]


def test_row_carries_about_only_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher.time, "sleep", lambda s: None)

    class AboutLabeler(FakeLabeler):
        def label(self, company, items):
            out = super().label(company, items)
            out["a0"]["about"] = "mention"
            return out
    out = tmp_path / "t.jsonl"
    teacher.run(AboutLabeler(), [_row("a0", "A", 1), _row("a1", "A", 2)], set(), max_items=10,
                min_interval=0, out_path=out)
    by = {r["id"]: r for r in read_jsonl(out)}
    assert by["a0"]["about"] == "mention" and "about" not in by["a1"]
