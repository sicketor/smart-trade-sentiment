from datetime import datetime, timedelta, timezone

from sentiment.types import Company, RawItem
from train.io import (append_jsonl, company_from_json, company_to_json, item_from_json, item_to_json,
                      read_jsonl)


def _item():
    return RawItem(id="abc", source="hn", channel="news", publisher="Reuters", title="Nvidia beats – עברית",
                   text="body", url="http://x", published_at=datetime(2026, 10, 1, 12, 0, tzinfo=timezone(timedelta(hours=2))),
                   ticker="NVDA", meta={"likes": 3})


def test_item_roundtrip_keeps_timezone():
    it = _item()
    back = item_from_json(item_to_json(it))
    assert back == it
    assert back.published_at.utcoffset() == timedelta(hours=2)


def test_company_roundtrip():
    c = Company(ticker="F", name="FORD MOTOR CO", short_name="Ford", cik=37996, aliases=["Ford Motor"], ambiguous=True)
    assert company_from_json(company_to_json(c)) == c


def test_read_missing_file_is_empty(tmp_path):
    assert list(read_jsonl(tmp_path / "nope.jsonl")) == []


def test_append_and_read_skips_corrupt_and_blank(tmp_path):
    p = tmp_path / "a" / "b" / "x.jsonl"
    append_jsonl(p, [{"a": 1}, {"t": "עברית"}])
    with p.open("a", encoding="utf-8") as f:
        f.write("{broken\n\n[1,2]\n")
    append_jsonl(p, [{"a": 2}])
    assert list(read_jsonl(p)) == [{"a": 1}, {"t": "עברית"}, {"a": 2}]
    assert "עברית" in p.read_text(encoding="utf-8")
