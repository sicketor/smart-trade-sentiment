import hashlib
import re
from types import SimpleNamespace

import numpy as np

from sentiment.shots import ShotPool


def fake_embed(texts):
    out = np.zeros((len(texts), 64))
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out


def entry(i, text, channel="news", cls=0):
    return {"id": i, "ticker": "AAA", "channel": channel, "text": text, "cls": cls}


def item(i, text, channel="news"):
    return SimpleNamespace(id=i, title="", text=text, channel=channel)


ENTRIES = [
    entry("a", "apple earnings beat revenue growth strong", cls=2),
    entry("b", "oil prices fall on weak demand outlook", cls=-1),
    entry("c", "fed holds rates steady", cls=0),
]


def test_nearest_chosen_first():
    p = ShotPool(ENTRIES, embed=fake_embed)
    out = p.select([item("x", "apple earnings strong revenue")], k_per_item=1)
    assert out[0]["id"] == "a"


def test_same_id_excluded():
    p = ShotPool(ENTRIES, embed=fake_embed)
    out = p.select([item("a", "apple earnings beat revenue growth strong!")], k_per_item=1)
    assert out[0]["id"] != "a"
    assert "a" not in [e["id"] for e in out]


def test_near_dup_excluded():
    p = ShotPool(ENTRIES, embed=fake_embed)
    out = p.select([item("x", "apple earnings beat revenue growth strong")], k_per_item=1)
    assert out[0]["id"] != "a"


def test_cap_respected():
    ents = [entry(str(n), f"word{n} common", cls=0) for n in range(20)]
    p = ShotPool(ents, embed=fake_embed)
    items = [item(f"i{n}", f"word{n} other{n}") for n in range(10)]
    assert len(p.select(items, k_per_item=2, cap=5)) == 5


def test_anchors_fill_missing_cells():
    p = ShotPool(ENTRIES + [entry("d", "social buzz", "social", 1)], embed=fake_embed)
    out = p.select([item("x", "apple earnings strong")], k_per_item=1)
    assert [e["id"] for e in out] == ["a", "b", "c", "d"]  # neighbor a, then anchors for uncovered cells


def test_empty_items_only_anchors():
    p = ShotPool(ENTRIES, embed=fake_embed)
    assert [e["id"] for e in p.select([])] == ["b", "c", "a"]


def test_anchor_skips_item_id():
    ents = [entry("a", "one", cls=2), entry("b", "two", cls=2)]
    p = ShotPool(ents, embed=fake_embed)
    out = p.select([item("a", "zzz")], k_per_item=0)
    assert [e["id"] for e in out] == ["b"]


def test_load_missing_or_invalid(tmp_path):
    assert ShotPool.load(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert ShotPool.load(bad) is None
    empty = tmp_path / "e.json"
    empty.write_text("[]", encoding="utf-8")
    assert ShotPool.load(empty) is None


def test_load_ok(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('[{"id":"a","ticker":"T","channel":"news","text":"hi","cls":1}]', encoding="utf-8")
    assert ShotPool.load(f, embed=fake_embed) is not None


def test_block_format():
    p = ShotPool([entry("a", "héllo", "social", 1)], embed=fake_embed)
    b = p.block([])
    assert b == ("Examples labeled by our analyst, chosen to resemble these items (apply the same standard):\n"
                 '- [social] target AAA: "héllo" -> cls 1\n')
    assert ShotPool([], embed=fake_embed).block([]) == ""
