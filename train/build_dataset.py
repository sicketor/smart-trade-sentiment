"""Build the student's training set (SPEC-he.md §5.0): teacher + public + StockTwits, soft labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from sentiment.config import ROOT

try:  # parallel contracts; tests monkeypatch these module attributes
    from train import io as tio
except Exception:  # pragma: no cover
    tio = None
try:
    from eval import datasets as ds
except Exception:  # pragma: no cover
    ds = None

MAP3 = {
    -1: [0.35, 0.65, 0.0, 0.0, 0.0],
    0: [0.0, 0.0, 1.0, 0.0, 0.0],
    1: [0.0, 0.0, 0.0, 0.65, 0.35],
}
ST_SOFT = {"Bullish": [0.0, 0.0, 0.15, 0.6, 0.25], "Bearish": [0.25, 0.6, 0.15, 0.0, 0.0]}
_WS = re.compile(r"\s+")


def smooth5(c: int) -> list[float]:
    """5-class label (-2..2) -> 0.85 on c, 0.15 split between existing neighbours."""
    i = int(c) + 2
    if not 0 <= i <= 4:
        raise ValueError(f"class out of range: {c}")
    out = [0.0] * 5
    out[i] = 0.85
    nb = [j for j in (i - 1, i + 1) if 0 <= j <= 4]
    for j in nb:
        out[j] = 0.15 / len(nb)
    return out


def map3(label3: int) -> list[float]:
    return list(MAP3[int(label3)])


MIN_TEACHER_VAL = 300


def norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "").lower()).strip()


def is_val(row_id: str) -> bool:
    return int(hashlib.sha1(row_id.encode("utf-8")).hexdigest(), 16) % 20 == 0


def item_text(item) -> str:
    return item.title if not item.text else f"{item.title}. {item.text[:600]}"


def corpus_target(ticker: str, company) -> str:
    return f"{ticker} | {company.short_name}"


WEAK_PATH = ROOT / "data" / "weak" / "stocktwits_train.jsonl"


def build(out: str | Path = "data/train", max_stocktwits: int = 20000, public_weight: float = 0.5,
          weak_path: str | Path = WEAK_PATH, weak_weight: float = 0.3) -> dict:
    out = Path(out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    test_set = ds.test_texts()
    gold_ids = set()
    if Path(tio.GOLD).exists():
        gold_ids = {r["id"] for r in tio.read_jsonl(tio.GOLD)}
    (out / "gold_ids.json").write_text(json.dumps(sorted(gold_ids)), encoding="utf-8")

    stats = {"rows": Counter(), "class_mass": [0.0] * 5, "dropped_leakage": 0, "dropped_gold": 0,
             "reserved_for_meta": 0}

    # corpus: first row per (id, ticker) and per id, skipping gold
    corpus: dict = {}      # (id, ticker) -> (item, company)
    by_id: dict = {}       # id -> first (item, company)
    companies: dict = {}   # ticker -> company, first seen
    gold_seen: set = set()
    if Path(tio.CORPUS).exists():
        for r in tio.read_jsonl(tio.CORPUS):
            iid = r["item"]["id"]
            if iid in gold_ids:
                if iid not in gold_seen:
                    gold_seen.add(iid)
                    stats["dropped_gold"] += 1
                continue
            key = (iid, r["item"].get("ticker"))
            if key in corpus:
                continue
            pair = (tio.item_from_json(r["item"]), tio.company_from_json(r["company"]))
            corpus[key] = pair
            by_id.setdefault(iid, pair)
            if r["item"].get("ticker") is not None:
                companies.setdefault(r["item"]["ticker"], pair[1])

    rows: list[dict] = []
    teacher_ids: set = set()

    def add(target, text, soft, weight, source, rid, title=""):
        if norm_text(text) in test_set or (title and norm_text(title) in test_set):
            stats["dropped_leakage"] += 1
            return
        rows.append({"target": target, "text": text, "soft": soft, "weight": weight, "source": source, "id": rid})

    # teacher
    seen = set()
    if Path(tio.TEACHER).exists():
        for t in tio.latest_teacher(tio.read_jsonl(tio.TEACHER)):
            pair = corpus.get((t["id"], t.get("ticker"))) or by_id.get(t["id"])
            if pair is None or t["id"] in gold_ids:
                continue
            key = (t["id"], t.get("ticker"))
            if key in seen:
                continue
            seen.add(key)
            teacher_ids.add(t["id"])
            if tio.bucket(t["id"]) >= tio.META_RESERVED:   # kept for the meta-model only
                stats["reserved_for_meta"] += 1
                continue
            item, _ = pair
            ticker = t.get("ticker") or item.ticker
            company = companies.get(ticker) or pair[1]
            cls = int(t["cls"]) if t.get("relevant", True) else 0
            add(corpus_target(ticker, company), item_text(item), smooth5(cls), 1.0, "teacher", t["id"],
                title=item.title)

    # public
    public_rows_start = len(rows)
    for name in ds.DATASETS:
        for ex in ds.load(name, "train"):
            soft = smooth5(ex.label5) if name == "fiqa" else map3(ex.label3)
            target = f"{ex.target} | {ex.target}" if ex.target else ""
            add(target, ex.text, soft, public_weight, f"public:{name}", f"{name}:{ex.id}")
    public_rows = rows[public_rows_start:]

    # stocktwits: the filtered weak-label file when present, else the corpus rows
    weak = list(tio.read_jsonl(weak_path)) if Path(weak_path).exists() else []
    if weak:
        wseen, cand = set(), []
        for w in weak:
            if w["id"] in gold_ids:
                stats["dropped_gold"] += 1
                continue
            if w["id"] in teacher_ids or w["ticker"] not in companies or (w["id"], w["ticker"]) in wseen:
                continue
            wseen.add((w["id"], w["ticker"]))
            cand.append(w)
        cand.sort(key=lambda w: w.get("created_at") or "", reverse=True)
        for w in cand[:max_stocktwits]:
            soft = ST_SOFT["Bullish"] if int(w["label"]) == 1 else ST_SOFT["Bearish"]
            add(corpus_target(w["ticker"], companies[w["ticker"]]), w["text"], list(soft), weak_weight,
                "weak_stocktwits", w["id"], title=w["text"])
    else:
        st = []
        for (iid, _tk), pair in corpus.items():
            if iid in teacher_ids:
                continue
            item, company = pair
            if item.source == "stocktwits" and item.meta.get("user_label") in ST_SOFT:
                st.append((item, company))
        st.sort(key=lambda p: p[0].published_at, reverse=True)
        for item, company in st[:max_stocktwits]:
            add(corpus_target(item.ticker, company), item_text(item), list(ST_SOFT[item.meta["user_label"]]),
                0.3, "stocktwits", item.id, title=item.title)

    # split: teacher val = buckets 65..69; topped up with 5% of public rows while it is small
    val = [r for r in rows if r["source"] == "teacher" and 65 <= tio.bucket(r["id"]) < tio.META_RESERVED]
    if len(val) < MIN_TEACHER_VAL:
        print(f"WARNING: only {len(val)} teacher val rows; adding 5% of public rows to validation")
        val += [r for r in public_rows if is_val(r["id"])]
    val_ids = {id(r) for r in val}
    train = [r for r in rows if id(r) not in val_ids]

    for name, data in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for r in rows:
        stats["rows"][r["source"]] += 1
        for k, p in enumerate(r["soft"]):
            stats["class_mass"][k] += p * r["weight"]
    stats["rows"] = dict(stats["rows"])
    stats["n_train"] = len(train)
    stats["n_val"] = len(val)
    stats["class_mass"] = [round(x, 2) for x in stats["class_mass"]]
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--max-stocktwits", type=int, default=20000)
    ap.add_argument("--public-weight", type=float, default=0.5)
    ap.add_argument("--weak-path", default=str(WEAK_PATH))
    ap.add_argument("--weak-weight", type=float, default=0.3)
    a = ap.parse_args()
    build(a.out, a.max_stocktwits, a.public_weight, a.weak_path, a.weak_weight)


if __name__ == "__main__":
    main()
