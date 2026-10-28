"""Score every model, the meta-models and Gemini against Tom's gold labels (SPEC-he.md §9).

python -m eval.gold [--min-bucket 65] [--device auto]

Only gold items with bucket >= --min-bucket are scored: the student trains on teacher buckets < 65,
so these are texts it never saw. Writes runs/gold_eval.json and prints a table per group.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from sentiment import config
from sentiment.meta import Meta, group_of
from sentiment.types import CLASS_VALUES
from train.io import CORPUS, GOLD, TEACHER, bucket, latest_teacher, read_jsonl
from train.meta_train import featurize, target_cls

from .metrics import macro_f1, mae


def load(min_bucket: int) -> list[dict]:
    corpus = {}
    for r in read_jsonl(CORPUS):
        corpus.setdefault((r["item"]["id"], r["item"]["ticker"]), r)
    gold = {}
    for g in read_jsonl(GOLD):
        gold[g["id"]] = g          # last write wins (re-labels)
    teacher = {(t["id"], t.get("ticker")): t for t in latest_teacher(read_jsonl(TEACHER))}
    out = []
    for gid, g in gold.items():
        key = (gid, g.get("ticker"))
        if g.get("skipped") or key not in corpus or bucket(gid) < min_bucket:
            continue
        out.append({"row": corpus[key], "label": g, "teacher": teacher.get(key)})
    return out


def _report(y: np.ndarray, cls: np.ndarray, score: np.ndarray) -> dict:
    """y, cls: -2..2 classes; score: expected value on -1..1 (CLASS_VALUES). Means are on -1..1."""
    sign = lambda a: np.sign(a).astype(int)
    rho = float(np.corrcoef(np.argsort(np.argsort(score)), np.argsort(np.argsort(y)))[0, 1]) if len(y) > 2 else 0.0
    return {"n": int(len(y)), "f1_dir": round(macro_f1(sign(y), sign(cls)), 3),
            "acc5": round(float((y == cls).mean()), 3), "mae5": round(mae(y, cls), 3),
            "spearman": round(rho, 3), "mean_pred": round(float(score.mean()), 3),
            "mean_gold": round(float(y.mean()) / 2, 3)}


def evaluate(exs: list[dict]) -> dict:
    learned = Meta()
    heuristic = Meta(models_dir=config.ROOT / "no-such-dir")
    res: dict = {}
    for group in ("text", "social"):
        g = [e for e in exs if group_of(e["item"].channel) == group]
        if not g:
            continue
        y = np.array([target_cls(e["label"]) for e in g])
        rows: dict = {}
        # single models: direction from 3-class argmax, intensity +-1 only
        per_model = defaultdict(list)
        for i, e in enumerate(g):
            for o in e["outputs"]:
                per_model[o.model].append((i, o.probs))
        for name, lst in sorted(per_model.items()):
            idx = np.array([i for i, _ in lst])
            p = np.array([pr for _, pr in lst])
            cls = p.argmax(axis=1) - 1
            rows[name] = _report(y[idx], cls, p[:, 2] - p[:, 0])
        for rname, m in (("heuristic", heuristic), ("learned", learned)):
            if rname == "learned" and group not in m.learned:
                continue
            r = [m.predict(e["item"].channel, e["outputs"], e["features"], e["events"]) for e in g]
            rows[rname] = _report(y, np.array([x.cls for x in r]),
                                  np.array([float(np.dot(x.probs5, CLASS_VALUES)) for x in r]))
        t_idx = [i for i, e in enumerate(g) if e["teacher"]]
        if t_idx:
            tc = np.array([target_cls(g[i]["teacher"]) for i in t_idx])
            rows["gemini"] = _report(y[t_idx], tc, tc / 2.0)
        res[group] = rows
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.gold")
    ap.add_argument("--min-bucket", type=int, default=65)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    exs = load(a.min_bucket)
    if not exs:
        print("no gold items in range: label with python -m labeling.server")
        return 2
    featurize(exs, a.device)
    res = evaluate(exs)
    out = config.ROOT / "runs" / "gold_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for group, rows in res.items():
        print(f"\n[{group}]  {'model':18} {'n':>4} {'f1_dir':>7} {'acc5':>6} {'mae5':>6} {'rho':>6} {'pred':>6} {'gold':>6}")
        for name, r in rows.items():
            print(f"{'':9}{name:18} {r['n']:>4} {r['f1_dir']:>7} {r['acc5']:>6} {r['mae5']:>6} "
                  f"{r['spearman']:>6} {r['mean_pred']:>6} {r['mean_gold']:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
