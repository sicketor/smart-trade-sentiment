"""Benchmark every registry model + the ensemble on the 5 datasets' test splits (SPEC-he.md §9).

python -m eval.benchmark [--datasets tfns,sentfin] [--models finbert,...] [--limit 1000] [--device auto]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sentiment.features import extract
from sentiment.meta import Meta
from sentiment.models.registry import SPECS, specs_for
from sentiment.models.runner import ModelRunner
from sentiment.types import Company, ModelOutput, RawItem

from . import datasets as D
from .metrics import accuracy, ece, macro_f1, mae, to3

log = logging.getLogger("eval.benchmark")
ROOT = Path(__file__).resolve().parent.parent
IDX_TO_LABEL = (-1, 0, 1)
LEAKAGE = [
    "finbert, fin_distilroberta ו-finbert_tone אומנו על Financial PhraseBank (fpb) - הציונים שלהם שם מנופחים.",
    "fintwit (FinTwitBERT) אולי ראה את TFNS - הציון שלו שם עלול להיות מנופח.",
]


def _metrics(y_true, y_pred, probs3=None, y5=None, p5=None) -> dict:
    r = {"macro_f1": round(macro_f1(y_true, y_pred), 4), "accuracy": round(accuracy(y_true, y_pred), 4),
         "n": len(y_true)}
    if probs3 is not None:
        r["ece"] = round(ece(np.asarray(probs3), np.asarray(y_true) + 1), 4)
    if y5 is not None:
        r["mae"] = round(mae(y5, p5), 4)
    return r


def run(datasets: list[str], models: list[str] | None, limit: int | None, device: str = "auto") -> dict:
    runner = ModelRunner(device=device)
    meta = Meta()
    heuristic = Meta.__new__(Meta)   # same class, learned models off -> heuristic path
    heuristic.learned = {}
    out: dict = {"datasets": {}, "results": {}, "generated_at": datetime.now(timezone.utc).isoformat(),
                 "limit": limit}
    failed: set[str] = set()
    now = datetime.now(timezone.utc)
    for ds in datasets:
        exs = D.load(ds, "test", limit=limit)
        y = [e.label3 for e in exs]
        out["datasets"][ds] = {"n": len(exs), "label_dist": {str(k): v for k, v in sorted(Counter(y).items())}}
        probs_of: dict[tuple[str, str], dict[int, tuple]] = {}
        for ch in sorted({e.channel for e in exs}):
            idxs = [i for i, e in enumerate(exs) if e.channel == ch]
            texts = [exs[i].text for i in idxs]
            asp = [exs[i].target or "the company" for i in idxs]
            for spec in specs_for(ch):
                if (models and spec.name not in models) or spec.name in failed:
                    continue
                try:
                    t0 = time.time()
                    if spec.input == "target":   # same format the student was trained on
                        tgt = [f"{exs[i].target} | {exs[i].target}" if exs[i].target else "" for i in idxs]
                        probs = runner.predict(spec.name, texts, tgt)
                    else:
                        probs = runner.predict(spec.name, texts, asp)
                    log.info("%s on %s/%s: %d texts in %.1fs", spec.name, ds, ch, len(texts), time.time() - t0)
                except Exception as e:  # load failure: skip the model, omit its row
                    log.warning("skipping %s: %s", spec.name, e)
                    failed.add(spec.name)
                    continue
                probs_of[(spec.name, ch)] = dict(zip(idxs, probs))
        for name in sorted({k[0] for k in probs_of}):
            sel = [i for i in range(len(exs)) if (name, exs[i].channel) in probs_of]
            p = np.array([probs_of[(name, exs[i].channel)][i] for i in sel])
            yp = [IDX_TO_LABEL[int(k)] for k in p.argmax(axis=1)]
            out["results"].setdefault(name, {})[ds] = _metrics([y[i] for i in sel], yp, p)
        rows = {"heuristic": heuristic}
        # the learned row is only fair when every model it was trained on ran here
        have = {k[0] for k in probs_of}
        need = {k.split(":")[1] for b in meta.learned.values() for k in b["keys"] if k.startswith("m:")}
        if meta.learned and need <= have:
            rows["learned"] = meta
        elif meta.learned:
            log.warning("skipping learned row on %s: missing models %s", ds, sorted(need - have))
        for rname, m in rows.items():
            preds = []
            for i, e in enumerate(exs):
                outs = [ModelOutput(s.name, tuple(probs_of[(s.name, e.channel)][i]))
                        for s in specs_for(e.channel) if (s.name, e.channel) in probs_of]
                if not outs:
                    continue
                t = e.target or "the company"
                item = RawItem(id=e.id, source="benchmark", channel=e.channel, publisher="", title=e.text,
                               text="", url="", published_at=now, ticker="X")
                feats, events, _ = extract(item, Company(ticker="X", name=t, short_name=t, cik=0))
                preds.append((i, m.predict(e.channel, outs, feats, events)))
            if not preds:
                continue
            sel = [i for i, _ in preds]
            yp = [int(np.sign(r.cls)) for _, r in preds]
            p3 = to3(np.array([r.probs5 for _, r in preds]))
            kw = {}
            if ds == "fiqa":
                kw = {"y5": [exs[i].label5 for i in sel], "p5": [r.cls for _, r in preds]}
            out["results"].setdefault(rname, {})[ds] = _metrics([y[i] for i in sel], yp, p3, **kw)
    return out


def write_markdown(res: dict, path: Path) -> None:
    dss = list(res["datasets"])
    L = ["# בנצ'מרק מודלים", "", f"תאריך: {res['generated_at'][:10]}", f"מגבלת דוגמאות לסט: {res['limit']}", "",
         "ציון macro-F1 על פיצול הבדיקה (מודגש = הטוב ביותר בעמודה).", "",
         "| מודל | " + " | ".join(dss) + " |", "|---|" + "---|" * len(dss)]
    best = {d: max((r[d]["macro_f1"] for r in res["results"].values() if d in r), default=None) for d in dss}
    for m, r in res["results"].items():
        cells = []
        for d in dss:
            if d not in r:
                cells.append("-")
                continue
            v = r[d]["macro_f1"]
            cells.append(f"**{v:.2f}**" if v == best[d] else f"{v:.2f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## דליפה אפשרית", ""] + [f"- {x}" for x in LEAKAGE]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(D.DATASETS))
    ap.add_argument("--models", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "runs" / "bench.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    models = [m for m in a.models.split(",") if m] or None
    for m in models or []:
        if m not in SPECS:
            raise SystemExit(f"unknown model {m}")
    res = run(a.datasets.split(","), models, a.limit, a.device)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(res, ROOT / "docs" / "benchmark.md")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
