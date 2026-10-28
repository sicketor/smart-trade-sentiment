"""Train and calibrate the meta-model (SPEC §5.4, §9).

python -m train.meta_train [--force [text|social|all]] [--min-bucket 65] [--folds 5] [--device auto]

Data: Gemini teacher labels + Tom's gold set, joined with the collected corpus. Each item is
re-scored by the runtime models (sqlite cache makes reruns cheap) and featurized exactly as in
the pipeline, so the meta-model sees the same inputs at train and run time.
One model per group ("text" = news+filings, "social"): StandardScaler + multinomial logistic
regression (gold weighted GOLD_WEIGHT), C by stratified CV, then temperature scaling.

Judging (per group) is the gold cross-fit of eval.crossfit: every eligible gold item is scored
once by a model that never saw it or its teacher row. With it, the saved model is fit on all
teacher + all gold rows, its temperature comes from the out-of-fold rows, and it is saved to
models/meta_<group>.joblib when its cross-fit RPS beats the heuristic's and its direction F1 is
at most F1_TOLERANCE below (or the group is in --force; bare --force = all).
A group with fewer than eval.crossfit.MIN_ELIGIBLE eligible gold falls back to the bucket split:
gold < 70 train, 70..84 calibration, >= 85 test; without enough gold there, calibration and test
use a teacher hold-out (hash bucket >= 90). Saved then when it beats the heuristic's test macro-F1.
The report says which path each group took.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from sentiment import config
from sentiment.features import FEATURE_KEYS, extract
from sentiment.meta import Meta, group_of, learned_probs, vector_value
from sentiment.models.registry import specs_for
from sentiment.types import ModelOutput

from .io import CORPUS, GOLD, TEACHER, company_from_json, item_from_json, latest_teacher, read_jsonl

GOLD_WEIGHT = 10.0   # CV on 501 gold items (27/10): w10 best acc5/MAE for text; Gemini labels run ~neutral vs Tom
MIN_TRAIN = 200
MIN_GOLD_EVAL = 30
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
MIN_CALIB = 20
F1_TOLERANCE = 0.01   # cross-fit save rule: direction F1 may trail the heuristic by this much
GROUPS = ("text", "social")


def bucket(item_id: str) -> int:
    from train.io import bucket as _b
    return _b(item_id)


def target_cls(label: dict) -> int:
    """Not about the ticker -> neutral for this ticker."""
    return int(label["cls"]) if label.get("relevant", True) else 0


def keys_for(group: str) -> list[str]:
    channel = "social" if group == "social" else "news"
    models = [s.name for s in specs_for(channel)]
    return [f"m:{m}:{p}" for m in models for p in ("neg", "neu", "pos")] + [f"f:{k}" for k in FEATURE_KEYS]


def load_labeled() -> list[dict]:
    """[{row, label, origin: "gold"|"teacher", split: "train"|"calib"|"test"|"tholdout"}].
    Gold: the latest label per id (re-labels win), dropped when that latest label is a skip; gold
    rows also carry "teacher" (Gemini's label for the same id + ticker, or None) for reports.
    Teacher rows of gold ids are left out: the gold label replaces them."""
    corpus, by_id = {}, {}   # one id can have rows under several tickers: match on (id, ticker)
    for r in read_jsonl(CORPUS):
        corpus.setdefault((r["item"]["id"], r["item"].get("ticker")), r)
        by_id.setdefault(r["item"]["id"], r)

    def row_of(item_id, ticker):
        return corpus.get((item_id, ticker)) or by_id.get(item_id)

    gold: dict[str, dict] = {}
    for g in read_jsonl(GOLD):
        if "id" in g:
            gold[g["id"]] = g
    gold = {gid: g for gid, g in gold.items() if not g.get("skipped")}
    teacher_rows = latest_teacher(read_jsonl(TEACHER))
    teacher_by_key = {(t["id"], t.get("ticker")): t for t in teacher_rows}
    out = []
    for gid, g in gold.items():
        row = row_of(gid, g.get("ticker"))
        if row is not None:
            b = bucket(gid)
            tk = row["item"].get("ticker")
            out.append({"row": row, "label": g, "origin": "gold",
                        "split": "train" if b < 70 else "calib" if b < 85 else "test",
                        "teacher": teacher_by_key.get((gid, tk))})
    seen = set(gold)
    # The student trained on teacher buckets < META_RESERVED: those rows would show the meta-model
    # an over-confident st_finsent (stacking leakage), so they are left out while it is registered.
    from sentiment.models.registry import SPECS
    from train.io import META_RESERVED
    min_bucket = META_RESERVED if "st_finsent" in SPECS else 0
    for t in teacher_rows:
        tid = t["id"]
        row = row_of(tid, t.get("ticker"))
        if tid in seen or row is None or bucket(tid) < min_bucket:
            continue
        seen.add(tid)
        out.append({"row": row, "label": t, "origin": "teacher",
                    "split": "tholdout" if bucket(tid) >= 90 else "train"})
    return out


def featurize(examples: list[dict], device: str) -> None:
    """Adds ex["outputs"], ex["features"], ex["events"] using the runtime models."""
    from sentiment.models.runner import ModelRunner
    runner = ModelRunner(device=device)
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for ex in examples:
        it = item_from_json(ex["row"]["item"])
        ex["item"] = it
        by_key[(it.ticker, it.channel)].append(ex)
    for (_, channel), group in by_key.items():
        company = company_from_json(group[0]["row"]["company"])
        scored = runner.score_items([ex["item"] for ex in group], company, channel)
        for ex in group:
            ex["outputs"] = scored.get(ex["item"].id, [])
            ex["features"], ex["events"], _ = extract(ex["item"], company)


def matrix(examples: list[dict], keys: list[str]) -> np.ndarray:
    return np.array([[vector_value(k, ex["outputs"], ex["features"], None) for k in keys] for ex in examples])


def fit_temperature(bundle: dict, x: np.ndarray, y: np.ndarray) -> float:
    return fit_temperature_multi([(bundle, x, y)])


def fit_temperature_multi(parts: list[tuple[dict, np.ndarray, np.ndarray]]) -> float:
    """One temperature T minimizing the NLL over all rows of all (bundle, x, y) parts, each part
    scored by its own model's logits / T. Used with the cross-fit fold models and their
    out-of-fold rows to calibrate the final model."""
    from scipy.optimize import minimize_scalar

    parts = [(b, x, y) for b, x, y in parts if len(y)]
    n = sum(len(y) for _, _, y in parts)
    if n < MIN_CALIB:   # too few rows to fit T reliably
        return 1.0

    def nll(t: float) -> float:
        total = 0.0
        for b, x, y in parts:
            p = learned_probs({**b, "temperature": t}, x)
            total += float(np.log(np.clip(p[np.arange(len(y)), y + 2], 1e-9, 1)).sum())
        return -total / n

    return float(minimize_scalar(nll, bounds=(0.25, 5.0), method="bounded").x)


def metrics(p5: np.ndarray, y: np.ndarray) -> dict:
    from eval.metrics import ece, macro_f1, mae
    pred = p5.argmax(axis=1) - 2
    return {"n": int(len(y)), "macro_f1_3": round(macro_f1(np.sign(y), np.sign(pred)), 4),
            "mae_5": round(mae(y, pred), 4), "ece_5": round(ece(p5, y + 2), 4)}


def fit_bundle(train: list[dict], calib: list[dict], keys: list[str]) -> dict:
    """StandardScaler + multinomial LR on `train` (gold weighted GOLD_WEIGHT), C by stratified CV,
    temperature fitted on `calib` (T = 1 under MIN_CALIB rows). -> {"model", "keys", "temperature", "C"}"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, y = matrix(train, keys), np.array([target_cls(e["label"]) for e in train])
    w = np.array([GOLD_WEIGHT if e["origin"] == "gold" else 1.0 for e in train])
    counts = np.bincount(y + 2)
    n_splits = max(2, min(5, int(counts[counts > 0].min())))   # rarest class must fit every fold
    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        {"logisticregression__C": list(C_GRID)},
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0), scoring="f1_macro")
    search.fit(x, y, logisticregression__sample_weight=w)
    bundle = {"model": search.best_estimator_, "keys": keys, "temperature": 1.0,
              "C": search.best_params_["logisticregression__C"]}
    if len(calib) >= MIN_CALIB:
        xc, yc = matrix(calib, keys), np.array([target_cls(e["label"]) for e in calib])
        bundle["temperature"] = round(fit_temperature(bundle, xc, yc), 4)
    return bundle


def _save(group: str, bundle: dict) -> None:
    import joblib
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODELS_DIR / f"meta_{group}.joblib")


def beats_heuristic(cv: dict) -> bool:
    """Cross-fit save rule: lower RPS than the heuristic without losing more than 0.01 direction F1."""
    raw = cv["_raw"]
    return raw["learned_rps"] < raw["heuristic_rps"] and raw["learned_f1"] >= raw["heuristic_f1"] - F1_TOLERANCE


def train_group(group: str, exs: list[dict], force: bool, cv: dict | None = None) -> dict:
    """cv: this group's eval.crossfit result. With it (and no "fallback" in it) the saved model is
    fit on all teacher + all gold rows and judged by the cross-fit; otherwise the bucket split below."""
    if cv is not None and "fallback" not in cv:
        return _train_group_crossfit(group, exs, force, cv)
    keys = keys_for(group)
    train = [e for e in exs if e["split"] == "train"]
    gold_calib = [e for e in exs if e["split"] == "calib"]
    gold_test = [e for e in exs if e["split"] == "test"]
    holdout = [e for e in exs if e["split"] == "tholdout"]
    use_gold = len(gold_calib) >= MIN_GOLD_EVAL and len(gold_test) >= MIN_GOLD_EVAL
    if use_gold:
        calib, test, eval_on = gold_calib, gold_test, "gold"
    else:   # half of the teacher hold-out calibrates, the other half tests
        calib = [e for e in holdout if bucket(e["item"].id) < 95]
        test = [e for e in holdout if bucket(e["item"].id) >= 95]
        train += gold_calib + gold_test   # too little gold to evaluate on: use it to train
        eval_on = "teacher-holdout"
    if len(train) < MIN_TRAIN or len(test) < 20:
        return {"group": group, "skipped": f"not enough data (train {len(train)}, test {len(test)})"}

    bundle = fit_bundle(train, calib, keys)
    xt, yt = matrix(test, keys), np.array([target_cls(e["label"]) for e in test])
    learned_m = metrics(learned_probs(bundle, xt), yt)
    heur = Meta(models_dir=config.ROOT / "no-such-dir")
    hp = np.array([heur.predict(e["item"].channel, e["outputs"], e["features"], e["events"]).probs5 for e in test])
    heuristic_m = metrics(hp, yt)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    bundle |= {"version": f"lr-{date}-n{len(train)}", "eval_on": eval_on,
               "metrics": {"learned": learned_m, "heuristic": heuristic_m},
               "counts": {"train": len(train), "calib": len(calib), "test": len(test),
                          "gold_train": sum(e["origin"] == "gold" for e in train)}}
    if cv is not None:   # cross-fit requested but not possible for this group: say why
        bundle["crossfit"] = {"fallback": cv["fallback"]}
    saved = force or learned_m["macro_f1_3"] > heuristic_m["macro_f1_3"]
    if saved:
        _save(group, bundle)
    return {"group": group, "saved": saved, **{k: v for k, v in bundle.items() if k not in ("model", "keys")}}


def _train_group_crossfit(group: str, exs: list[dict], force: bool, cv: dict) -> dict:
    """Final model = fit_bundle on every teacher + gold row of the group (no calib split). Its
    temperature is fitted, by NLL, on the cross-fit's out-of-fold rows: each eligible gold item is
    scored by the fold model that never saw it (raw logits / T), and one T is chosen over all of
    them (fit_temperature_multi). The final model's own logits on gold are in-sample, so they
    cannot calibrate it; the fold models are the same recipe on ~(folds-2)/folds of the gold."""
    keys = keys_for(group)
    train = list(exs)
    if len(train) < MIN_TRAIN:
        return {"group": group, "skipped": f"not enough data (train {len(train)})"}
    bundle = fit_bundle(train, [], keys)
    bundle["temperature"] = round(fit_temperature_multi(cv["_parts"]), 4)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    n_oof = int(cv["n"])
    bundle |= {"version": f"lr-{date}-n{len(train)}", "eval_on": "gold-crossfit",
               "metrics": cv["_metrics"],
               "counts": {"train": len(train), "calib": n_oof, "test": n_oof,
                          "gold_train": sum(e["origin"] == "gold" for e in train)},
               "crossfit": {k: v for k, v in cv.items() if not k.startswith("_")}}
    saved = force or beats_heuristic(cv)
    if saved:
        _save(group, bundle)
    return {"group": group, "saved": saved, **{k: v for k, v in bundle.items() if k not in ("model", "keys")}}


def forced_groups(value: str | None) -> set[str]:
    if value is None:
        return set()
    return set(GROUPS) if value == "all" else {value}


def main(argv: list[str] | None = None) -> int:
    from eval.crossfit import DEFAULT_MIN_BUCKET, crossfit, load_student_gold, prepare_student

    p = argparse.ArgumentParser(prog="train.meta_train")
    p.add_argument("--force", nargs="?", const="all", default=None, choices=(*GROUPS, "all"),
                   help="save this group (or all) even if it does not beat the heuristic")
    p.add_argument("--min-bucket", type=int, default=DEFAULT_MIN_BUCKET,
                   help="gold buckets >= this are cross-fit eligible; lower ones always train")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--student-oof", help="out-of-fold stage-2 student scores for gold "
                                         "(default when the active student is stage-2: runs/student_gold_oof.json)")
    args = p.parse_args(argv)
    forced = forced_groups(args.force)
    labeled = load_labeled()
    if not labeled:
        print("no labeled items: run train.corpus, then train.teacher and/or labeling.server")
        return 2
    featurize(labeled, args.device)
    # a stage-2 student saw every gold id: gold rows train and test on its out-of-fold scores instead
    s2_gold = prepare_student([e for e in labeled if e["origin"] == "gold"], args.student_oof)
    student_gold = s2_gold if s2_gold is not None else load_student_gold()
    report = []
    for group in GROUPS:
        exs = [e for e in labeled if group_of(e["item"].channel) == group]
        gold = [e for e in exs if e["origin"] == "gold"]
        teacher = [e for e in exs if e["origin"] == "teacher"]
        cv = crossfit({group: gold}, teacher, args.folds, min_bucket=args.min_bucket,
                      student_gold=student_gold)[group]
        report.append(train_group(group, exs, group in forced, cv))
    out = config.ROOT / "runs" / "meta_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
