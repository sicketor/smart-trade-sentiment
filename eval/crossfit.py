"""Gold cross-fit: every eligible gold item scored once by a meta-model that never saw it (SPEC-he.md §9).

python -m eval.crossfit [--min-bucket 65] [--folds 5] [--device auto] [--no-student-gold]

Eligible gold = the latest gold label per id, not skipped, present in the corpus, bucket >= min_bucket.
The default 65 holds until the student is retrained without gold ids: the student trained on teacher
buckets < 65, so for gold items there it saw the text (with Gemini's label) and its st_finsent feature
is optimistic. Gold below min_bucket is never scored, only used for training ("always-train"), unless its id is in
models/st-finsent/gold_ids.json (gold ids the student never saw): those are eligible at any bucket.

Fold of an eligible item = bucket % folds. For fold k, per group:
  train = teacher rows whose id is not a gold id (a gold label replaces its teacher row, so no teacher
          row of a fold-k id can enter) + always-train gold + eligible gold outside folds k and k+1,
  calib = eligible gold of fold (k + 1) % folds: the fold model's temperature, never fold k,
  test  = eligible gold of fold k.
Training is meta_train.fit_bundle, the same code path as the saved model.

Report per group: n, f1_dir, acc5, mae5, spearman (eval.gold._report) for every row; for the 5-class
rows (learned_cv, heuristic) also rps, log_loss, brier, ece_eqmass, ece_floor and 95% bootstrap CIs on
f1_dir and rps; learned_cv vs heuristic paired (bootstrap on f1_dir and rps, McNemar on direction).
Single models (3-class argmax) and Gemini are scored on the same rows, as in eval.gold.
Writes runs/crossfit.json and prints a table per group.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from sentiment import config
from sentiment.meta import Meta, group_of, learned_probs
from sentiment.types import CLASS_VALUES
from train.io import bucket
from train.meta_train import (GROUPS, MIN_TRAIN, featurize, fit_bundle, keys_for, load_labeled, matrix,
                              metrics, target_cls)

from .gold import _report
from .metrics import (bootstrap_ci, brier5, ece_eqmass, ece_floor, log_loss5, mcnemar, paired_bootstrap,
                      rps)

DEFAULT_MIN_BUCKET = 65
STUDENT_GOLD = config.ROOT / "models" / "st-finsent" / "gold_ids.json"   # gold ids the student never saw
MIN_ELIGIBLE = 60    # fewer eligible gold in a group -> meta_train keeps the bucket split for it
HEURISTIC_DIR = config.ROOT / "no-such-dir"


def _id(ex: dict) -> str:
    return ex["item"].id


def load_student_gold(path=None) -> set[str]:
    """Gold ids the student was trained without (written by train.build_dataset); missing file -> empty set."""
    p = STUDENT_GOLD if path is None else path
    try:
        return set(json.loads(open(p, encoding="utf-8").read()))
    except (OSError, ValueError):
        return set()


def split_gold(gold: list[dict], min_bucket: int, student_gold=frozenset()) -> tuple[list[dict], list[dict]]:
    """-> (eligible, always_train); eligible = bucket >= min_bucket or id in student_gold"""
    def ok(e):
        return bucket(_id(e)) >= min_bucket or _id(e) in student_gold
    return [e for e in gold if ok(e)], [e for e in gold if not ok(e)]


def fold_splits(gold: list[dict], teacher: list[dict], folds: int, min_bucket: int,
                student_gold=frozenset()) -> list[dict]:
    """[{"fold", "train", "calib", "test"}] for every fold with test rows (see module docstring)."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    eligible, always = split_gold(gold, min_bucket, student_gold)
    gold_ids = {_id(e) for e in gold}
    pool = [t for t in teacher if _id(t) not in gold_ids]
    by_fold: dict[int, list[dict]] = defaultdict(list)
    for e in eligible:
        by_fold[bucket(_id(e)) % folds].append(e)
    out = []
    for k in range(folds):
        if not by_fold[k]:
            continue
        c = (k + 1) % folds
        train_gold = always + [e for f in range(folds) if f not in (k, c) for e in by_fold[f]]
        out.append({"fold": k, "train": pool + train_gold, "calib": list(by_fold[c]), "test": list(by_fold[k])})
    return out


def _f1_dir(y: np.ndarray, cls: np.ndarray) -> float:
    """Macro-F1 over direction (-1, 0, 1); numpy twin of eval.metrics.macro_f1 (zero_division=0),
    fast enough for the bootstrap."""
    t, p = np.sign(y), np.sign(cls)
    f1 = []
    for c in (-1, 0, 1):
        tp = int(((t == c) & (p == c)).sum())
        denom = 2 * tp + int(((t != c) & (p == c)).sum()) + int(((t == c) & (p != c)).sum())
        f1.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1))


def _r(v: float, nd: int = 4) -> float:
    return round(float(v), nd)


def _five(y: np.ndarray, p5: np.ndarray, n_boot: int, seed: int) -> dict:
    cls = p5.argmax(axis=1) - 2
    row = _report(y, cls, p5 @ CLASS_VALUES)
    row |= {"rps": _r(rps(p5, y)), "log_loss": _r(log_loss5(p5, y)), "brier": _r(brier5(p5, y)),
            "ece_eqmass": _r(ece_eqmass(p5, y)), "ece_floor": _r(ece_floor(p5, seed=seed)),
            "f1_dir_ci": [_r(v, 3) for v in bootstrap_ci(_f1_dir, y, cls, n=n_boot, seed=seed)],
            "rps_ci": [_r(v) for v in bootstrap_ci(rps, p5, y, n=n_boot, seed=seed)]}
    return row


def _single_models(rows: list[dict], y: np.ndarray) -> dict:
    """Each runtime model on the rows it scored: direction from its 3-class argmax (as eval.gold)."""
    per_model: dict[str, list] = defaultdict(list)
    for i, e in enumerate(rows):
        for o in e["outputs"]:
            per_model[o.model].append((i, o.probs))
    out = {}
    for name, lst in sorted(per_model.items()):
        idx = np.array([i for i, _ in lst])
        p = np.array([pr for _, pr in lst], dtype=float)
        out[name] = _report(y[idx], p.argmax(axis=1) - 1, p[:, 2] - p[:, 0])
    t_idx = [i for i, e in enumerate(rows) if e.get("teacher")]
    if t_idx:
        tc = np.array([target_cls(rows[i]["teacher"]) for i in t_idx])
        out["gemini"] = _report(y[t_idx], tc, tc / 2.0)
    return out


def crossfit_group(group: str, gold: list[dict], teacher: list[dict], folds: int = 5,
                   min_bucket: int = DEFAULT_MIN_BUCKET, n_boot: int = 2000, seed: int = 0,
                   student_gold=frozenset()) -> dict:
    """One group's cross-fit. Keys starting with "_" hold arrays / fold models for meta_train:
    _parts [(fold bundle, x_test, y_test)], _raw (unrounded save-rule inputs), _metrics (bundle format)."""
    eligible, always = split_gold(gold, min_bucket, student_gold)
    via_file = sum(bucket(_id(e)) < min_bucket for e in eligible)
    head = {"group": group, "n": len(eligible), "n_always_train": len(always), "folds": folds,
            "min_bucket": min_bucket, "n_via_student_gold": via_file}
    if len(eligible) < MIN_ELIGIBLE:
        return head | {"fallback": f"{len(eligible)} eligible gold (< {MIN_ELIGIBLE}): bucket-split evaluation"}
    splits = fold_splits(gold, teacher, folds, min_bucket, student_gold)
    if min(len(s["train"]) for s in splits) < MIN_TRAIN:
        return head | {"fallback": f"a fold has under {MIN_TRAIN} training rows: bucket-split evaluation"}

    keys = keys_for(group)
    rows: list[dict] = []
    p_learned, parts, fold_info = [], [], []
    for s in splits:
        bundle = fit_bundle(s["train"], s["calib"], keys)
        xt = matrix(s["test"], keys)
        yt = np.array([target_cls(e["label"]) for e in s["test"]])
        p_learned.append(learned_probs(bundle, xt))
        parts.append((bundle, xt, yt))
        rows.extend(s["test"])
        fold_info.append({"fold": s["fold"], "train": len(s["train"]), "calib": len(s["calib"]),
                          "test": len(s["test"]), "C": bundle["C"], "temperature": bundle["temperature"]})
    y = np.array([target_cls(e["label"]) for e in rows])
    pl = np.vstack(p_learned)
    heur = Meta(models_dir=HEURISTIC_DIR)
    ph = np.array([heur.predict(e["item"].channel, e["outputs"], e["features"], e["events"]).probs5
                   for e in rows], dtype=float)

    models = {"learned_cv": _five(y, pl, n_boot, seed), "heuristic": _five(y, ph, n_boot, seed)}
    models |= _single_models(rows, y)
    cl, ch = pl.argmax(axis=1) - 2, ph.argmax(axis=1) - 2
    f1p = paired_bootstrap(_f1_dir, (y, cl), (y, ch), n=n_boot, seed=seed)
    rpsp = paired_bootstrap(rps, (pl, y), (ph, y), n=n_boot, seed=seed)
    paired = {"f1_dir": {k: _r(v) for k, v in f1p.items()}, "rps": {k: _r(v) for k, v in rpsp.items()},
              "mcnemar_dir_p": _r(mcnemar(np.sign(cl) == np.sign(y), np.sign(ch) == np.sign(y)))}
    gold_ids = {_id(e) for e in gold}
    return head | {
        "n_teacher": sum(_id(t) not in gold_ids for t in teacher),
        "fold_info": fold_info, "models": models, "learned_vs_heuristic": paired,
        "_parts": parts,
        "_raw": {"learned_rps": rps(pl, y), "heuristic_rps": rps(ph, y),
                 "learned_f1": _f1_dir(y, cl), "heuristic_f1": _f1_dir(y, ch)},
        "_metrics": {"learned": metrics(pl, y), "heuristic": metrics(ph, y)},
    }


def crossfit(examples_by_group: dict[str, list[dict]], teacher_examples: list[dict], folds: int = 5, *,
             min_bucket: int = DEFAULT_MIN_BUCKET, n_boot: int = 2000, seed: int = 0,
             student_gold=frozenset()) -> dict:
    """examples_by_group: {group: featurized gold examples}; teacher_examples: featurized teacher
    examples of any group (split by channel here). -> {group: crossfit_group result}"""
    out = {}
    for group, gold in examples_by_group.items():
        teacher = [t for t in teacher_examples if group_of(t["item"].channel) == group]
        out[group] = crossfit_group(group, gold, teacher, folds, min_bucket, n_boot, seed, student_gold)
    return out


def public(res: dict) -> dict:
    """Drops the "_" keys (fold models, arrays) so the result is JSON."""
    return {g: {k: v for k, v in r.items() if not k.startswith("_")} for g, r in res.items()}


def _print(res: dict) -> None:
    for group, r in res.items():
        if "fallback" in r:
            print(f"\n[{group}]  n={r['n']}  {r['fallback']}")
            continue
        print(f"\n[{group}]  n={r['n']} eligible, {r['n_always_train']} always-train gold, {r['folds']} folds"
              f"  ({r.get('n_via_student_gold', 0)} eligible via student gold_ids)")
        print(f"{'':9}{'model':18} {'n':>4} {'f1_dir':>7} {'f1 95% CI':>13} {'acc5':>6} {'mae5':>6} {'rho':>6} "
              f"{'rps':>7} {'rps 95% CI':>15} {'logloss':>7} {'brier':>6} {'ece':>6} {'floor':>6}")
        for name, m in r["models"].items():
            line = (f"{'':9}{name:18} {m['n']:>4} {m['f1_dir']:>7} "
                    f"{'':>13} {m['acc5']:>6} {m['mae5']:>6} {m['spearman']:>6}")
            if "rps" in m:
                lo, hi = m["f1_dir_ci"]
                rlo, rhi = m["rps_ci"]
                line = (f"{'':9}{name:18} {m['n']:>4} {m['f1_dir']:>7} {f'[{lo}, {hi}]':>13} {m['acc5']:>6} "
                        f"{m['mae5']:>6} {m['spearman']:>6} {m['rps']:>7} {f'[{rlo}, {rhi}]':>15} "
                        f"{m['log_loss']:>7} {m['brier']:>6} {m['ece_eqmass']:>6} {m['ece_floor']:>6}")
            print(line)
        p = r["learned_vs_heuristic"]
        print(f"{'':9}learned_cv - heuristic: f1_dir {p['f1_dir']['diff']:+.4f} [{p['f1_dir']['lo']:+.4f}, "
              f"{p['f1_dir']['hi']:+.4f}]  rps {p['rps']['diff']:+.4f} [{p['rps']['lo']:+.4f}, "
              f"{p['rps']['hi']:+.4f}]  McNemar p={p['mcnemar_dir_p']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.crossfit")
    ap.add_argument("--min-bucket", type=int, default=DEFAULT_MIN_BUCKET)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-student-gold", action="store_true", help="ignore models/st-finsent/gold_ids.json")
    a = ap.parse_args(argv)
    labeled = load_labeled()
    if not any(e["origin"] == "gold" for e in labeled):
        print("no gold items: label with python -m labeling.server")
        return 2
    featurize(labeled, a.device)
    by_group = {g: [e for e in labeled if e["origin"] == "gold" and group_of(e["item"].channel) == g] for g in GROUPS}
    teacher = [e for e in labeled if e["origin"] == "teacher"]
    res = public(crossfit({g: v for g, v in by_group.items() if v}, teacher, a.folds, min_bucket=a.min_bucket,
                           student_gold=set() if a.no_student_gold else load_student_gold()))
    out = config.ROOT / "runs" / "crossfit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    _print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
