"""Compare Tom's blind stock grades (labeling.stocks) with the engine's composite grade (SPEC §9.3, route 1).

    python -m eval.stock_grades [--batch data/gold/stock_batch.json] [--grades data/gold/stock_grades.jsonl]
                                [--out runs/stock_grades.json]

Prints agreement (exact, within one grade, same direction), Spearman rho, the mean gap, a 7x7 table, and the
z cuts that would match Tom best: each cut is the midpoint between the median engine z of two neighbouring
Tom grades. The cuts are fitted on the same stocks they are scored on, so their agreement is optimistic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from labeling.stocks import BATCH, GRADES, latest_grades
from sentiment.aggregate import grade_from_cuts, load_norm

G = list(range(-3, 4))


def pairs(batch: dict, grades: dict[str, dict]) -> list[dict]:
    stocks = {s["ticker"]: s for s in batch["stocks"]}
    out = []
    for t, r in sorted(grades.items()):
        s = stocks.get(t)
        e = (s or {}).get("engine") or {}
        if r.get("grade") is None or e.get("grade") is None:
            continue
        out.append({"ticker": t, "tom": int(r["grade"]), "engine": int(e["grade"]), "z": e.get("z"),
                    "score": e.get("score"), "note": r.get("note", "")})
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort(kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):           # average ranks for ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def agreement(tom, eng) -> dict:
    t, e = np.asarray(tom), np.asarray(eng)
    return {"n": int(len(t)), "exact": float(np.mean(t == e)), "within1": float(np.mean(abs(t - e) <= 1)),
            "direction": float(np.mean(np.sign(t) == np.sign(e))), "mean_gap": float(np.mean(e - t)),
            "spearman": spearman(t, e)}


def fit_cuts(rows: list[dict], current: list[float]) -> list[float]:
    """Midpoints between the median z of neighbouring Tom grades; a missing grade keeps the current cut.
    Forced non-decreasing."""
    med = {g: float(np.median([r["z"] for r in rows if r["tom"] == g]))
           for g in G if any(r["tom"] == g and r["z"] is not None for r in rows)}
    cuts = []
    for i in range(6):
        lo, hi = G[i], G[i + 1]
        below = [med[g] for g in G[:i + 1] if g in med]
        above = [med[g] for g in G[i + 1:] if g in med]
        c = (med[lo] + med[hi]) / 2 if lo in med and hi in med else (
            (below[-1] + above[0]) / 2 if below and above else current[i])
        cuts.append(max(c, cuts[-1]) if cuts else c)
    return [round(c, 3) for c in cuts]


def confusion(tom, eng) -> list[list[int]]:
    m = [[0] * 7 for _ in G]
    for t, e in zip(tom, eng):
        m[t + 3][e + 3] += 1
    return m


def report(batch: dict, grades: dict[str, dict]) -> dict:
    rows = pairs(batch, grades)
    if not rows:
        return {"n": 0}
    norm = load_norm() or {}
    current = ((norm.get("groups") or {}).get("composite") or {}).get("cuts")
    tom, eng = [r["tom"] for r in rows], [r["engine"] for r in rows]
    res = {"n": len(rows), "skipped": sum(1 for r in grades.values() if r.get("grade") is None),
           "engine": agreement(tom, eng), "confusion": confusion(tom, eng), "current_cuts": current,
           "tom_dist": {g: tom.count(g) for g in G}, "engine_dist": {g: eng.count(g) for g in G}}
    zr = [r for r in rows if r["z"] is not None]
    if current and len(zr) >= 4:
        res["z_by_tom"] = {g: round(float(np.median([r["z"] for r in zr if r["tom"] == g])), 3)
                           for g in G if any(r["tom"] == g for r in zr)}
        cuts = fit_cuts(zr, [float(c) for c in current])
        refit = [grade_from_cuts(r["z"], cuts) for r in zr]
        res["fitted_cuts"] = cuts
        res["fitted"] = agreement([r["tom"] for r in zr], refit)   # in-sample: optimistic
    res["biggest_gaps"] = sorted(rows, key=lambda r: -abs(r["engine"] - r["tom"]))[:8]
    return res


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def print_report(res: dict) -> None:
    if not res.get("n"):
        print("no graded stocks yet")
        return
    a = res["engine"]
    print(f"stocks graded: {res['n']} (skipped {res['skipped']})")
    print(f"engine vs Tom: exact {_pct(a['exact'])}  within one {_pct(a['within1'])}  "
          f"same direction {_pct(a['direction'])}  spearman {a['spearman'] if a['spearman'] is None else round(a['spearman'], 2)}"
          f"  mean gap (engine - Tom) {a['mean_gap']:+.2f}")
    print("\nrows = Tom, columns = engine")
    print("      " + "".join(f"{g:>4}" for g in G))
    for g, row in zip(G, res["confusion"]):
        print(f"{g:>4}  " + "".join(f"{v:>4}" for v in row))
    if "fitted_cuts" in res:
        print(f"\nmedian engine z per Tom grade: {res['z_by_tom']}")
        print(f"current cuts: {[round(c, 3) for c in res['current_cuts']]}")
        print(f"fitted cuts:  {res['fitted_cuts']}")
        f = res["fitted"]
        print(f"with fitted cuts (in-sample): exact {_pct(f['exact'])}  within one {_pct(f['within1'])}  "
              f"same direction {_pct(f['direction'])}")
    print("\nbiggest gaps:")
    for r in res["biggest_gaps"]:
        print(f"  {r['ticker']:<6} Tom {r['tom']:+d}  engine {r['engine']:+d}  z={r['z']}  {r['note'][:80]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=Path, default=BATCH)
    ap.add_argument("--grades", type=Path, default=GRADES)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    batch = json.loads(a.batch.read_text(encoding="utf-8"))
    res = report(batch, latest_grades(a.grades, str(batch["id"])))
    print_report(res)
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
