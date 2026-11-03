"""A/B the Gemini teacher prompts against Tom's gold labels (SPEC-he.md §0.6).

python -m eval.prompt_ab [--min-bucket 70] [--min-interval 4.5] [--versions 2,4] [--fresh-v1] [--fresh-v2]
                         [--gemini-models gemini-3.8-flash,gemini-pro-latest] [--timeout 60] [--key-env GEMINI_API_KEY_PAID]

Scores the prompt versions in --versions (default 2,3; any of 1,2,3,4) on gold items with bucket >= --min-bucket.
The v2 few-shot examples (sentiment/fewshot.json) are always dropped; v3 draws shots from buckets < 65, so it needs
--min-bucket >= 65. --min-bucket 0 tests every other gold item (~550 instead of ~190).
v1 and v2 reuse the existing teacher labels (pv 1 / pv 2) unless --fresh-v1 / --fresh-v2.
--gemini-models runs prompt v2 on each named model alone (no fallback), reported as "v2@<model>".
Labels are cached in runs/prompt_ab.json, never in the teacher file, so a rerun only calls the API for what is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from sentiment import config
from train.io import CORPUS, GOLD, TEACHER, bucket, company_from_json, item_from_json, read_jsonl
from train.meta_train import target_cls

from .metrics import macro_f1, mcnemar, paired_bootstrap

CACHE = config.ROOT / "runs" / "prompt_ab.json"


def load(min_bucket: int) -> list[dict]:
    corpus = {}
    for r in read_jsonl(CORPUS):
        corpus.setdefault((r["item"]["id"], r["item"]["ticker"]), r)
    gold = {}
    for g in read_jsonl(GOLD):
        gold[g["id"]] = g
    return [{"key": f"{gid}|{g['ticker']}", "row": corpus[(gid, g["ticker"])], "label": g}
            for gid, g in gold.items()
            if not g.get("skipped") and (gid, g.get("ticker")) in corpus and bucket(gid) >= min_bucket]


def agreement(tom: list[int], gem: list[int]) -> dict:
    y, p = np.array(tom), np.array(gem)
    if not len(y):
        return {"n": 0}
    return {"n": int(len(y)), "exact": round(float((y == p).mean()), 3),
            "direction": round(float((np.sign(y) == np.sign(p)).mean()), 3),
            "f1_dir": round(macro_f1(np.sign(y).astype(int), np.sign(p).astype(int)), 3),
            "gemini_zero_tom_dir": int(((p == 0) & (y != 0)).sum()),
            "mean_gem": round(float(p.mean()) / 2, 3), "mean_tom": round(float(y.mean()) / 2, 3)}


def report(exs: list[dict], labels: dict[str, dict]) -> dict:
    out = {}
    for name, keep in (("all", lambda e: True), ("news", lambda e: e["row"]["item"]["channel"] != "social"),
                       ("social", lambda e: e["row"]["item"]["channel"] == "social")):
        sub = [e for e in exs if keep(e) and e["key"] in labels]
        out[name] = agreement([target_cls(e["label"]) for e in sub], [target_cls(labels[e["key"]]) for e in sub])
    return out


def paired(exs: list[dict], base: dict, other: dict) -> dict:
    """other vs base on the same items: direction fixed/broken, McNemar p, macro-F1 diff with a 95% bootstrap CI."""
    y = np.sign([target_cls(e["label"]) for e in exs]).astype(int)
    a = np.sign([target_cls(base[e["key"]]) for e in exs]).astype(int)
    b = np.sign([target_cls(other[e["key"]]) for e in exs]).astype(int)
    if not len(y):
        return {"n": 0}
    ok_a, ok_b = a == y, b == y
    d = paired_bootstrap(macro_f1, (y, b), (y, a))
    return {"n": int(len(y)), "fixed": int((ok_b & ~ok_a).sum()), "broke": int((ok_a & ~ok_b).sum()),
            "mcnemar_p": round(mcnemar(ok_b, ok_a), 4),
            "f1_diff": round(d["diff"], 3), "ci": [round(d["lo"], 3), round(d["hi"], 3)]}


def untouched(exs: list[dict]) -> list[dict]:
    """Gold never re-reviewed against Gemini: re-reviews moved labels toward it, which favors any Gemini prompt."""
    return [e for e in exs if "review" not in e["label"]]


def run_labeler(labeler, exs: list[dict], have: dict, min_interval: float) -> dict:
    by_ticker = defaultdict(list)
    for e in exs:
        if e["key"] not in have:
            by_ticker[e["row"]["item"]["ticker"]].append(e)
    last = None
    for ticker, lst in by_ticker.items():
        if labeler.exhausted:
            print("quota exhausted — rerun tomorrow, cached labels are kept")
            break
        if last is not None:
            time.sleep(max(0.0, min_interval - (time.monotonic() - last)))
        last = time.monotonic()
        ans = labeler.label(company_from_json(lst[0]["row"]["company"]), [item_from_json(e["row"]["item"]) for e in lst])
        for e in lst:
            a = ans.get(e["row"]["item"]["id"])
            if a:
                have[e["key"]] = a
    return have


def parse_versions(s: str) -> list[int]:
    try:
        vs = sorted({int(x) for x in s.split(",") if x.strip()})
    except ValueError:
        raise SystemExit(f"--versions: not a comma list of ints: {s!r}")
    if not set(vs) <= {1, 2, 3, 4}:
        raise SystemExit("--versions: choose from 1,2,3,4")
    return vs


def parse_providers(s: str) -> list[str]:
    ps = [x.strip() for x in s.split(",") if x.strip()]
    bad = [p for p in ps if p not in config.LLM_PROVIDERS]
    if bad:
        raise SystemExit(f"--providers: unknown {bad}; choose from {sorted(config.LLM_PROVIDERS)}")
    return list(dict.fromkeys(ps))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.prompt_ab")
    ap.add_argument("--min-bucket", type=int, default=70)
    ap.add_argument("--min-interval", type=float, default=4.5)
    ap.add_argument("--versions", default="2,3")
    ap.add_argument("--fresh-v1", action="store_true")
    ap.add_argument("--fresh-v2", action="store_true")
    ap.add_argument("--providers", default="", help="comma list of " + ",".join(config.LLM_PROVIDERS))
    ap.add_argument("--gemini-models", default="", help="comma list of Gemini model ids, each run alone with prompt v2")
    ap.add_argument("--timeout", type=float, default=60, help="seconds per Gemini request (--gemini-models only)")
    ap.add_argument("--key-env", default="GEMINI_API_KEY", help="env var holding the key for --gemini-models")
    a = ap.parse_args(argv)
    versions = parse_versions(a.versions)
    providers = parse_providers(a.providers)
    models = list(dict.fromkeys(m.strip() for m in a.gemini_models.split(",") if m.strip()))
    if 3 in versions and a.min_bucket < 65:
        raise SystemExit("--versions 3 draws shots from buckets < 65: use --min-bucket 65 or more")
    from sentiment.llm import FEWSHOT, GeminiLabeler, OpenAICompatLabeler
    exs = load(a.min_bucket)
    if FEWSHOT.exists():
        shot_ids = {e["id"] for e in json.loads(FEWSHOT.read_text(encoding="utf-8"))}
        kept = [e for e in exs if e["label"]["id"] not in shot_ids]
        print(f"gold items: {len(kept)} (dropped {len(exs) - len(kept)} few-shot examples)")
        exs = kept
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for k in ("v1", "v2", "v3", "v4"):
        cache.setdefault(k, {})
    reuse = [pv for pv, fresh in ((1, a.fresh_v1), (2, a.fresh_v2)) if pv in versions and not fresh]
    if reuse:
        teacher: dict[int, dict] = {pv: {} for pv in reuse}
        for t in read_jsonl(TEACHER):
            if t.get("pv", 1) in teacher:
                teacher[t.get("pv", 1)].setdefault(f"{t['id']}|{t.get('ticker')}", t)
        for pv in reuse:
            for e in exs:
                if e["key"] in teacher[pv]:
                    cache[f"v{pv}"].setdefault(e["key"], teacher[pv][e["key"]])
    CACHE.parent.mkdir(exist_ok=True)
    for pv in versions:
        lab = GeminiLabeler(prompt_version=pv)
        if not lab.available:
            print("GEMINI_API_KEY missing")
            return 2
        cache[f"v{pv}"] = run_labeler(lab, exs, cache[f"v{pv}"], a.min_interval)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    for p in providers:
        lab = OpenAICompatLabeler(p, prompt_version=2)
        if not lab.available:
            print(f"{config.LLM_PROVIDERS[p]['key_env']} missing")
            continue
        cache[p] = run_labeler(lab, exs, cache.setdefault(p, {}), a.min_interval)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    for m in models:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        key = os.environ.get(a.key_env)
        if not key:  # never fall back to the free GEMINI_API_KEY behind the caller's back
            print(f"{a.key_env} missing")
            return 2
        lab = GeminiLabeler(key, prompt_version=2, models=(m,), timeout=a.timeout)
        cache[f"v2@{m}"] = run_labeler(lab, exs, cache.setdefault(f"v2@{m}", {}), a.min_interval)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    names = ([f"v{pv}" for pv in versions] + [p for p in providers if cache.get(p)]
             + [f"v2@{m}" for m in models if cache.get(f"v2@{m}")])
    if not names:
        print("nothing to report")
        return 0
    both = set.intersection(*(set(cache.get(n, {})) for n in names))
    print(f"n_common={len(both)}")
    common = [e for e in exs if e["key"] in both]
    res = {n: report(common, cache[n]) for n in names}
    res |= {f"{n} untouched": report(untouched(common), cache[n]) for n in names}
    res |= {f"{n} vs {names[0]}": {"all": paired(common, cache[names[0]], cache[n])} for n in names[1:]}
    (config.ROOT / "runs" / "prompt_ab_report.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for n, rows in res.items():
        for name, r in rows.items():
            print(f"{n} {name:7}", "  ".join(f"{k}={v}" for k, v in r.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
