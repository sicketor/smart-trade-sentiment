"""Stage-2 fine-tune of the student on Tom's gold labels, with a 5-fold out-of-fold evaluation.

python -m train.student_gold oof     # judge: base vs stage-2 on gold never seen in training
python -m train.student_gold final   # fine-tune on all gold (+ teacher replay) and save

Gold rows are mixed with a random replay of teacher rows from data/train/train.jsonl so the model
does not forget the teacher distribution. Fixed epochs, no validation split, no best-epoch pick.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch

from eval.gold import _report
from sentiment.config import ROOT
from sentiment.meta import group_of
from sentiment.types import CLASS_VALUES
from train import io as tio
from train import student
from train.build_dataset import corpus_target, item_text, smooth5
from train.meta_train import target_cls

TRAIN_JSONL = ROOT / "data" / "train" / "train.jsonl"


def gold_rows() -> list[dict]:
    """One student-format row per usable gold label (latest per id wins, skips dropped), sorted by id."""
    corpus, by_id = {}, {}
    for r in tio.read_jsonl(tio.CORPUS):
        corpus.setdefault((r["item"]["id"], r["item"].get("ticker")), r)
        by_id.setdefault(r["item"]["id"], r)
    gold: dict[str, dict] = {}
    for g in tio.read_jsonl(tio.GOLD):
        if "id" in g:
            gold[g["id"]] = g
    rows = []
    for gid in sorted(gold):
        g = gold[gid]
        if g.get("skipped"):
            continue
        r = corpus.get((gid, g.get("ticker"))) or by_id.get(gid)
        if r is None:
            continue
        item = tio.item_from_json(r["item"])
        ticker = g.get("ticker") or item.ticker
        company = tio.company_from_json(r["company"])
        cls = target_cls(g)
        rows.append({"id": gid, "ticker": ticker, "channel": item.channel,
                     "target": corpus_target(ticker, company), "text": item_text(item),
                     "soft": smooth5(cls), "weight": 1.0, "cls": cls})
    return rows


def split_folds(rows: list[dict], folds: int = 5) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {k: [] for k in range(folds)}
    for i, r in enumerate(rows):
        out[tio.bucket(r["id"]) % folds].append(i)
    return out


def replay_rows(path, n: int, seed: int) -> list[dict]:
    pool = [r for r in tio.read_jsonl(path) if r.get("source") == "teacher"]
    if len(pool) <= n:
        return pool
    return random.Random(seed).sample(pool, n)


def finetune(init_dir, rows: list[dict], *, epochs: int, lr: float, batch: int, accum: int,
             max_len: int, seed: int, device, out=None):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device)
    tok = AutoTokenizer.from_pretrained(str(init_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(init_dir)).to(device)
    enc = student.encode(tok, rows, max_len)

    steps_per_epoch = (len(rows) + batch - 1) // batch
    total_updates = max(1, epochs * ((steps_per_epoch + accum - 1) // accum))
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_updates), total_updates)

    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(rows)).tolist()
        run_loss, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b in range(steps_per_epoch):
            idx = order[b * batch:(b + 1) * batch]
            bt, soft, w = student.collate(tok, enc, rows, idx, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**bt).logits
            loss = student.soft_loss(logits, soft, w)
            (loss / accum).backward()
            run_loss += loss.item(); n += 1
            if (b + 1) % accum == 0 or b == steps_per_epoch - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        print(f"  epoch {epoch}: train loss {run_loss / max(n, 1):.4f}", flush=True)

    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out); tok.save_pretrained(out)
    return model


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _device(args):
    return torch.device(getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))


def _ft_kwargs(args, device) -> dict:
    return dict(epochs=args.epochs, lr=args.lr, batch=args.batch, accum=args.accum,
                max_len=args.max_len, seed=args.seed, device=device)


def oof(rows: list[dict], args) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _device(args)
    init = _abs(args.init)
    tok = AutoTokenizer.from_pretrained(str(init))
    enc = student.encode(tok, rows, args.max_len)

    base_model = AutoModelForSequenceClassification.from_pretrained(str(init)).to(device)
    base = student.predict(base_model, tok, enc, rows, device)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    s2 = np.zeros((len(rows), 5), dtype=np.float32)
    folds = split_folds(rows, args.folds)
    for k, idx in folds.items():
        if not idx:
            continue
        held = set(idx)
        train_rows = [r for i, r in enumerate(rows) if i not in held]
        train_rows += replay_rows(TRAIN_JSONL, args.replay, args.seed)
        print(f"fold {k}: train {len(train_rows)} rows, eval {len(idx)} gold", flush=True)
        model = finetune(init, train_rows, **_ft_kwargs(args, device))
        sub = [rows[i] for i in idx]
        s2[idx] = student.predict(model, tok, [enc[i] for i in idx], sub, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"base": base, "s2": s2}


def report(rows: list[dict], probs: np.ndarray) -> dict:
    probs = np.asarray(probs, dtype=float)
    y = np.array([r["cls"] for r in rows])
    groups = {"all": np.ones(len(rows), dtype=bool)}
    ch = np.array([group_of(r["channel"]) for r in rows])
    for g in ("text", "social"):
        groups[g] = ch == g
    out = {}
    for name, m in groups.items():
        if not m.any():
            continue
        p = probs[m]
        cls5 = p.argmax(1) - 2
        score5 = p @ np.array(CLASS_VALUES)
        p3 = np.stack([p[:, 0] + p[:, 1], p[:, 2], p[:, 3] + p[:, 4]], axis=1)
        cls3 = p3.argmax(1) - 1
        score3 = p3[:, 2] - p3[:, 0]
        out[name] = {"5class": _report(y[m], cls5, score5), "3class": _report(y[m], cls3, score3)}
    return out


def _print_table(reports: dict) -> None:
    for kind in ("3class", "5class"):
        print(f"[{kind}]")
        print(f"{'group':8}{'model':6}{'n':>5}{'f1_dir':>8}{'acc5':>8}{'mae5':>8}")
        for g in ("all", "text", "social"):
            for m in ("base", "s2"):
                r = reports[m].get(g)
                if r:
                    d = r[kind]
                    print(f"{g:8}{m:6}{d['n']:>5}{d['f1_dir']:>8}{d['acc5']:>8}{d['mae5']:>8}")


def cmd_oof(rows, args) -> None:
    res = oof(rows, args)
    rep = {"base": report(rows, res["base"]), "s2": report(rows, res["s2"])}
    out = ROOT / "runs" / "student_gold_oof.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(args), "ids": [r["id"] for r in rows], "tickers": [r["ticker"] for r in rows],
        "base": res["base"].tolist(), "s2": res["s2"].tolist(), "report": rep}, indent=2), encoding="utf-8")
    _print_table(rep)
    print(f"wrote {out}")


def cmd_final(rows, args) -> None:
    out = _abs(args.out)
    replay = replay_rows(TRAIN_JSONL, args.replay, args.seed)
    device = _device(args)
    print(f"final: {len(rows)} gold + {len(replay)} replay -> {out}", flush=True)
    model = finetune(_abs(args.init), rows + replay, out=out, **_ft_kwargs(args, device))
    del model
    (out / "gold_ids.json").write_text("[]", encoding="utf-8")   # saw every gold id: crossfit has no unseen ones
    (out / "stage2.json").write_text(json.dumps({
        "init": str(args.init), "n_gold": len(rows), "n_replay": len(replay), "args": vars(args)},
        indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("oof", "final"))
    ap.add_argument("--init", default="models/st-finsent")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--replay", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="models/st-finsent-s2")
    args = ap.parse_args()
    rows = gold_rows()
    if not rows:
        raise SystemExit("no gold rows")
    (cmd_oof if args.mode == "oof" else cmd_final)(rows, args)


if __name__ == "__main__":
    main()
