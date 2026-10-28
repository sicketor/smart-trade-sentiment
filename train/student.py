"""Train ST-FinSent (deberta-v3-base) on soft labels (SPEC-he.md §5.0)."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn.functional as F

from sentiment.config import ROOT

BASE = "microsoft/deberta-v3-base"
ID2LABEL = {0: "-2", 1: "-1", 2: "0", 3: "1", 4: "2"}


def soft_loss(logits: torch.Tensor, soft: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """sum(-(soft . log_softmax(logits)) * w) / sum(w)."""
    ce = -(soft * F.log_softmax(logits.float(), dim=-1)).sum(-1)
    return (ce * weight).sum() / weight.sum()


def to3(c5: np.ndarray) -> np.ndarray:
    """5 classes (0..4) -> 3 classes (0 neg, 1 neu, 2 pos)."""
    return np.where(c5 <= 1, 0, np.where(c5 == 2, 1, 2))


def macro_f1_3(pred5: np.ndarray, truth5: np.ndarray) -> float:
    p, t = to3(pred5), to3(truth5)
    fs = []
    for c in range(3):
        tp = float(((p == c) & (t == c)).sum())
        fp = float(((p == c) & (t != c)).sum())
        fn = float(((p != c) & (t == c)).sum())
        fs.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(fs))


def mae5(pred5: np.ndarray, truth5: np.ndarray) -> float:
    return float(np.abs(pred5 - truth5).mean())


def ece(probs: np.ndarray, truth5: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == truth5).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(tok, rows: list[dict], max_len: int):
    """Tokenize target/text pairs (text alone when target is empty). Returns list of dicts."""
    out = []
    for r in rows:
        if r["target"]:
            enc = tok(r["target"], r["text"], truncation="only_second", max_length=max_len)
        else:
            enc = tok(r["text"], truncation=True, max_length=max_len)
        out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return out


def collate(tok, enc: list[dict], rows: list[dict], idx: list[int], device):
    batch = tok.pad([enc[i] for i in idx], return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    soft = torch.tensor([rows[i]["soft"] for i in idx], dtype=torch.float32, device=device)
    w = torch.tensor([rows[i]["weight"] for i in idx], dtype=torch.float32, device=device)
    return batch, soft, w


@torch.no_grad()
def predict(model, tok, enc, rows, device, bs=64, amp=True):
    model.eval()
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]))
    probs = np.zeros((len(enc), 5), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch, _, _ = collate(tok, enc, rows, idx, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = model(**batch).logits
        probs[idx] = F.softmax(logits.float(), -1).cpu().numpy()
    return probs


def val_metrics(probs: np.ndarray, rows: list[dict]) -> dict:
    truth = np.array([int(np.argmax(r["soft"])) for r in rows])
    pred = probs.argmax(1)
    return {"macro_f1_3": macro_f1_3(pred, truth), "mae5": mae5(pred, truth), "ece": ece(probs, truth)}


def eval_public(model, tok, device, max_len, name, limit=2000):
    from eval import datasets as ds
    exs = ds.load(name, "test", limit=limit)
    rows = [{"target": f"{e.target} | {e.target}" if e.target else "", "text": e.text, "soft": [0] * 5, "weight": 1}
            for e in exs]
    probs = predict(model, tok, encode(tok, rows, max_len), rows, device)
    pred3 = to3(probs.argmax(1))
    truth3 = np.array([e.label3 + 1 for e in exs])
    fs = []
    for c in range(3):
        tp = float(((pred3 == c) & (truth3 == c)).sum())
        d = 2 * tp + float(((pred3 == c) & (truth3 != c)).sum()) + float(((pred3 != c) & (truth3 == c)).sum())
        fs.append(2 * tp / d if d else 0.0)
    return float(np.mean(fs))


def train(args) -> dict:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    tr, va = read_rows(data / "train.jsonl"), read_rows(data / "val.jsonl")
    if args.limit:
        tr, va = tr[:args.limit], va[:max(1, args.limit // 5)]
    if not tr or not va:
        raise SystemExit(f"empty split (train {len(tr)}, val {len(va)}): rebuild with train.build_dataset")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=5, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}).to(device)
    tr_enc, va_enc = encode(tok, tr, args.max_len), encode(tok, va, args.max_len)

    steps_per_epoch = (len(tr) + args.batch - 1) // args.batch
    total_updates = max(1, args.epochs * ((steps_per_epoch + args.accum - 1) // args.accum))  # accum resets per epoch
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_updates), total_updates)

    rng = np.random.default_rng(args.seed)
    history, best, step, t0 = [], -1.0, 0, time.time()
    step_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(tr)).tolist()
        run_loss, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b in range(steps_per_epoch):
            ts = time.time()
            idx = order[b * args.batch:(b + 1) * args.batch]
            batch, soft, w = collate(tok, tr_enc, tr, idx, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            loss = soft_loss(logits, soft, w)
            (loss / args.accum).backward()
            run_loss += loss.item(); n += 1; step += 1
            if (b + 1) % args.accum == 0 or b == steps_per_epoch - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.time() - ts)
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {run_loss / n:.4f}", flush=True)
        m = val_metrics(predict(model, tok, va_enc, va, device), va)
        m.update(epoch=epoch, train_loss=run_loss / max(n, 1))
        history.append(m)
        print(f"epoch {epoch}: {m}", flush=True)
        if m["macro_f1_3"] > best:
            best = m["macro_f1_3"]
            model.save_pretrained(out); tok.save_pretrained(out)

    sec_per_100 = float(np.median(step_times) * 100) if step_times else 0.0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    metrics = {"epochs": history, "best_macro_f1_3": best, "args": vars(args),
               "data": {"n_train": len(tr), "n_val": len(va)},
               "train_seconds": time.time() - t0, "sec_per_100_steps": sec_per_100, "peak_gpu_gb": peak}

    try:
        from eval import datasets as _ds  # noqa: F401
        best_model = AutoModelForSequenceClassification.from_pretrained(out).to(device)
        for name in ("sentfin", "tfns"):
            try:
                metrics[f"test_macro_f1_{name}"] = eval_public(best_model, tok, device, args.max_len, name)
            except Exception as e:  # dataset missing offline etc.
                metrics[f"test_macro_f1_{name}"] = f"error: {e}"
    except ImportError:
        pass
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if (data / "gold_ids.json").exists():   # gold ids the student never saw: eval.crossfit reads this
        shutil.copyfile(data / "gold_ids.json", out / "gold_ids.json")
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"}, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="models/st-finsent")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
