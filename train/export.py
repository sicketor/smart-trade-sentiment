"""Export the student to ONNX fp32 + dynamic int8 and check parity against torch."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

from sentiment.config import ROOT

SYNTHETIC = [
    "Nvidia beats earnings estimates and raises full-year guidance",
    "Tesla recalls 200,000 vehicles over safety defect",
    "The company will hold its annual meeting on Tuesday",
    "Apple shares fall after weak iPhone demand in China",
    "Analysts upgrade Microsoft to buy on cloud growth",
    "SEC opens investigation into the firm's accounting practices",
    "Amazon announces a new distribution center in Ohio",
    "Bankruptcy filing wipes out shareholders of retailer",
]


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.glob("*.onnx")) / 2**20


def sample_inputs(model_dir: Path, n: int = 200) -> list[tuple[str, str]]:
    val = ROOT / "data" / "train" / "val.jsonl"
    pairs = []
    if val.exists():
        import json
        with open(val, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pairs.append((r["target"], r["text"]))
                if len(pairs) >= n:
                    break
    if not pairs:
        pairs = [("", SYNTHETIC[i % len(SYNTHETIC)] + ("" if i < len(SYNTHETIC) else f" ({i})")) for i in range(n)]
    return pairs


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def run_onnx(model_dir: Path, tok, pairs, bs=16, file_name=None) -> np.ndarray:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    m = ORTModelForSequenceClassification.from_pretrained(model_dir, file_name=file_name)
    return run(m, tok, pairs, bs, torch_out=False)


def run(model, tok, pairs, bs=16, torch_out=True) -> np.ndarray:
    import torch
    outs = []
    for s in range(0, len(pairs), bs):
        chunk = pairs[s:s + bs]
        for t, x in chunk:  # per-row to mirror training tokenization
            enc = tok(t, x, truncation="only_second", max_length=256, return_tensors="pt") if t \
                else tok(x, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                lg = model(**enc).logits
            outs.append(softmax(lg.float().cpu().numpy())[0])
    return np.array(outs)


def export(model: str, out: str, int8: bool = False) -> int:
    """fp32 ONNX is the shipped artifact (parity-gated). Dynamic int8 is optional and only
    reported: on DeBERTa-v3 it agreed with PyTorch on 57% (per-tensor) / 82% (per-channel
    MatMul) of argmaxes, so it is not used at runtime."""
    import shutil

    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src, dst = _abs(model), _abs(out)
    dst.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(src)
    ort = ORTModelForSequenceClassification.from_pretrained(src, export=True)
    ort.save_pretrained(dst)
    tok.save_pretrained(dst)

    pairs = sample_inputs(src)
    pt = AutoModelForSequenceClassification.from_pretrained(src).float().eval()
    p_torch = run(pt, tok, pairs)
    max_diff = float(np.abs(p_torch - run_onnx(dst, tok, pairs)).max())
    fp32_mb = sum(f.stat().st_size for f in dst.glob("*.onnx")) / 2**20
    print(f"rows={len(pairs)} onnx_fp32_max_abs_diff={max_diff:.2e} size={fp32_mb:.0f} MB")

    if int8:
        q = ORTQuantizer.from_pretrained(dst)
        cfg = AutoQuantizationConfig.avx2(is_static=False, per_channel=True, operators_to_quantize=["MatMul"])
        q.quantize(save_dir=dst / "int8", quantization_config=cfg)
        tok.save_pretrained(dst / "int8")
        if (dst / "config.json").exists():
            shutil.copy(dst / "config.json", dst / "int8" / "config.json")
        p_int8 = run_onnx(dst / "int8", tok, pairs, file_name="model_quantized.onnx")
        agree = float((p_torch.argmax(1) == p_int8.argmax(1)).mean())
        int8_mb = sum(f.stat().st_size for f in (dst / "int8").glob("*.onnx")) / 2**20
        print(f"int8 (not used at runtime): argmax_agreement={agree:.3f} size={int8_mb:.0f} MB")

    ok = max_diff < 1e-3
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/st-finsent")
    ap.add_argument("--out", default="models/st-finsent-onnx")
    ap.add_argument("--int8", action="store_true", help="also try dynamic int8 (report only)")
    a = ap.parse_args()
    sys.exit(export(a.model, a.out, a.int8))


if __name__ == "__main__":
    main()
