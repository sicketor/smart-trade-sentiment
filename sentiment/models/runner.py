"""Runs the registry models and maps every output to (neg, neu, pos). See SPEC-he.md §5."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import sqlite3

from .. import config
from ..types import Company, ModelOutput, RawItem
from .registry import SPECS, STUDENT_DIR, ModelSpec, specs_for, student_paths

log = logging.getLogger(__name__)
CACHE_TEMPLATE = "v2"   # v2: NLI hypotheses "The stock of {n} will go up/down."
Probs = tuple[float, float, float]

_HANDLE = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+|www\.\S+")


def _social_fallback(text: str) -> str:
    return _URL.sub("http", _HANDLE.sub("@user", text))


def student_backend(device: str, env: str | None, has_torch: bool, has_onnx: bool) -> str:
    """"onnx" | "torch" for the student. env = ST_BACKEND: "onnx", "torch", or unset/"auto"
    (onnx on cpu when the export exists, torch otherwise). Without torch weights onnx is forced."""
    mode = (env or "auto").strip().lower()
    if mode not in ("auto", "onnx", "torch"):
        raise ValueError(f"ST_BACKEND must be auto, onnx or torch, got {env!r}")
    if not has_torch:
        if not has_onnx:
            raise RuntimeError("model st_finsent unavailable: no torch weights and no model.onnx")
        return "onnx"
    if mode == "onnx":
        if not has_onnx:
            raise RuntimeError("ST_BACKEND=onnx but model.onnx is missing")
        return "onnx"
    if mode == "torch":
        return "torch"
    return "onnx" if device == "cpu" and has_onnx else "torch"


def _for_social(text: str) -> str:
    try:
        from ..text import for_social_model
    except ImportError:
        return _social_fallback(text)
    return for_social_model(text)


class ModelRunner:
    def __init__(self, device: str = "auto", cache: bool = True):
        import torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.cache = cache
        self._models: dict[str, tuple] = {}
        self._failed: set[str] = set()
        self._db: sqlite3.Connection | None = None

    # ---- loading / forward -------------------------------------------------
    def _load_student_onnx(self, spec: ModelSpec, onnx_dir) -> tuple:
        from transformers import AutoTokenizer
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification  # lazy: torch-only setups
            tok = AutoTokenizer.from_pretrained(onnx_dir)
            model = ORTModelForSequenceClassification.from_pretrained(onnx_dir)
        except Exception as e:
            raise RuntimeError(f"model {spec.name} unavailable (onnx)") from e
        return tok, model, "onnx"

    def _load(self, spec: ModelSpec):
        if spec.name not in self._models and spec.cache_id == STUDENT_DIR:
            tdir, odir = student_paths()
            # auto falls back to torch when onnxruntime is not installed (requirements.txt has no onnx extras)
            has_onnx = odir is not None and (tdir is None or os.environ.get("ST_BACKEND") == "onnx"
                                             or importlib.util.find_spec("onnxruntime") is not None)
            backend = student_backend(self.device, os.environ.get("ST_BACKEND"), tdir is not None, has_onnx)
            log.info("st_finsent backend: %s (%s)", backend, odir if backend == "onnx" else tdir)
            if backend == "onnx":  # torch: spec.hf_id already is the torch dir (registry prefers it)
                self._models[spec.name] = self._load_student_onnx(spec, odir)
        if spec.name not in self._models:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            try:
                tok = AutoTokenizer.from_pretrained(spec.hf_id)
                model = AutoModelForSequenceClassification.from_pretrained(spec.hf_id)
            except Exception as e:  # network, missing repo, corrupt cache
                raise RuntimeError(f"model {spec.name} unavailable") from e
            model.eval().to(self.device)
            if self.device.startswith("cuda"):
                model.half()
            self._models[spec.name] = (tok, model, "torch")
        return self._models[spec.name]

    def _run(self, spec: ModelSpec, first: list[str], second: list[str] | None) -> list[list[float]]:
        """Softmax class probabilities for one batch (second = pair input or None)."""
        import torch
        tok, model, backend = self._load(spec)
        args = (first, second) if second is not None else (first,)
        enc = tok(*args, truncation=True, max_length=spec.max_len, padding=True, return_tensors="pt")
        if backend == "torch":  # onnxruntime runs on its own provider (cpu); no .to()/half()
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits.float()
        return torch.softmax(logits, dim=-1).cpu().tolist()

    # ---- cache -------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(config.CACHE_DIR / "model_cache.sqlite")
            self._db.execute("CREATE TABLE IF NOT EXISTS cache "
                             "(model TEXT, key TEXT, probs TEXT, PRIMARY KEY (model, key))")
        return self._db

    @staticmethod
    def _key(text: str, aspect: str, spec=None) -> str:
        # checkpoint (cache_id: path-independent for local models), length and prompt template are part of the key: changing any of them
        # must not serve old probabilities. Bump CACHE_TEMPLATE when NLI hypotheses change.
        head = f"{spec.cache_id or spec.hf_id}\x1f{spec.version}\x1f{spec.max_len}\x1f{CACHE_TEMPLATE}\x1f" if spec else ""
        return hashlib.sha1((head + text + "\x1f" + aspect).encode("utf-8")).hexdigest()

    # ---- prediction --------------------------------------------------------
    @staticmethod
    def _map(spec: ModelSpec, row: list[float]) -> Probs:
        out = {"neg": 0.0, "neu": 0.0, "pos": 0.0}
        for lab, p in zip(spec.labels, row):
            out[lab] += p
        return out["neg"], out["neu"], out["pos"]

    def _compute(self, spec: ModelSpec, texts: list[str], aspects: list[str] | None,
                 batch_size: int) -> list[Probs]:
        res: list[Probs] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            asp = aspects[i:i + batch_size] if aspects else None
            if spec.kind == "nli":
                names = asp or [""] * len(chunk)
                # "good/bad news for X" gave entailment ~0.06 even on clear beats; direction works.
                pos_h = [f"The stock of {n} will go up." for n in names]
                neg_h = [f"The stock of {n} will go down." for n in names]
                e_pos = [r[0] for r in self._run(spec, chunk, pos_h)]
                e_neg = [r[0] for r in self._run(spec, chunk, neg_h)]
                for ep, en in zip(e_pos, e_neg):
                    raw = (en, max(0.0, 1 - max(ep, en)), ep)
                    s = sum(raw) or 1.0
                    res.append((raw[0] / s, raw[1] / s, raw[2] / s))
                continue
            if spec.input == "target":
                # trained as tokenizer(target, text), or text alone when there is no target
                tgt = asp or [""] * len(chunk)
                rows: list = [None] * len(chunk)
                with_t = [j for j, t in enumerate(tgt) if t]
                without = [j for j, t in enumerate(tgt) if not t]
                if with_t:
                    for j, r in zip(with_t, self._run(spec, [tgt[j] for j in with_t], [chunk[j] for j in with_t])):
                        rows[j] = r
                if without:
                    for j, r in zip(without, self._run(spec, [chunk[j] for j in without], None)):
                        rows[j] = r
                res.extend(self._map(spec, r) for r in rows)
                continue
            if spec.input == "social":
                chunk = [_for_social(t) for t in chunk]
            rows = self._run(spec, chunk, asp if spec.input == "pair" else None)
            res.extend(self._map(spec, r) for r in rows)
        return res

    def predict(self, spec_name: str, texts: list[str], aspects: list[str] | None = None,
                batch_size: int = 32) -> list[Probs]:
        spec = SPECS[spec_name]
        if not texts:
            return []
        asp = aspects if aspects is not None else [""] * len(texts)
        keys = [self._key(t, a, spec) for t, a in zip(texts, asp)]
        out: list[Probs | None] = [None] * len(texts)
        if self.cache:
            db = self._conn()
            for i, k in enumerate(keys):
                r = db.execute("SELECT probs FROM cache WHERE model=? AND key=?", (spec.name, k)).fetchone()
                if r:
                    out[i] = tuple(json.loads(r[0]))
        todo = [i for i, o in enumerate(out) if o is None]
        if todo:
            res = self._compute(spec, [texts[i] for i in todo], [asp[i] for i in todo], batch_size)
            for i, p in zip(todo, res):
                out[i] = p
                if self.cache:
                    self._conn().execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)",
                                         (spec.name, keys[i], json.dumps(p)))
            if self.cache:
                self._conn().commit()
        return out  # type: ignore[return-value]

    def score_items(self, items: list[RawItem], company: Company, channel: str) -> dict[str, list[ModelOutput]]:
        result: dict[str, list[ModelOutput]] = {it.id: [] for it in items}
        if not items:
            return result
        texts = []
        for it in items:
            if channel == "social":
                texts.append(it.text or it.title)
            else:
                texts.append(f"{it.title}. {it.text[:600]}" if it.text else it.title)
        aspects = [company.short_name] * len(items)
        for spec in specs_for(channel):
            if spec.name in self._failed:
                continue
            asp = [f"{company.ticker} | {company.short_name}"] * len(items) if spec.input == "target" else aspects
            try:
                probs = self.predict(spec.name, texts, asp)
            except RuntimeError as e:
                log.warning("%s", e)
                self._failed.add(spec.name)
                continue
            for it, p in zip(items, probs):
                result[it.id].append(ModelOutput(spec.name, tuple(round(x, 4) for x in p)))
        return result
