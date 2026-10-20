"""Load the 83M persona model (ONNX Runtime, CPU) and time an answer with or without the KV/SSM cache."""

from __future__ import annotations

import json
import os
import resource
import shutil
import threading
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PERSONA_RULE = ("You are the person this resume describes. Answer as yourself, in first person, using only the "
                "resume. If the resume does not say, say it is not in your resume.")

_STATE: dict = {}
_LOCK = threading.Lock()  # one answer at a time: the caches live on the model, and two would double peak memory
CACHE = ["K", "V", "theta", "ssm", "sk", "sv", "copy_k", "ids_full"]
CHUNK = 256  # prompt tokens per graph call: memory grows with CHUNK x 1024, not prompt length squared


class OrtModel:
    """onnx/chunk.onnx reads C new tokens on top of the caches (attention K/V, Mamba-3 state, copy-head
    keys) and returns the next-token distribution. The prompt is read in chunks; decoding is C = 1."""
    max_len = 1024

    def __init__(self, sess, weights):
        self.sess, self.w = sess, weights

    def _zero(self):
        f = np.float32
        return {"K": np.zeros((24, 2, 1024, 64), f), "V": np.zeros((24, 2, 1024, 64), f),
                "theta": np.zeros((24, 12, 16), f), "ssm": np.zeros((24, 12, 64, 64), f),
                "sk": np.zeros((24, 12, 64), f), "sv": np.zeros((24, 12, 64), f),
                "copy_k": np.zeros((1024, 128), f), "ids_full": np.zeros(1024, np.int64)}

    def _run(self, ids):
        out = self.sess.run(None, {"ids": np.array([ids], np.int64), "pos": np.array(self.pos, np.int64),
                                   **self.cache, **self.w})
        self.cache, self.pos = dict(zip(CACHE, out[1:])), self.pos + len(ids)
        return out[0]

    def prefill(self, ids):
        self.cache, self.pos = self._zero(), 0
        for i in range(0, len(ids), CHUNK):
            p = self._run(ids[i : i + CHUNK])
        return p

    def step(self, tok):
        return self._run([tok])


def _weights(sess) -> dict:
    """fp16 on disk -> fp32 graph inputs: 2-D weights transposed to (in, out) as "p.T__<name>", the
    embedding also as is for the lookup (same layout as export_onnx.weight_inputs)."""
    w = {}
    for f in sorted((HERE / "onnx").glob("w*.npz")):
        with np.load(f) as z:
            for k in z.files:
                v = z[k].astype(np.float32)
                if v.ndim == 2:
                    w["p.T__" + k.replace(".", "__")] = np.ascontiguousarray(v.T)
                if v.ndim != 2 or k == "tok_emb.weight":
                    w["p." + k.replace(".", "__")] = v
    want = {i.name for i in sess.get_inputs()}
    return {k: v for k, v in w.items() if k in want}


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load():
    if _STATE:
        return _STATE
    t0 = time.time()
    import tokenmonster
    tm_dir = Path("/tmp/tokenmonster")
    tm_dir.mkdir(exist_ok=True)
    if not (tm_dir / "tokenmonsterserver").exists():  # bundled, so no download at cold start
        shutil.copy(HERE / "tokenmonsterserver", tm_dir / "tokenmonsterserver")
        (tm_dir / "tokenmonsterserver").chmod(0o755)
    tokenmonster.set_local_directory(str(tm_dir))
    import onnxruntime as ort
    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # graph saved pre-optimized
    opt.enable_mem_reuse = False  # its planner takes ~13 s on this 13k-node graph; peak memory is the same
    sess = ort.InferenceSession(str(HERE / "onnx/chunk.onnx"), opt, providers=["CPUExecutionProvider"])
    model = OrtModel(sess, _weights(sess))
    vocab = tokenmonster.load(str(HERE / "tokenizer.vocab"))
    _STATE.update(model=model, vocab=vocab, resume=(HERE / "resume.txt").read_text().strip(),
                  stop=(vocab.token_to_id("<eos>"), vocab.token_to_id("<sep>")), load_s=time.time() - t0)
    return _STATE


def answer(question: str, cached: bool = True, max_new: int = 32, budget_s: float = 240.0) -> dict:
    with _LOCK:
        return _answer(question, cached, max_new, budget_s)


def _answer(question: str, cached: bool, max_new: int, budget_s: float) -> dict:
    s = load()
    model, vocab, stop = s["model"], s["vocab"], s["stop"]
    prompt = [int(t) for t in vocab.tokenize(
        f"<sys>{PERSONA_RULE}<sep><user>Resume:\n{s['resume']}\n\nQuestion: {question}<sep><assistant>")]
    max_new = min(max_new, model.max_len - len(prompt) - 1)
    toks, step_s, t_start = [], [], time.time()
    p = model.prefill(prompt)
    first_s = time.time() - t_start
    stopped = "max_new"
    for _ in range(max_new):
        nxt = int(p.argmax())
        if nxt in stop:
            stopped = "eos"
            break
        toks.append(nxt)
        if time.time() - t_start > budget_s:
            stopped = "time budget"
            break
        t0 = time.time()
        if cached:  # one new position against the stored keys/values and SSM state
            p = model.step(nxt)
        else:  # no cache: reread the whole prompt plus everything generated so far
            p = model.prefill(prompt + toks)
        step_s.append(time.time() - t0)
    total = time.time() - t_start
    return {
        "mode": "cached" if cached else "uncached",
        "answer": vocab.decode(toks).strip(),
        "prompt_tokens": len(prompt),
        "new_tokens": len(toks),
        "stopped_by": stopped,
        "first_token_s": round(first_s, 3),
        "per_token_s_avg": round(sum(step_s) / len(step_s), 3) if step_s else None,
        "per_token_s_last": round(step_s[-1], 3) if step_s else None,
        "total_s": round(total, 2),
        "tokens_per_s": round(len(toks) / total, 2) if total else None,
        "model_load_s": round(s["load_s"], 2),
        "cpus_visible": os.cpu_count(),
        "peak_rss_mb": round(_rss_mb()),
    }
