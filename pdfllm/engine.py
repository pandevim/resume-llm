"""Load the 83M persona model (ONNX Runtime, CPU), time an answer with or without the KV/SSM cache, or
stream an answer token by token with the PDF words its copy head read."""

from __future__ import annotations

import json
import os
import queue
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
    keys) and returns the next-token distribution. The prompt is read in chunks; decoding is C = 1.
    It also returns the copy head of the last position: its attention over all positions so far
    (copy_attn) and the gate (copy_gate), with next = gate * vocab + (1 - gate) * copy."""
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
        self.cache, self.pos = dict(zip(CACHE, out[1 : 1 + len(CACHE)])), self.pos + len(ids)
        self.copy_attn, self.copy_gate = out[-2], float(np.ravel(out[-1])[0])
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
    resume = (HERE / "resume.txt").read_text().strip()
    word_of = np.full(len(resume), -1)  # resume.txt char -> PDF word (tools/word_map.py), -1 if none
    for i, (*_, lo, hi) in enumerate(json.loads((HERE / "resume_words.json").read_text())["words"]):
        word_of[lo:hi] = i
    _STATE.update(model=model, vocab=vocab, resume=resume, word_of=word_of,
                  stop=(vocab.token_to_id("<eos>"), vocab.token_to_id("<sep>")), load_s=time.time() - t0)
    return _STATE


def answer(question: str, cached: bool = True, max_new: int = 32, budget_s: float = 240.0) -> dict:
    with _LOCK:
        return _answer(question, cached, max_new, budget_s)


def _prompt(s: dict, question: str) -> tuple[list[int], int]:
    """Prompt token ids, and the character offset where the resume starts in the prompt text."""
    head = f"<sys>{PERSONA_RULE}<sep><user>Resume:\n"
    text = f"{head}{s['resume']}\n\nQuestion: {question}<sep><assistant>"
    return [int(t) for t in s["vocab"].tokenize(text)], len(head)


def _answer(question: str, cached: bool, max_new: int, budget_s: float) -> dict:
    s = load()
    model, vocab, stop = s["model"], s["vocab"], s["stop"]
    prompt, _ = _prompt(s, question)
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


def stream(question: str, max_new: int = 64, budget_s: float = 240.0):
    """Answer with the cache, yielding events as the model goes:
      {"type": "start", "prompt_tokens"}                 before reading the prompt (~7 s on Vercel)
      {"type": "token", "t", "p", "copy", "src", "q"}    one per answer token: its text, its probability,
          the copy share (1 - gate), src = [[PDF word, weight], ...], the copy head's attention
          times the copy share, summed over the resume tokens inside each word (largest first), and
          q, the same for the question and the answer so far (copying from outside the resume)
      {"type": "done", "answer", "first_token_s", "total_s", "new_tokens", "stopped_by"}
      {"type": "error", "detail"}                        if it fails
    The answer is made in its own thread, so a client that hangs up cannot leave the lock held."""
    events: queue.Queue = queue.Queue()

    def work():
        with _LOCK:
            try:
                for ev in _stream(question, max_new, budget_s):
                    events.put(ev)
            except Exception as e:
                events.put({"type": "error", "detail": str(e)})
            finally:
                events.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (ev := events.get()) is not None:
        yield ev


def _stream(question: str, max_new: int, budget_s: float):
    s = load()
    model, vocab, stop, word_of = s["model"], s["vocab"], s["stop"], s["word_of"]
    prompt, r0 = _prompt(s, question)
    words_at, pos, dec = [], 0, vocab.decoder()  # the PDF words each prompt token falls in
    r_end = len(prompt)  # first token after the resume
    for i, t in enumerate(prompt):
        if pos >= r0 + len(s["resume"]) and r_end == len(prompt):
            r_end = i
        n = len(dec.decode([t]))
        lo, hi = max(pos - r0, 0), min(pos + n - r0, len(word_of))
        words_at.append([int(w) for w in set(word_of[lo:hi].tolist()) - {-1}] if lo < hi else [])
        pos += n
    yield {"type": "start", "prompt_tokens": len(prompt)}
    max_new = min(max_new, model.max_len - len(prompt) - 1)
    toks, out, t_start = [], vocab.decoder(), time.time()
    p = model.prefill(prompt)
    first_s, stopped = time.time() - t_start, "max_new"
    for _ in range(max_new):
        nxt = int(p.argmax())
        if nxt in stop:
            stopped = "eos"
            break
        toks.append(nxt)
        piece, share = out.decode([nxt]), 1.0 - model.copy_gate
        attn = model.copy_attn[: model.pos]
        heat: dict[int, float] = {}
        for j in np.flatnonzero(attn[: len(prompt)] > 1e-3):
            for w in words_at[j]:
                heat[w] = heat.get(w, 0.0) + share * float(attn[j]) / len(words_at[j])
        src = sorted(([w, round(v, 3)] for w, v in heat.items() if v >= 0.01), key=lambda x: -x[1])[:12]
        if piece:  # capcode markers decode to "" and carry no text
            yield {"type": "token", "t": piece, "p": round(float(p[nxt]), 3), "copy": round(share, 3), "src": src,
                   "q": round(share * float(attn[r_end:].sum()), 3)}
        if time.time() - t_start > budget_s:
            stopped = "time budget"
            break
        p = model.step(nxt)
    yield {"type": "done", "answer": vocab.decode(toks).strip(), "first_token_s": round(first_s, 2),
           "total_s": round(time.time() - t_start, 2), "new_tokens": len(toks), "stopped_by": stopped}
