"""Export the 83M persona model (PDFLLM p1_grpo_persona_nosquad) for pdfllm/engine.py.

  python export_onnx.py <weights dir: w0.npz (fp16 state dict, no lm_head) + weights.json> <out dir>
then save chunk.onnx pre-optimized (SessionOptions.optimized_model_filepath, ORT_ENABLE_EXTENDED) as
pdfllm/onnx/chunk.onnx, and split w0.npz into files under 100 MB (Vercel's per-file limit).

Original notes:
 to two ONNX graphs that share one set of fp16 weight files.

  prefill.onnx  ids (1, T)                      -> probs (V,), cache
  step.onnx     tok (1,), pos (1,), cache       -> probs (V,), cache

cache = attention K/V (L, kv, MAX, hd), Mamba-3 theta/ssm/k/v (L, ...), copy-head keys (MAX, C), ids (MAX,).
Fixed-size caches; step writes position `pos` with a one-hot mask and masks attention to <= pos,
so the graph has no data-dependent shapes. Same maths as pdfllm/npmodel.py (a port of models/decode.py).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx.external_data_helper import convert_model_to_external_data

DEMO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
EPS, A_FLOOR = 1e-5, 1e-4


def rms(x, w):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * w


def rot(v, theta):
    R = theta.shape[-1]
    a0, a1 = v[..., : 2 * R : 2], v[..., 1 : 2 * R : 2]
    c, s = theta.cos(), theta.sin()
    return torch.cat([torch.stack((a0 * c - a1 * s, a0 * s + a1 * c), -1).flatten(-2), v[..., 2 * R :]], -1)


def weight_inputs(w):
    """fp32 graph inputs: 2-D weights transposed to (in, out) under "T__..."; the embedding also as is
    (for the lookup). Keys use "__" for "." (parameter names). The app builds the same dict at load."""
    out = {}
    for k, v in w.items():
        v = v.astype(np.float32)
        if v.ndim == 2:
            out[("T." + k).replace(".", "__")] = np.ascontiguousarray(v.T)
        if v.ndim != 2 or k == "tok_emb.weight":
            out[k.replace(".", "__")] = v
    return out


class LM(nn.Module):
    def __init__(self, w, a):
        super().__init__()
        # fp32 weights; the app passes them in at load (shared by both sessions), see engine.py
        self.p = nn.ParameterDict({k: nn.Parameter(torch.from_numpy(v), requires_grad=False)
                                   for k, v in weight_inputs(w).items()})
        self.L, self.d, self.nh, self.nkv = a["layers"], a["hidden"], a["heads"], a["kv_heads"]
        self.hd = self.d // self.nh
        self.inter = int(a["ssm_expand"] * self.d)
        self.P, self.N = a["ssm_head_dim"], a["state"]
        self.H, self.R = self.inter // self.P, (self.N // 2) // 2
        self.MAX, self.C, self.V = a["seq"], a["copy_dim"], w["tok_emb.weight"].shape[0]
        inv = 1.0 / 10000.0 ** (torch.arange(0, self.hd, 2).float() / self.hd)
        ang = torch.outer(torch.arange(self.MAX).float(), inv)
        self.register_buffer("cos", ang.cos())
        self.register_buffer("sin", ang.sin())

    def g(self, name, l=None):
        if name.startswith("T."):
            key = "T." + (f"blocks.{l}.{name[2:]}" if l is not None else name[2:])
        else:
            key = f"blocks.{l}.{name}" if l is not None else name
        key = key.replace(".", "__")
        return self.p[key].float()

    def lin(self, x, name, l=None):
        return x @ self.g("T." + name, l)  # weights stored as (in, out): no transpose per call

    def qkv(self, l, u, c, s):
        T = u.shape[0]
        q = self.lin(u, "mixer.parts.0.q.weight", l).view(T, self.nh, self.hd).transpose(0, 1)
        kv = self.lin(u, "mixer.parts.0.kv.weight", l).view(T, 2, self.nkv, self.hd)
        k, v = kv[:, 0].transpose(0, 1), kv[:, 1].transpose(0, 1)
        q, k = rms(q, self.g("mixer.parts.0.q_norm.weight", l)), rms(k, self.g("mixer.parts.0.k_norm.weight", l))
        rope = lambda x: torch.stack((x[..., ::2] * c - x[..., 1::2] * s, x[..., ::2] * s + x[..., 1::2] * c), -1).flatten(-2)
        return rope(q), rope(k), v

    def attend(self, l, q, K, V, mask):
        rep = self.nh // self.nkv
        K, V = K.repeat_interleave(rep, 0), V.repeat_interleave(rep, 0)
        sc = (q @ K.transpose(1, 2)) / math.sqrt(self.hd)
        out = sc.masked_fill(~mask, float("-inf")).softmax(-1) @ V
        return self.lin(out.transpose(0, 1).reshape(q.shape[1], -1), "mixer.parts.0.o.weight", l)

    def m3_inputs(self, l, u):
        p = "mixer.parts.1."
        T = u.shape[0]
        i, H, N = self.inter, self.H, self.N
        z, x, B, C, dd_dt, dd_A, trap, ang = torch.split(self.lin(u, p + "in_proj.weight", l), [i, i, N, N, H, H, H, self.R], -1)
        dt = F.softplus(dd_dt + self.g(p + "dt_bias", l))
        adt = (-F.softplus(dd_A)).clamp(max=-A_FLOOR) * dt
        dth = torch.tanh(ang)[:, None, :] * math.pi * dt[..., None]
        q_pre = rms(C, self.g(p + "C_norm.weight", l))[:, None, :] + self.g(p + "C_bias", l)[:, 0]
        k_pre = rms(B, self.g(p + "B_norm.weight", l))[:, None, :] + self.g(p + "B_bias", l)[:, 0]
        return z.view(T, H, self.P), x.view(T, H, self.P), q_pre, k_pre, adt, dt, torch.sigmoid(trap), dth

    def m3_out(self, l, y, x, z):
        y = (y + self.g("mixer.parts.1.D", l)[:, None] * x) * F.silu(z)
        return self.lin(y.reshape(y.shape[0], -1), "mixer.parts.1.out_proj.weight", l)

    def mlp(self, l, x):
        h = rms(x, self.g("norm2.weight", l))
        return self.lin(F.silu(self.lin(h, "mlp.gate.weight", l)) * self.lin(h, "mlp.up.weight", l), "mlp.down.weight", l)

    def mix(self, h, ck, ids, valid):
        logits = self.lin(h, "tok_emb.weight")
        sc = (self.lin(h, "copy.q.weight") @ ck.t()) * self.C ** -0.5
        a = sc.masked_fill(~valid, float("-inf")).softmax(-1)
        p_copy = torch.zeros(self.V).scatter_add(0, ids, a)
        gte = torch.sigmoid(self.lin(h, "copy.gate.weight") + self.g("copy.gate.bias"))
        return gte * logits.softmax(-1) + (1 - gte) * p_copy


class Prefill(LM):
    def forward(self, ids):
        ids = ids[0]
        T = ids.shape[0]
        x = self.g("tok_emb.weight")[ids]
        c, s = self.cos[:T], self.sin[:T]
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        pad = lambda t: F.pad(t, (0, 0, 0, self.MAX - T))
        Ks, Vs, TH, SS, KK, VV = [], [], [], [], [], []
        for l in range(self.L):
            u = rms(x, self.g("norm1.weight", l))
            q, k, v = self.qkv(l, u, c, s)
            a = self.attend(l, q, k, v, causal)
            Ks.append(pad(k)), Vs.append(pad(v))
            z, xx, q_pre, k_pre, adt, dt, trap, dth = self.m3_inputs(l, u)
            theta = dth.cumsum(0)
            qm, km = rot(q_pre, theta), rot(k_pre, theta)
            gamma = (trap * dt).t()
            beta = (adt.exp() * dt * (1 - trap)).t()
            cc = adt.cumsum(0).t()
            decay = (cc[:, :, None] - cc[:, None, :]).masked_fill(~causal, float("-inf")).exp()
            w = decay * gamma[:, None, :]
            w = w + F.pad(decay[:, :, 1:] * beta[:, None, 1:], (0, 1))
            qh, kh, xh = qm.transpose(0, 1), km.transpose(0, 1), xx.transpose(0, 1)
            y = ((w * (qh @ kh.transpose(1, 2))) @ xh).transpose(0, 1)
            SS.append((xh * w[:, -1, :, None]).transpose(1, 2) @ kh)
            TH.append(torch.remainder(theta[-1], 2 * math.pi)), KK.append(km[-1]), VV.append(xx[-1])
            x = x + a + self.m3_out(l, y, xx, z)
            x = x + self.mlp(l, x)
        h = rms(x, self.g("norm.weight"))
        ck = pad(self.lin(h, "copy.k.weight"))
        ids_full = F.pad(ids, (0, self.MAX - T))
        valid = torch.arange(self.MAX) < T
        probs = self.mix(h[-1], ck, ids_full, valid)
        return (probs, torch.stack(Ks), torch.stack(Vs), torch.stack(TH), torch.stack(SS), torch.stack(KK),
                torch.stack(VV), ck, ids_full)


class Step(LM):
    def forward(self, tok, pos, Kc, Vc, TH, SS, KK, VV, ck, ids_full):
        x = self.g("tok_emb.weight")[tok]                        # (1, d)
        c, s = self.cos[pos], self.sin[pos]                     # (1, hd/2)
        here = torch.arange(self.MAX) == pos                    # (MAX,)
        seen = (torch.arange(self.MAX) <= pos)[None, :]         # (1, MAX)
        Ko, Vo, THo, SSo, KKo, VVo = [], [], [], [], [], []
        for l in range(self.L):
            u = rms(x, self.g("norm1.weight", l))
            q, k, v = self.qkv(l, u, c, s)
            K = torch.where(here[None, :, None], k, Kc[l])
            V = torch.where(here[None, :, None], v, Vc[l])
            Ko.append(K), Vo.append(V)
            a = self.attend(l, q, K, V, seen)
            z, xx, q_pre, k_pre, adt, dt, trap, dth = (t[0] for t in self.m3_inputs(l, u))
            theta = torch.remainder(TH[l] + dth, 2 * math.pi)
            qm, km = rot(q_pre, theta), rot(k_pre, theta)
            al = adt.exp()
            beta, gamma = al * dt * (1 - trap), trap * dt
            ssm = (SS[l] * al[:, None, None] + (beta[:, None] * VV[l])[:, :, None] * KK[l][:, None, :]
                   + (gamma[:, None] * xx)[:, :, None] * km[:, None, :])
            y = (ssm @ qm[:, :, None])[..., 0]
            THo.append(theta), SSo.append(ssm), KKo.append(km), VVo.append(xx)
            x = x + a + self.m3_out(l, y[None], xx[None], z[None])
            x = x + self.mlp(l, x)
        h = rms(x, self.g("norm.weight"))[0]
        ck = torch.where(here[:, None], self.lin(h, "copy.k.weight")[None], ck)
        ids_full = torch.where(here, tok, ids_full)
        probs = self.mix(h, ck, ids_full, seen[0])
        return (probs, torch.stack(Ko), torch.stack(Vo), torch.stack(THo), torch.stack(SSo), torch.stack(KKo),
                torch.stack(VVo), ck, ids_full)


class Chunk(LM):
    """Read C new tokens at positions pos..pos+C-1 on top of the caches (C = 1 is one decode step).
    Memory is O(C x MAX) instead of the full prefill's O(T x T): a long prompt is read in chunks."""

    def forward(self, ids, pos, Kc, Vc, TH, SS, KK, VV, ck, ids_full):
        ids = ids[0]
        C = ids.shape[0]
        x = self.g("tok_emb.weight")[ids]
        tpos = pos + torch.arange(C)                              # (C,) absolute positions
        c, s = self.cos[tpos], self.sin[tpos]
        j = torch.arange(self.MAX)
        new = (j >= pos) & (j < pos + C)                          # cache slots written by this chunk
        src = (j - pos).clamp(0, C - 1)                           # which chunk row lands in slot j
        seen = j[None, :] <= tpos[:, None]                        # (C, MAX) causal over the whole cache
        causal = torch.ones(C, C, dtype=torch.bool).tril()
        Ko, Vo, THo, SSo, KKo, VVo = [], [], [], [], [], []
        for l in range(self.L):
            u = rms(x, self.g("norm1.weight", l))
            q, k, v = self.qkv(l, u, c, s)
            K = torch.where(new[None, :, None], k[:, src], Kc[l])
            V = torch.where(new[None, :, None], v[:, src], Vc[l])
            Ko.append(K), Vo.append(V)
            a = self.attend(l, q, K, V, seen)
            # Mamba-3 over the chunk, starting from state (h0, theta0, previous k/v):
            # h_t = e^{c_t} h0 + beta_0 e^{c_t - c_0} v_prev k_prev^T + sum_s w(t, s) v_s k_s^T
            z, xx, q_pre, k_pre, adt, dt, trap, dth = self.m3_inputs(l, u)
            theta = TH[l][None] + dth.cumsum(0)
            qm, km = rot(q_pre, theta), rot(k_pre, theta)
            gamma = (trap * dt).t()                                # (H, C)
            beta = (adt.exp() * dt * (1 - trap)).t()
            cc = adt.cumsum(0).t()                                 # (H, C)
            decay = (cc[:, :, None] - cc[:, None, :]).masked_fill(~causal, float("-inf")).exp()
            w = decay * gamma[:, None, :] + F.pad(decay[:, :, 1:] * beta[:, None, 1:], (0, 1))
            qh, kh, xh = qm.transpose(0, 1), km.transpose(0, 1), xx.transpose(0, 1)   # (H, C, .)
            e0 = cc.exp()                                          # (H, C) = e^{c_t}
            b0 = beta[:, 0:1] * decay[:, :, 0]                     # (H, C) = beta_0 e^{c_t - c_0}
            y = ((w * (qh @ kh.transpose(1, 2))) @ xh                                   # within the chunk
                 + e0[..., None] * (qh @ SS[l].transpose(1, 2))                         # h0 q_t
                 + b0[..., None] * (qh @ KK[l][:, :, None]) * VV[l][:, None, :])        # previous token
            y = y.transpose(0, 1)
            ssm = (e0[:, -1, None, None] * SS[l] + b0[:, -1, None, None] * VV[l][:, :, None] * KK[l][:, None, :]
                   + (xh * w[:, -1, :, None]).transpose(1, 2) @ kh)
            THo.append(torch.remainder(theta[-1], 2 * math.pi)), SSo.append(ssm), KKo.append(km[-1]), VVo.append(xx[-1])
            x = x + a + self.m3_out(l, y, xx, z)
            x = x + self.mlp(l, x)
        h = rms(x, self.g("norm.weight"))
        ck = torch.where(new[:, None], self.lin(h, "copy.k.weight")[src], ck)
        ids_full = torch.where(new, ids[src], ids_full)
        probs = self.mix(h[-1], ck, ids_full, j <= pos + C - 1)
        return (probs, torch.stack(Ko), torch.stack(Vo), torch.stack(THo), torch.stack(SSo), torch.stack(KKo),
                torch.stack(VVo), ck, ids_full)


CACHE = ["K", "V", "theta", "ssm", "sk", "sv", "copy_k", "ids_full"]


def main():
    meta = json.loads((DEMO / "weights.json").read_text())
    w = {}
    for i in range(meta["n_shards"]):
        with np.load(DEMO / f"w{i}.npz") as z:
            w.update({k: z[k] for k in z.files})
    a = meta["args"]
    pre, stp = Prefill(w, a).eval(), Step(w, a).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = torch.randint(0, 16000, (1, 20))
    with torch.no_grad():
        outs = pre(ids)
    torch.onnx.export(pre, (ids,), OUT / "prefill.onnx", input_names=["ids"], output_names=["probs"] + CACHE,
                      dynamic_axes={"ids": {1: "T"}}, opset_version=17, do_constant_folding=False, dynamo=False)
    chk = Chunk(w, a).eval()
    cache = [torch.zeros_like(t) for t in outs[1:]]
    torch.onnx.export(chk, (ids[:, :8], torch.tensor(0), *cache), OUT / "chunk.onnx", input_names=["ids", "pos"] + CACHE,
                      output_names=["probs"] + [c + "_out" for c in CACHE], dynamic_axes={"ids": {1: "C"}},
                      opset_version=17, do_constant_folding=False, dynamo=False)
    for name in ("prefill", "chunk"):  # weights become graph inputs, fed once-built arrays by the app
        m = onnx.load(OUT / f"{name}.onnx")
        keep = []
        for t in m.graph.initializer:
            if t.name.startswith("p."):
                m.graph.input.append(onnx.helper.make_tensor_value_info(t.name, t.data_type, list(t.dims)))
            else:
                keep.append(t)
        del m.graph.initializer[:]
        m.graph.initializer.extend(keep)
        onnx.save(m, OUT / f"{name}.onnx")
    print("ok")


if __name__ == "__main__":
    main()
