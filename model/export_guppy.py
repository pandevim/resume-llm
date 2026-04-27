#!/usr/bin/env python3
"""
export_guppy.py — Download GuppyLM and serialize to weights.bin.

Verified against arman-bd/guppylm-9M:
  - HF files:        pytorch_model.bin, tokenizer.json, config.json
  - State dict keys: tok_emb.weight, pos_emb.weight,
                     blocks.{i}.{norm1,norm2}.{weight,bias},
                     blocks.{i}.attn.{qkv,out}.{weight,bias},
                     blocks.{i}.ffn.{up,down}.{weight,bias},
                     norm.{weight,bias}
                     (lm_head.weight is tied to tok_emb.weight; we skip it)
  - Pre-norm; LayerNorm; ReLU FFN; learned positional embeddings.

Output binary layout (read by engine/inference.c):

  Header (little-endian):
    char[4]   magic   = "GPPY"
    u32       version = 1
    i32       vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len
    i32       bos_id, eos_id, pad_id, unk_id

  Vocab:
    u32       total_bytes
    repeat vocab_size times, in token-id order:
      u16     byte_length
      bytes   token_string

  Tensors (all fp16):
    tok_emb     [vocab_size,  d_model]
    pos_emb     [max_seq_len, d_model]
    for each block:
      norm1.{w,b}    [d_model] x2
      attn.qkv.{w,b} [3*d_model, d_model] / [3*d_model]
      attn.out.{w,b} [d_model, d_model]   / [d_model]
      norm2.{w,b}    [d_model] x2
      ffn.up.{w,b}   [d_ff, d_model]      / [d_ff]
      ffn.down.{w,b} [d_model, d_ff]      / [d_model]
    norm.{w,b}       [d_model] x2
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"GPPY"
VERSION = 1


def fp16_bytes(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr.astype(np.float16)).tobytes()


def get(state, key, *, shape=None):
    if key not in state:
        keys_preview = "\n  ".join(sorted(state.keys())[:40])
        raise KeyError(
            f"missing key: {key}\n"
            f"available (first 40):\n  {keys_preview}"
        )
    t = state[key].detach().cpu().float().numpy()
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"{key}: expected {shape}, got {t.shape}")
    return np.ascontiguousarray(t)


def export(state, tokenizer, cfg, out_path: Path):
    V = cfg["vocab_size"]
    D = cfg["d_model"]
    H = cfg["n_heads"]
    L = cfg["n_layers"]
    F = cfg["d_ff"]
    M = cfg["max_seq_len"]

    bos = cfg["bos_id"]
    eos = cfg["eos_id"]
    pad = cfg["pad_id"]
    unk = cfg.get("unk_id", -1)

    # Vocab: ids -> surface bytes, in id order.
    vocab_strings = [b""] * V
    for tok, idx in tokenizer.get_vocab().items():
        if 0 <= idx < V:
            vocab_strings[idx] = tok.encode("utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        for n in (V, D, H, L, F, M):
            f.write(struct.pack("<i", n))
        for n in (bos, eos, pad, unk):
            f.write(struct.pack("<i", n))

        # Vocab
        total = sum(len(s) for s in vocab_strings)
        f.write(struct.pack("<I", total))
        for s in vocab_strings:
            f.write(struct.pack("<H", len(s)))
            f.write(s)

        # Embeddings
        f.write(fp16_bytes(get(state, "tok_emb.weight", shape=(V, D))))
        f.write(fp16_bytes(get(state, "pos_emb.weight", shape=(M, D))))

        # Blocks
        for i in range(L):
            f.write(fp16_bytes(get(state, f"blocks.{i}.norm1.weight", shape=(D,))))
            f.write(fp16_bytes(get(state, f"blocks.{i}.norm1.bias",   shape=(D,))))

            f.write(fp16_bytes(get(state, f"blocks.{i}.attn.qkv.weight", shape=(3 * D, D))))
            f.write(fp16_bytes(get(state, f"blocks.{i}.attn.qkv.bias",   shape=(3 * D,))))

            f.write(fp16_bytes(get(state, f"blocks.{i}.attn.out.weight", shape=(D, D))))
            f.write(fp16_bytes(get(state, f"blocks.{i}.attn.out.bias",   shape=(D,))))

            f.write(fp16_bytes(get(state, f"blocks.{i}.norm2.weight", shape=(D,))))
            f.write(fp16_bytes(get(state, f"blocks.{i}.norm2.bias",   shape=(D,))))

            f.write(fp16_bytes(get(state, f"blocks.{i}.ffn.up.weight", shape=(F, D))))
            f.write(fp16_bytes(get(state, f"blocks.{i}.ffn.up.bias",   shape=(F,))))

            f.write(fp16_bytes(get(state, f"blocks.{i}.ffn.down.weight", shape=(D, F))))
            f.write(fp16_bytes(get(state, f"blocks.{i}.ffn.down.bias",   shape=(D,))))

        # Final norm
        f.write(fp16_bytes(get(state, "norm.weight", shape=(D,))))
        f.write(fp16_bytes(get(state, "norm.bias",   shape=(D,))))

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"wrote {out_path} ({size_mb:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="arman-bd/guppylm-9M")
    ap.add_argument("--out",  default="model/weights.bin", type=Path)
    ap.add_argument("--inspect", action="store_true",
                    help="Print state-dict keys and exit")
    args = ap.parse_args()

    try:
        import torch
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as e:
        print(f"missing dependency: {e}\n"
              f"  pip install torch huggingface_hub tokenizers", file=sys.stderr)
        sys.exit(1)

    ckpt_path = hf_hub_download(args.repo, "pytorch_model.bin")
    tok_path  = hf_hub_download(args.repo, "tokenizer.json")
    cfg_path  = hf_hub_download(args.repo, "config.json")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    if args.inspect:
        for k, v in sorted(state.items()):
            print(f"{k:60s} {tuple(v.shape)}")
        return

    tokenizer = Tokenizer.from_file(tok_path)
    with open(cfg_path) as fp:
        hf_cfg = json.load(fp)

    cfg = {
        "vocab_size":  hf_cfg["vocab_size"],
        "d_model":     hf_cfg["hidden_size"],
        "n_heads":     hf_cfg["num_attention_heads"],
        "n_layers":    hf_cfg["num_hidden_layers"],
        "d_ff":        hf_cfg["intermediate_size"],
        "max_seq_len": hf_cfg["max_position_embeddings"],
        "bos_id":      hf_cfg.get("bos_token_id", 1),
        "eos_id":      hf_cfg.get("eos_token_id", 2),
        "pad_id":      hf_cfg.get("pad_token_id", 0),
        # GuppyLM doesn't define an unk; pass -1 so the C engine knows.
        "unk_id":      tokenizer.token_to_id("<unk>")
                       if tokenizer.token_to_id("<unk>") is not None else -1,
    }

    export(state, tokenizer, cfg, args.out)


if __name__ == "__main__":
    main()
