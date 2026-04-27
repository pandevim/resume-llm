# resume_pdf_v2 — a PDF that talks back

A single file that is **simultaneously a valid PDF and a Linux ELF
executable**. Open it in a PDF reader and you see a cover page with the
original document attached. Run it from a shell and it answers prompts
using a tiny language model embedded in the binary.

```
$ ./dist/resume.pdf "hi guppy"
oh hello. the water is fresh today.

$ ./dist/resume.pdf "do you like bubbles"
the bubbles are back. bubbles are my second favorite thing after food.

$ file dist/resume.pdf
dist/resume.pdf: ELF 64-bit LSB executable, x86-64, statically linked, stripped

$ pdfinfo dist/resume.pdf | head -3
Pages:           1
Page size:       612 x 792 pts (letter)
PDF version:     1.4
```

## Roadmap

- **Phase 1 (this scaffold — working)** — embed
  [GuppyLM](https://huggingface.co/arman-bd/guppylm-9M) (8.7M params,
  MIT) inside the polyglot. The executable is a generic chat with
  Guppy; it ignores the PDF content. End-to-end pipeline verified:
  weight export → C inference engine → static ELF → ELF/PDF polyglot.
- **Phase 2** — extractive QA with a pointer/copy head. At build time
  we run GuppyLM's encoder over the source PDF tokens and cache the K/V
  vectors into the binary. At inference time the question goes through
  GuppyLM and a single cross-attention step picks out the most relevant
  PDF tokens. The attention distribution _is_ the answer; phase 2 also
  ships a viridis heatmap renderer (`viz/`).

## Architecture (phase 1)

```
   ┌─────────────────── resume.pdf ───────────────────┐
   │  ELF header (bytes 0..63)                          │  ← Linux loader
   │  Program header table (64..400)                    │      maps via phdrs
   │  '%PDF-1.4' marker injected at byte 400            │  ← PDF reader
   │  ...rest of ELF body...                            │      scans first 1KB
   │  .text   inference.c                               │      for marker, then
   │  .rodata weights.bin (GuppyLM, fp16, ~17 MB)       │      parses forward
   │  ...end of ELF...                                  │
   │  PDF objects: catalog, cover page, /EmbeddedFile   │
   │  xref + trailer + %%EOF                            │
   └────────────────────────────────────────────────────┘
```

- **ELF** is statically linked against musl libc; no dynamic loader, no
  network, no Python at runtime.
- **GuppyLM** is loaded from a flat fp16 blob produced by
  `model/export_guppy.py` and linked in via `objcopy`.
- **Tokenizer** is ByteLevel BPE (GPT-2 style) — special tokens matched
  literally, then bytes go through the byte→codepoint table in
  `engine/bytelevel.h`, then greedy longest-match over the vocab.
  Output is reverse-mapped before printing.
- **PDF readers** locate `%PDF-1.4` within the first 1024 bytes. The
  assembler picks a NUL run **outside the program header table** — the
  zeros inside phdr fields are load-bearing (they're the upper bytes of
  small `p_flags` values and the literal-zero `p_offset`), so naïve
  "find any NUL run" corrupts the loader.
- **Original PDF** is attached as an `/EmbeddedFile` so it shows up in
  the Attachments pane of any standards-compliant viewer.

## Layout

```
.
├── build.sh                       end-to-end pipeline
├── engine/
│   ├── inference.c                forward pass, tokenizer, sampler
│   ├── bytelevel.h                generated GPT-2 byte_to_unicode table
│   └── Makefile                   musl-static + objcopy
├── model/
│   └── export_guppy.py            HF checkpoint -> weights.bin
├── polyglot/
│   ├── assemble.py                ELF + PDF merger (phdr-aware)
│   └── make_sample_pdf.py         tiny sample doc for the worked example
├── examples/                      sample PDFs (generated)
├── viz/                           phase-2 attention heatmap (stub)
└── dist/                          output: resume.pdf
```

## Build

Requires:

- `musl-gcc` (Debian/Ubuntu: `sudo apt install musl-tools`).
  Falls back to `gcc -static` if musl isn't installed.
- `objcopy` (`binutils`)
- Python 3 with `torch`, `huggingface_hub`, `tokenizers` (export step
  only — at runtime the polyglot has zero Python dependencies)

```bash
./build.sh raw/resume.pdf        # embed a different source PDF
```

The pipeline:

1. `model/export_guppy.py` downloads `arman-bd/guppylm-9M` and writes
   `model/weights.bin` (~17 MB, fp16).
2. `engine/Makefile` wraps that file with `objcopy` and statically links
   it together with `engine/inference.c` into `engine/guppy`.
3. `polyglot/assemble.py` parses the ELF program-header table, finds a
   safe NUL run in the first KB outside any header table, writes
   `%PDF-1.4` there, and appends a PDF wrapper with the source as an
   `/EmbeddedFile`.

Final output: `dist/resume.pdf` (~17 MB, executable, also a valid PDF).

### Regenerating `engine/bytelevel.h`

The file is committed and rarely changes (it encodes the fixed GPT-2
ByteLevel mapping), but if you need to regenerate it:

```bash
python3 -c '
def b2u():
    bs = (list(range(33,127)) + list(range(161,173)) + list(range(174,256)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n += 1
    return {b:c for b,c in zip(bs,cs)}
m = b2u()
print("#ifndef BYTELEVEL_H\n#define BYTELEVEL_H")
print("static const char* const bytelevel[256] = {")
for b in range(256):
    s = chr(m[b]).encode(); print("    \"" + "".join(f"\\\\x{x:02x}" for x in s) + "\",")
print("};\nstatic const int bytelevel_len[256] = {")
for b in range(256): print(f"    {len(chr(m[b]).encode())},")
print("};")
mx = max(m.values())
print(f"#define BYTELEVEL_INV_SIZE 0x{mx+1:04x}")
print("static const int bytelevel_inv[BYTELEVEL_INV_SIZE] = {")
inv = {m[b]:b for b in range(256)}
for cp in range(mx+1): print(f"    {inv.get(cp,-1)},")
print("};\n#endif")
' > engine/bytelevel.h
```

## Usage

```bash
chmod +x dist/resume.pdf
./dist/resume.pdf "do you like bubbles"

# or open in any PDF reader — the original is in the Attachments pane.
xdg-open dist/document.pdf
```

## Verified architecture (against `arman-bd/guppylm-9M`)

| Component                                         | Value                                              |
| ------------------------------------------------- | -------------------------------------------------- |
| Params                                            | 8.7M (≈ 8,726,016 — checked)                       |
| Layers                                            | 6                                                  |
| `d_model` / `n_heads` / `head_dim`                | 384 / 6 / 64                                       |
| `d_ff`                                            | 768                                                |
| `vocab_size` (config) / vocab entries (tokenizer) | 4096 / 2418                                        |
| `max_seq_len`                                     | 128                                                |
| Norm                                              | LayerNorm, **pre-norm**                            |
| FFN                                               | ReLU                                               |
| Position                                          | Learned embeddings                                 |
| LM head                                           | **Weight-tied** with `tok_emb` (skipped at export) |
| Tokenizer                                         | BPE, **ByteLevel** pretokenizer/decoder            |
| Specials                                          | `<pad>`=0, `<\|im_start\|>`=1, `<\|im_end\|>`=2    |
| Chat template                                     | `<\|im_start\|>{role}\n{content}<\|im_end\|>`      |
| Sampling                                          | temperature 0.7, top-k 50 (matches reference)      |

State-dict key names used by the exporter (verified via
`python3 model/export_guppy.py --inspect`):

```
tok_emb.weight                     pos_emb.weight
blocks.{i}.norm1.{weight,bias}     blocks.{i}.norm2.{weight,bias}
blocks.{i}.attn.qkv.{weight,bias}  blocks.{i}.attn.out.{weight,bias}
blocks.{i}.ffn.up.{weight,bias}    blocks.{i}.ffn.down.{weight,bias}
norm.{weight,bias}
```

## Remaining caveats

- **BPE matching is greedy longest-match, not full merge order.** We
  apply the correct ByteLevel preprocessing and decoding, but skip the
  BPE merge-rule application — at the current vocab and prompt sizes
  longest-match almost always picks the same tokens HF would, and chat
  output is coherent. Edge-case prompts may tokenize slightly
  differently from the reference Python.
- **No KV cache.** The forward pass recomputes the full sequence on
  every step. At ≤128 tokens this is fast enough on a single CPU core;
  add a cache only if it becomes a bottleneck.
- **Strict PDF viewers (e.g. pdf.js)** may reject the file because
  bytes precede `%PDF-`. Most desktop readers (Evince, Okular, Adobe,
  macOS Preview, Foxit) accept it — the spec only requires the marker
  within the first 1024 bytes, which we honor.
- **Linux x86_64 only.** No fallback for ARM, macOS, or Windows in this
  iteration.
