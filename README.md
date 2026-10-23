# resume-llm — chat with my resume

[self-aware-resume.vercel.app](https://self-aware-resume.vercel.app) shows my resume above a chat box.
The answers come from an 83M-parameter language model that reads the resume in its context and
answers as me, in the first person. It runs on CPU inside a Vercel function, with no GPU and no
outside API. While it answers, the PDF lights up where each word was copied from.

```
$ curl -s -X POST -F "message=Where did you study?" https://self-aware-resume.vercel.app/api/chat
{"response":"I studied at Stevens Institute of Technology, Hoboken, NJ."}
```

## Copy-head heatmap (2026-10-20)

The model has a copy head: at each step it attends over every earlier token and mixes
`next = gate * vocab + (1 - gate) * copy`. The site shows that head live:

- **On the PDF**, each answer token lights the words it attended to, weighted by its copy share
  (1 - gate). Words built up over the answer stay shaded; the newest token's words flash.
- **The token stream card** lists every answer token with where it came from and how much:
  `0.98 resume` (copied from the resume, red bar), `question` (copied from your question), or
  `0.89 model` (the vocabulary side, grey bar). The last line gives the token count and the time.
- **Hover** an answer word to show only its sources.

For "Where did you study?", "Hoboken" is 100% copied and 87% of the head's attention is on the
Hoboken in the Education line; "studied" and "of" are 0–1% copied and light nothing.

How it is wired:

- `chunk.onnx` returns two more outputs, `copy_attn` (attention over positions) and `copy_gate`.
  They were added to the existing graph as Identity nodes; `export_onnx.py` now emits them too.
- `pdfllm/tools/word_map.py` aligns the PDF's words (PyMuPDF boxes) with `pdfllm/resume.txt` and
  writes `pdfllm/resume_words.json`. 379 of the PDF's 410 words map; the rest are separators and
  the PDF's intro line, which `resume.txt` does not have. `build.sh` reruns it.
- `pdfllm.engine.stream()` turns each step's attention into weights on PDF words and yields one
  event per token; `POST /api/chat/stream` sends them as NDJSON. `POST /api/chat` is unchanged.
- The page replays tokens at a steady pace if the response arrives in one burst, so it works
  whether or not the host streams.

## What's new (2026-10-20)

- **New model.** The chat now uses `p1_grpo_persona_nosquad`, the standard model from
  [PDFLLM](https://github.com/pandevim/PDFLLM). It is an 83M hybrid: each of its 24 blocks runs
  attention and Mamba-3 in parallel, and a copy head lets it quote names, dates and numbers straight
  from the resume. It was pretrained on 100B tokens, then tuned with SFT and GRPO on resume and
  persona question answering. It replaces GuppyLM, which ignored the resume (see
  [Decommissioned](#decommissioned-guppylm-polyglot)).
- **ONNX Runtime instead of PyTorch.** PyTorch alone is larger than Vercel's 500 MB limit for a
  Python function. The model is exported to ONNX and run with `onnxruntime`, whose CPU package is
  about 63 MB. The whole function is about 290 MB.
- **KV/SSM cache.** The model reads the resume once and then adds one token at a time, reusing the
  attention keys/values and the Mamba-3 state. Without the cache, every new token re-reads the
  whole ~950-token prompt.
- **Chunked prompt reading.** The prompt is read 256 tokens at a time on top of the cache, so
  memory grows with 256 × 1024 instead of the prompt length squared. Reading the whole prompt in
  one call peaked near 1.9 GB; chunks of 256 peak near 950 MB. The next-token probabilities match
  a single full read (largest difference 1.8e-7 when checked with 128-token chunks).
- **The site's PDF is now a plain PDF.** `dist/resume.pdf` is the rendered resume (160 KB). It used
  to be the 17.7 MB ELF/PDF polyglot.

Measured on Vercel (Hobby plan: 1 vCPU, 2 GB), question "Where did you study?", 952-token prompt:

| | with cache | without cache |
|---|---|---|
| read the resume (first token) | ~7 s | ~7 s |
| each further token | 0.14 s | 6.6 s |
| whole answer (19 tokens) | ~10 s | 131 s |
| peak memory | ~950 MB | ~950 MB |

The site always uses the cache.

## How it works

```
browser ──POST /api/chat[/stream] (message)──▶ api/upload.py (FastAPI)
                                         │  one answer at a time (a lock), question ≤ 500 chars
                                         ▼
                                   pdfllm/engine.py
   prompt = <sys>persona rule<sep><user>Resume:\n{pdfllm/resume.txt}\n\nQuestion: {q}<sep><assistant>
                                         │  tokenmonster 16K tokenizer
                                         ▼
                 onnx/chunk.onnx  (ONNX Runtime, CPU, fp32)
                 read the prompt in 256-token chunks, then 1 token per call
                 caches: attention K/V, Mamba-3 state, copy-head keys
                                         │  greedy, stops at <eos>/<sep> or 64 tokens
                                         ▼
              {"response": "..."}  or, streamed, one line per token:
              {"type":"token","t":" Hoboken","p":1.0,"copy":1.0,"src":[[word, weight], ...],"q":0.0}
```

- **Weights** are stored as fp16 in `pdfllm/onnx/w0.npz` and `w1.npz` (166 MB, split because
  Vercel caps each file at 100 MB). At start-up they are cast to fp32 and passed to the graph as
  inputs, with 2-D weights pre-transposed.
- **The graph** (`chunk.onnx`, 1 MB) holds the computation only. It is saved already optimized,
  and the session runs with `enable_mem_reuse=False`. With memory reuse on, ORT spent about 13 s
  planning this 13k-node graph at start-up; with it off, start-up takes about 0.2 s and speed and
  peak memory are unchanged.
- **The resume the model reads** is `pdfllm/resume.txt`, a cleaned text version in the format the
  model was trained on. It is not the PDF.

## Layout

```
.
├── index.html            the site: PDF (pdf.js) with the copy-head heatmap, token stream, chat
├── api/upload.py         POST /api/chat -> {"response": ...}; POST /api/chat/stream -> NDJSON tokens
├── pdfllm/
│   ├── engine.py         load the model, build the prompt, generate (cached or uncached), stream()
│   ├── onnx/             chunk.onnx + fp16 weights (w0.npz, w1.npz)
│   ├── resume.txt        the resume text the model reads
│   ├── resume_words.json PDF word boxes -> resume.txt ranges (tools/word_map.py)
│   ├── tokenizer.vocab   tokenmonster vocabulary (+ tokenmonsterserver, its helper binary)
│   └── tools/            export_onnx.py (checkpoint -> ONNX), word_map.py (PDF words -> resume.txt)
├── raw/                  resume.tex (source of truth) and its rendered PDF
├── dist/resume.pdf       the PDF the site shows
├── build.sh              render raw/resume.tex -> raw/resume.pdf + dist/resume.pdf, rebuild the word map
├── vercel.json           function config: 300 s, 2 GB, bundles pdfllm/
└── engine/ model/ polyglot/ viz/   decommissioned GuppyLM polyglot (kept for reference)
```

## Updating the resume

```bash
# 1. edit raw/resume.tex
./build.sh              # renders it to raw/resume.pdf and dist/resume.pdf
# 2. make the same edit in pdfllm/resume.txt (what the chat model reads)
python3 pdfllm/tools/word_map.py   # re-align the heatmap's word boxes (needs pymupdf)
# 3. commit and push to main; Vercel deploys from main
```

The prompt plus answer must fit in the model's 1,024-token context. The current resume is about 950
tokens with the question, which leaves room for a 64-token answer. A much longer resume would need
trimming.

## Re-exporting the model

`pdfllm/tools/export_onnx.py` turns a PDFLLM checkpoint into the graph and weights above. Its
docstring lists the steps: export, save the graph pre-optimized, split the weights under 100 MB.
The checkpoint itself (`final.pt`, 332 MB) is not in this repo; it lives with PDFLLM on the cluster.

## Decommissioned: GuppyLM polyglot

Until 2026-10-20 the site ran **GuppyLM** ([arman-bd/guppylm-9M](https://huggingface.co/arman-bd/guppylm-9M),
8.7M parameters, MIT), a small chat model that speaks as a fish. It was compiled into a static C
binary and merged with the resume into one file that was both a valid PDF and a Linux ELF
executable:

```
$ ./dist/resume.pdf "hi guppy"
oh hello. the water is fresh today.
```

The API copied that file, ran it with the question, and returned what it printed.

Why it was retired:

- **It never read the resume.** It answered every prompt as Guppy, so it could not answer questions
  about me. Making it read the PDF was planned as "phase 2" and never built.
- **The new model does read it.** On our resume eval, PDFLLM's persona model scores 0.80.

The code is still in the repo for reference, and nothing on the site uses it:

- `engine/`: C inference engine and Makefile (static musl build, weights linked with `objcopy`)
- `model/export_guppy.py`: HF checkpoint → `weights.bin`
- `polyglot/assemble.py`: ELF + PDF merger (writes `%PDF-1.4` in a safe NUL run in the first KB, outside the program headers)
- `viz/`: the unfinished phase-2 attention heatmap (the live site now has one, built from the copy head)

`build.sh` no longer builds the polyglot. To rebuild it, use `engine/Makefile` and
`polyglot/assemble.py` from the git history before commit `e0072b3`.
