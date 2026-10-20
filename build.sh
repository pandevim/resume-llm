#!/usr/bin/env bash
# build.sh — render the resume and publish it for the site:
#   1. Compile raw/resume.tex -> PDF (pdflatex x2 for refs)
#   2. Copy it to raw/resume.pdf and dist/resume.pdf (the site's PDF preview)
#   3. Map the PDF's words to pdfllm/resume.txt -> pdfllm/resume_words.json (the heatmap's boxes)
#
# The chat model reads pdfllm/resume.txt, not the PDF. That file is a hand-cleaned
# text version of the resume (the format the model was trained on), so after editing
# the .tex, update pdfllm/resume.txt to match.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SOURCE="${1:-raw/resume.tex}"
OUT="${OUT:-dist/resume.pdf}"

if [ ! -f "$SOURCE" ]; then
    echo "source not found: $SOURCE"; exit 1
fi
if ! command -v pdflatex >/dev/null 2>&1; then
    echo "pdflatex not found; install texlive (apt install texlive-latex-base texlive-latex-recommended)"
    exit 1
fi

# --- 1. Compile in a temp copy so .cls/.sty/images resolve and no build files land in raw/ ---
echo "[1/3] compiling $SOURCE"
SRC_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
SRC_BASE="$(basename "$SOURCE" .tex)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
cp -r "$SRC_DIR/." "$BUILD_DIR/"
( cd "$BUILD_DIR" && pdflatex -interaction=batchmode -halt-on-error "$SRC_BASE.tex" >/dev/null \
                  && pdflatex -interaction=batchmode -halt-on-error "$SRC_BASE.tex" >/dev/null ) \
    || { cp "$BUILD_DIR/$SRC_BASE.log" "/tmp/$SRC_BASE.log" 2>/dev/null; echo "pdflatex failed; see /tmp/$SRC_BASE.log"; exit 1; }

# --- 2. Publish ---
echo "[2/3] publishing"
mkdir -p "$(dirname "$OUT")"
cp "$BUILD_DIR/$SRC_BASE.pdf" "$SRC_DIR/$SRC_BASE.pdf"
cp "$BUILD_DIR/$SRC_BASE.pdf" "$OUT"
echo "  $SRC_DIR/$SRC_BASE.pdf"
echo "  $OUT"

# --- 3. Word boxes for the heatmap (needs PyMuPDF) ---
echo "[3/3] mapping PDF words to pdfllm/resume.txt"
if [ "$OUT" = "dist/resume.pdf" ]; then
    python3 pdfllm/tools/word_map.py || echo "  word map failed (pip install pymupdf); the heatmap uses the old pdfllm/resume_words.json"
else
    echo "  skipped: the map is built from dist/resume.pdf"
fi

echo
echo "done. If the resume content changed, update pdfllm/resume.txt too (the chat model reads that),"
echo "then rerun: python3 pdfllm/tools/word_map.py"
