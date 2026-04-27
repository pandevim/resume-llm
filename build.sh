#!/usr/bin/env bash
# build.sh — End-to-end pipeline:
#   1. Export GuppyLM from HuggingFace -> model/weights.bin
#   2. Compile engine/guppy with weights embedded as a data section
#   3. Generate (or accept) a source PDF
#   4. Merge ELF + PDF -> dist/document.pdf  (the polyglot)
#
# Phase 1 only: the executable answers any prompt by sampling Guppy.
# It does not yet read the PDF — that's phase 2.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SOURCE="${1:-raw/resume.tex}"
OUT="${OUT:-dist/resume.pdf}"

mkdir -p dist

# --- 1. Toolchain check ---
echo "[1/5] checking toolchain"
make -C engine check

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found"; exit 1
fi

# --- 2. Export weights (skip if already present and newer than this script) ---
echo "[2/5] exporting GuppyLM weights"
if [ ! -f model/weights.bin ]; then
    python3 model/export_guppy.py --out model/weights.bin
else
    echo "  model/weights.bin already exists (delete to re-export)"
fi

# --- 3. Build the static ELF ---
echo "[3/5] compiling inference engine"
make -C engine clean
make -C engine

# Sanity check: must be a static ELF.
file_out=$(file engine/guppy)
case "$file_out" in
    *"statically linked"*) echo "  ok: $file_out" ;;
    *) echo "  WARNING: not statically linked: $file_out" ;;
esac

# --- 4. Source: compile .tex -> PDF if needed ---
echo "[4/5] preparing source"
if [ ! -f "$SOURCE" ]; then
    echo "source not found: $SOURCE"; exit 1
fi

case "$SOURCE" in
    *.tex)
        if ! command -v pdflatex >/dev/null 2>&1; then
            echo "pdflatex not found; install texlive (apt install texlive-latex-base texlive-latex-recommended)"
            exit 1
        fi
        SRC_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
        SRC_BASE="$(basename "$SOURCE" .tex)"
        BUILD_DIR="$(mktemp -d)"
        trap 'rm -rf "$BUILD_DIR"' EXIT

        # Copy the source dir so .cls/.sty/images resolve relative to the .tex
        cp -r "$SRC_DIR/." "$BUILD_DIR/"
        echo "  compiling $SOURCE -> rendered PDF (pdflatex x2 for refs)"
        ( cd "$BUILD_DIR" && pdflatex -interaction=batchmode -halt-on-error "$SRC_BASE.tex" >/dev/null \
                          && pdflatex -interaction=batchmode -halt-on-error "$SRC_BASE.tex" >/dev/null ) \
            || { echo "pdflatex failed; see $BUILD_DIR/$SRC_BASE.log"; trap - EXIT; exit 1; }
        RENDERED_PDF="$BUILD_DIR/$SRC_BASE.pdf"
        ATTACH_FILE="$SOURCE"
        ;;
    *.pdf)
        RENDERED_PDF="$SOURCE"
        ATTACH_FILE="$SOURCE"
        ;;
    *)
        echo "unsupported source extension: $SOURCE (expected .tex or .pdf)"; exit 1
        ;;
esac

# --- 5. Assemble polyglot ---
echo "[5/5] assembling polyglot -> $OUT"
python3 polyglot/assemble.py \
    --elf      engine/guppy \
    --source   "$RENDERED_PDF" \
    --attach   "$ATTACH_FILE" \
    --out      "$OUT"

chmod +x "$OUT"

echo
echo "done."
echo "  open in PDF reader:   xdg-open $OUT"
echo "  run from shell:       $OUT \"hello guppy\""
