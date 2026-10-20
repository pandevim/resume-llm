"""Map the words of dist/resume.pdf to character ranges of pdfllm/resume.txt (what the model reads).

  python pdfllm/tools/word_map.py        # needs PyMuPDF (pip install pymupdf); run by build.sh

Writes pdfllm/resume_words.json:
  {"pages": [[w, h], ...],                          page sizes in PDF points
   "words": [[page, x0, y0, x1, y1, start, end]]}   boxes in points (origin top-left), text range in resume.txt

The two texts differ (resume.txt is hand-cleaned), so both are reduced to lowercase letters and digits
and aligned with difflib; a PDF word gets the resume.txt range of its aligned characters. PDF words with
no aligned characters are left out. The engine uses the ranges to turn copy-head attention over prompt
tokens into weights on PDF words; the site uses the boxes to draw them.
"""

import difflib
import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
FOLD = {"’": "'", "‘": "'", "∼": "~", "–": "-", "—": "-"}


def keep(text):
    """(folded chars, index of each in text), letters and digits only."""
    out, idx = [], []
    for i, ch in enumerate(text):
        ch = FOLD.get(ch, ch).lower()
        if ch.isalnum():
            out.append(ch), idx.append(i)
    return "".join(out), idx


def main():
    resume = (ROOT / "pdfllm/resume.txt").read_text().strip()  # engine.py strips it the same way
    doc = fitz.open(ROOT / "dist/resume.pdf")
    words, pdf_chars, pdf_word = [], [], []
    for pno, page in enumerate(doc):
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            chars, _ = keep(text)
            words.append([pno, round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])
            pdf_chars.append(chars)
            pdf_word += [len(words) - 1] * len(chars)
    a = "".join(pdf_chars)
    b, b_idx = keep(resume)
    span = {}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for blk in sm.get_matching_blocks():
        if blk.size < 3:  # stray one- or two-letter matches are noise
            continue
        for k in range(blk.size):
            w, r = pdf_word[blk.a + k], b_idx[blk.b + k]
            lo, hi = span.get(w, (r, r + 1))
            span[w] = (min(lo, r), max(hi, r + 1))
    # Second pass: dates sit in their own column in the PDF, out of resume.txt's order, so the in-order
    # alignment skips them. Place each leftover word at its nearest free occurrence in the resume,
    # near where the rest of its line (or the word before it) was matched.
    claimed = [False] * len(resume)
    for lo, hi in span.values():
        claimed[lo:hi] = [True] * (hi - lo)
    for w, chars in enumerate(pdf_chars):
        if w in span or len(chars) < 2:
            continue
        line = [span[v][1] for v in span if words[v][0] == words[w][0] and abs(words[v][2] - words[w][2]) < 3]
        prev = [span[v][1] for v in range(w) if v in span]  # a date on its own line: the word before it
        if not line and not prev:
            continue
        anchor, best = max(line) if line else prev[-1], None
        for k in range(len(b) - len(chars) + 1):
            if b[k : k + len(chars)] != chars:
                continue
            lo, hi = b_idx[k], b_idx[k + len(chars) - 1] + 1
            whole = (lo == 0 or not resume[lo - 1].isalnum()) and (hi == len(resume) or not resume[hi].isalnum())
            if whole and not any(claimed[lo:hi]) and abs(lo - anchor) < 150:
                if best is None or abs(lo - anchor) < abs(best[0] - anchor):
                    best = (lo, hi)
        if best:
            span[w] = best
            claimed[best[0] : best[1]] = [True] * (best[1] - best[0])
    out =[words[w] + list(span[w]) for w in sorted(span)]
    pages = [[round(p.rect.width, 1), round(p.rect.height, 1)] for p in doc]
    (ROOT / "pdfllm/resume_words.json").write_text(json.dumps({"pages": pages, "words": out}, separators=(",", ":")))
    print(f"{len(out)}/{len(words)} PDF words mapped; {sum(e - s for *_, s, e in out)}/{len(resume)} resume chars covered")


if __name__ == "__main__":
    main()
