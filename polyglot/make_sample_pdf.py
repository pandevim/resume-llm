#!/usr/bin/env python3
"""
make_sample_pdf.py — Emit a tiny single-page PDF for the worked example.

Pure-Python, no dependencies. Produces examples/sample.pdf with a few
paragraphs about a fictional fish so phase-2 QA has something extractive
to point at.
"""

import argparse
import zlib
from pathlib import Path


SAMPLE_TEXT = [
    "About Guppy",
    "",
    "Guppy is a small freshwater fish that lives in a glass tank.",
    "Guppy is fed twice per day, in the morning and at sunset.",
    "Guppy enjoys bubbles, warm water, and the occasional algae wafer.",
    "Guppy does not understand money, phones, or politics.",
    "",
    "When Guppy is hungry, Guppy swims toward the surface.",
    "When Guppy is scared, Guppy hides behind the largest plant.",
    "Guppy's favorite color is the green of the back wall of the tank.",
]


def encode_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_content_stream(lines):
    out = "BT\n/F1 12 Tf\n72 740 Td\n14 TL\n"
    for i, line in enumerate(lines):
        if i == 0:
            out += f"({encode_string(line)}) Tj\n"
        else:
            out += f"T*\n({encode_string(line)}) Tj\n"
    out += "ET\n"
    return out.encode("utf-8")


def build_pdf(lines):
    content = build_content_stream(lines)
    compressed = zlib.compress(content, 9)

    objects = {}
    objects[4] = (
        b"<< /Length " + str(len(compressed)).encode()
        + b" /Filter /FlateDecode >>\n"
        b"stream\n" + compressed + b"\nendstream"
    )
    objects[3] = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica >> >> >> >>"
    )
    objects[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    body = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for n in sorted(objects):
        offsets[n] = len(body)
        body += f"{n} 0 obj\n".encode()
        body += objects[n]
        body += b"\nendobj\n"

    xref = len(body)
    n_objs = max(objects) + 1
    body += f"xref\n0 {n_objs}\n".encode()
    body += b"0000000000 65535 f \n"
    for n in range(1, n_objs):
        body += f"{offsets[n]:010d} 00000 n \n".encode()
    body += b"trailer\n"
    body += f"<< /Size {n_objs} /Root 1 0 R >>\n".encode()
    body += f"startxref\n{xref}\n".encode()
    body += b"%%EOF\n"
    return bytes(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/sample.pdf", type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(build_pdf(SAMPLE_TEXT))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
