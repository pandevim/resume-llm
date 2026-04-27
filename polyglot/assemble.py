#!/usr/bin/env python3
"""
assemble.py — Merge a static ELF and a rendered PDF into a single polyglot.

Strategy:
  1. The ELF header occupies bytes 0..63. The Linux loader requires this.
  2. PDF readers scan the first 1024 bytes for the literal '%PDF-' marker
     and parse forward from there using the trailer's startxref.
  3. We inject '%PDF-1.4' into a benign NUL run inside the ELF's first 1024
     bytes (outside the program/section header tables).
  4. The rendered PDF is normalized through pypdf (which emits a classical
     xref table and lets us add the .tex source as an attachment), then
     placed AFTER the ELF body. All object offsets in the xref table — and
     the trailing 'startxref' value — are rewritten by adding len(elf), so
     they remain absolute file offsets in the polyglot.
  5. The PDF parser ignores the ELF prefix because it parses from the first
     '%PDF-' marker and follows startxref to the xref. The ELF loader
     ignores everything past its declared section extents.
"""

import argparse
import io
import os
import re
import struct
from pathlib import Path

import pypdf


PDF_MARKER = b"%PDF-1.4\n"


def find_marker_slot(elf: bytes) -> int:
    """
    Find an offset in the first 1024 bytes of the ELF where we can safely
    write PDF_MARKER without breaking the loader.
    """
    if elf[:4] != b"\x7fELF":
        raise RuntimeError("not an ELF file")
    if elf[4] != 2:
        raise RuntimeError("only ELF64 is supported")

    e_phoff     = struct.unpack_from("<Q", elf, 0x20)[0]
    e_shoff     = struct.unpack_from("<Q", elf, 0x28)[0]
    e_phentsize = struct.unpack_from("<H", elf, 0x36)[0]
    e_phnum     = struct.unpack_from("<H", elf, 0x38)[0]
    e_shentsize = struct.unpack_from("<H", elf, 0x3A)[0]
    e_shnum     = struct.unpack_from("<H", elf, 0x3C)[0]

    phdr_end = e_phoff + e_phnum * e_phentsize
    shdr_end = e_shoff + e_shnum * e_shentsize if e_shoff else 0

    def in_table(off: int, length: int) -> bool:
        if off < 64:
            return True
        if off < phdr_end and off + length > e_phoff:
            return True
        if shdr_end and off < shdr_end and off + length > e_shoff:
            return True
        return False

    need = len(PDF_MARKER)
    off = 64
    while off + need <= 1024:
        if in_table(off, need):
            off += 1
            continue
        if elf[off:off + need] == b"\x00" * need:
            return off
        off += 1

    raise RuntimeError(
        "no safe NUL run in ELF[64:1024] big enough for the PDF marker."
    )


def normalize_pdf_with_attachment(pdf_path: Path, attach_path: Path | None) -> bytes:
    """
    Load `pdf_path` through pypdf (forces a fresh, classical xref table),
    optionally add `attach_path` as an /EmbeddedFile attachment, and return
    the resulting PDF bytes.
    """
    reader = pypdf.PdfReader(str(pdf_path))
    writer = pypdf.PdfWriter(clone_from=reader)

    if attach_path is not None:
        writer.add_attachment(attach_path.name, attach_path.read_bytes())
        # Surface the attachments pane on open.
        try:
            writer._root_object[pypdf.generic.NameObject("/PageMode")] = (
                pypdf.generic.NameObject("/UseAttachments")
            )
        except Exception:
            pass

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


_XREF_LINE = re.compile(rb"^(\d{10}) (\d{5}) ([nf]) $", re.MULTILINE)


def shift_pdf_offsets(pdf: bytes, shift: int) -> bytes:
    """
    Add `shift` to every in-use offset in the xref table and to the
    trailing `startxref` value, so the PDF remains valid when prefixed
    with `shift` bytes of foreign data (the ELF).

    Assumes a classical xref table (one or more `xref` sections terminated
    by `trailer`), as produced by pypdf.PdfWriter.
    """
    # Locate the xref section that the trailer's startxref points to.
    m = re.search(rb"startxref\s+(\d+)\s+%%EOF", pdf)
    if not m:
        raise RuntimeError("no startxref/%%EOF found — unexpected PDF format")
    startxref_old = int(m.group(1))
    startxref_new = startxref_old + shift

    # Walk every xref subsection and rewrite each in-use entry's 10-digit
    # offset. Free entries (the head '0000000000 65535 f') stay as-is —
    # their first field is a pointer into the free-object linked list,
    # not a byte offset, so it must NOT be shifted.
    out = bytearray(pdf)

    # Find each `xref\n` block and process the entries that follow.
    for xref_match in re.finditer(rb"\nxref\n", pdf):
        cursor = xref_match.end()
        while True:
            # Subsection header: "first count\n"
            header_end = pdf.find(b"\n", cursor)
            if header_end == -1:
                break
            header = pdf[cursor:header_end]
            if header.startswith(b"trailer"):
                break
            try:
                first_str, count_str = header.split()
                first = int(first_str)
                count = int(count_str)
            except ValueError:
                break
            cursor = header_end + 1
            for i in range(count):
                # Each entry is exactly 20 bytes: "OOOOOOOOOO GGGGG t \n"
                entry = pdf[cursor:cursor + 20]
                if len(entry) != 20 or entry[18:19] != b" " or entry[19:20] not in (b"\n", b"\r"):
                    break
                obj_num = first + i
                kind = entry[17:18]
                if kind == b"n" and obj_num != 0:
                    off = int(entry[:10]) + shift
                    new_entry = f"{off:010d}".encode() + entry[10:]
                    out[cursor:cursor + 20] = new_entry
                cursor += 20
            # After a subsection's entries, either another subsection
            # header or `trailer` follows.
            peek_end = pdf.find(b"\n", cursor)
            if peek_end == -1:
                break
            if pdf[cursor:peek_end].startswith(b"trailer"):
                break

    # Replace the startxref number. The textual length may grow, but it
    # sits past every offset we just wrote, so nothing earlier shifts.
    out[m.start():m.end()] = (
        f"startxref\n{startxref_new}\n%%EOF".encode()
    )
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf",    required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path,
                    help="Rendered PDF to use as the visible polyglot content")
    ap.add_argument("--attach", type=Path, default=None,
                    help="Additional file to embed as a PDF attachment "
                         "(e.g. the original .tex source)")
    ap.add_argument("--out",    required=True, type=Path)
    args = ap.parse_args()

    elf = args.elf.read_bytes()

    # 1. Patch the ELF so PDF readers find '%PDF-' in the first 1024 bytes.
    slot = find_marker_slot(elf)
    elf = elf[:slot] + PDF_MARKER + elf[slot + len(PDF_MARKER):]

    # 2. Normalize the rendered PDF (forces traditional xref + attaches source).
    pdf_bytes = normalize_pdf_with_attachment(args.source, args.attach)

    # 3. Shift all xref offsets and startxref by len(elf).
    shifted = shift_pdf_offsets(pdf_bytes, shift=len(elf))

    # 4. Concatenate.
    args.out.write_bytes(elf + shifted)
    os.chmod(args.out, 0o755)

    total = args.out.stat().st_size
    print(f"polyglot: {args.out}  ({total/1024/1024:.2f} MB)")
    print(f"  ELF prefix:        {len(elf):>10} bytes")
    print(f"  PDF marker @byte:  {slot}")
    print(f"  PDF body length:   {len(shifted):>10} bytes")


if __name__ == "__main__":
    main()
