"""
Helper: pull candidate Chapter/Section Note text out of the FBR tariff PDF.

The text dump in data/raw/ contains only the code table - no notes. The notes
live in the full FBR PDF, on the page before each chapter's tariff lines and
in the Section XI preamble. This script finds those pages and dumps the text
so you can paste it into data/legal/notes_ch61_62.json.

It does NOT write the JSON for you on purpose. Note text is legal evidence
shown to the user, so a human reads it against the PDF before it ships.

Usage:
    python src/extract_notes.py path/to/FBR_tariff.pdf
    python src/extract_notes.py path/to/FBR_tariff.pdf --pages 812-820
"""

import argparse
import re
from pathlib import Path

# Phrases that mark the start of the note blocks we need.
MARKERS = [
    (r"this chapter applies only to made up knitted", "Chapter 61 Note 1"),
    (r"this chapter applies only to made up articles", "Chapter 62 Note 1"),
    (r"predominates by weight", "Section XI Note 2 (blends)"),
    (r"cannot be identified as either men's", "Gender default note"),
    (r"stitches per linear cent", "Chapter 61 Note 4 (stitch density)"),
    (r"body height not exceeding", "Babies' garments note"),
    (r"drawstring, ribbed waistband", "Chapter 61 Note 5 (6109)"),
    (r"classification of goods shall be determined", "GRI 1"),
    (r"most specific description", "GRI 3(a)"),
    (r"essential character", "GRI 3(b)"),
    (r"last in numerical order", "GRI 3(c)"),
]


def parse_pages(spec):
    if not spec:
        return None
    if "-" in spec:
        a, b = spec.split("-", 1)
        return range(int(a) - 1, int(b))
    return [int(spec) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="the full FBR Pakistan Customs Tariff PDF")
    ap.add_argument("--pages", help="1-based page range, e.g. 812-820")
    ap.add_argument("--out", default="data/legal/notes_raw.txt")
    args = ap.parse_args()

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise SystemExit("pip install pymupdf")

    doc = fitz.open(args.pdf)
    pages = parse_pages(args.pages) or range(len(doc))

    found = []
    dump = []
    for pno in pages:
        if pno >= len(doc):
            continue
        text = doc[pno].get_text()
        low = text.lower()
        hits = [label for pat, label in MARKERS if re.search(pat, low)]
        if hits:
            found.append((pno + 1, hits))
            dump.append(f"\n{'=' * 70}\nPAGE {pno + 1}  -> {', '.join(hits)}\n{'=' * 70}\n{text}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(dump), encoding="utf-8")

    if not found:
        print("No note markers found.")
        print("Either this PDF has no notes layer, or the wording differs.")
        print("Try --pages around the Chapter 61 tariff lines and read by hand.")
        return

    print(f"Pages containing note text (dumped to {out}):")
    for pno, hits in found:
        print(f"  p.{pno}: {', '.join(hits)}")
    print(
        "\nNext: open that file, copy each note VERBATIM into\n"
        "data/legal/notes_ch61_62.json, set status to 'verified', and fill in\n"
        "verified_by and verified_date."
    )


if __name__ == "__main__":
    main()
