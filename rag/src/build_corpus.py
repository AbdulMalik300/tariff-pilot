"""
Step 2 - Dataset C: build the RAG corpus.

Two kinds of chunk go into the corpus:

  1. tariff_text  - the heading and tariff-line descriptions from Dataset A.
                    These are legal text and can be cited as evidence.
  2. legal_note   - GRIs, Section XI Notes, Chapter 61/62 Notes, typed into
                    data/legal/notes_ch61_62.json from the FBR PDF.

One chunk per legal unit (one note, one heading). Notes are not split by
character count: a note is the smallest citable thing, and cutting it in
half produces evidence that is wrong in the user's face.

Usage:
    python src/build_corpus.py
"""

import csv
import json
from pathlib import Path

TARIFF = Path("data/processed/tariff_ch61_62.csv")
NOTES = Path("data/legal/notes_ch61_62.json")
OUT = Path("data/processed/corpus.jsonl")


def load_tariff_chunks():
    chunks = []
    with TARIFF.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["level"] == "heading":
                text = f"Heading {row['code']}: {row['description']}."
                chunk_id = f"TARIFF-H-{row['code']}"
            else:
                text = (
                    f"Tariff line {row['code']} (under heading {row['heading']}): "
                    f"{row['full_description']}. Customs duty {row['duty_cd']}%."
                )
                chunk_id = f"TARIFF-L-{row['code']}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "chunk_type": "tariff_text",
                    "level": "FBR-specific",
                    "chapter": row["chapter"],
                    "headings": row["heading"],
                    "code": row["code"],
                    "source": f"Pakistan Customs Tariff {row['tariff_version']}, First Schedule",
                    "note_type": "tariff_description",
                }
            )
    return chunks


def load_note_chunks():
    if not NOTES.exists():
        print(f"WARNING: {NOTES} not found - corpus will have no legal notes.")
        return []

    raw = json.loads(NOTES.read_text(encoding="utf-8"))
    chunks, skipped = [], 0
    for note in raw:
        text = (note.get("text") or "").strip()
        status = note.get("status", "")
        if not text or status in {"TODO", "TEMPLATE_DELETE_ME"}:  # unverified_* still loads
            skipped += 1
            continue
        note_type = note.get("note_type", "note")
        label = "GRI" if note_type == "gri" else note_type.replace("_", " ").title()
        num = note.get("note_number", "")
        ch = note.get("chapter", "")
        prefix = f"{label} {num}".strip()
        if ch:
            prefix = f"Chapter {ch} {prefix}"
        chunks.append(
            {
                "id": note["id"],
                "text": f"{prefix}: {text}",
                "chunk_type": "legal_note",
                "level": note.get("level", "HS-level"),
                "chapter": ch,
                # Chroma metadata values must be scalars, so join the list.
                "headings": ",".join(note.get("headings", [])),
                "code": "",
                "source": note.get("source", ""),
                "note_type": note.get("note_type", ""),
            }
        )

    if skipped:
        print(f"Skipped {skipped} notes still marked TODO or empty.")
    return chunks


def main():
    tariff_chunks = load_tariff_chunks()
    note_chunks = load_note_chunks()
    all_chunks = note_chunks + tariff_chunks

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT}")
    print(f"  legal notes   : {len(note_chunks)}")
    print(f"  tariff text   : {len(tariff_chunks)}")
    print(f"  total chunks  : {len(all_chunks)}")
    if not note_chunks:
        print(
            "\n  The corpus can answer 'which heading covers this' but NOT\n"
            "  'why is this knitted garment excluded from Chapter 62'.\n"
            "  Fill in data/legal/notes_ch61_62.json to fix that."
        )


if __name__ == "__main__":
    main()
