"""
Step 1 - Dataset A: extract the Chapter 61/62 tariff table.

Reads the FBR Pakistan Customs Tariff text dump and writes a structured CSV:
code, level, description, duty_cd, parent, chapter, heading, tariff_version

This is NOT the RAG corpus. It is the tariff table that the classifier
predicts into and the duty engine reads rates from. The RAG corpus is
built separately in build_corpus.py.

Usage:
    python src/extract_tariff.py
"""

import csv
import re
from pathlib import Path

RAW = Path("data/raw/tariff_2026_27.txt")
OUT = Path("data/processed/tariff_ch61_62.csv")
TARIFF_VERSION = "FY2026-27"

# Chapters we care about. Widen this later if scope grows.
CHAPTERS = {"61", "62"}

# "61.09 T- shirts, singlets and other vests, knitted or crocheted."
HEADING_RE = re.compile(r"^(\d{2})\.(\d{2})\s+(.*)$")
# "6109.1000 - Of cotton 20"  -> code, description, optional duty
CODE_RE = re.compile(r"^(\d{4})\.(\d{4})\s+(.*?)\s*(\d{1,2}(?:\.\d+)?)?\s*$")
# "- - Of cotton" / "- - - Bulls": FBR separates the dashes with spaces, so the
# nesting level is the number of leading dashes, not the length of one run.
DASH_RE = re.compile(r"^\s*((?:-\s*)+)(.*?):?\s*$")


def clean(text: str) -> str:
    """Repair the PDF text layer.

    FBR's export leaves soft-hyphen control bytes inside words ('wind\\x02jackets')
    and stray spaces after hyphens ('T- shirts', 'man- made').
    """
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f­​]", "", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1-\2", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().rstrip(".").strip()


def dash_depth(text: str) -> int:
    """Nesting level of a tariff line: '- - Of cotton' is depth 2."""
    m = DASH_RE.match(text)
    return m.group(1).count("-") if m else 0


DUTY_ONLY_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d+)?)\s*$")


def parse(lines):
    """Walk the file, keep only Ch. 61/62, and rebuild the parent-child tree."""
    rows = []
    in_scope = False
    current_heading = None
    # dash depth -> the label text at that depth, so a code can inherit context
    context = {}
    buffer = ""
    # The last tariff-line row, so a wrapped description or an orphan duty
    # value on the following line can be folded back into it.
    pending = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        # A code's description sometimes wraps, pushing the duty onto its own
        # line. Fold those continuation lines back into the row above.
        if pending is not None:
            stripped = line.strip()
            d_only = DUTY_ONLY_RE.match(stripped)
            if d_only:
                pending["duty_cd"] = d_only.group(1)
                pending = None
                continue
            is_new_record = (
                CODE_RE.match(stripped)
                or HEADING_RE.match(stripped)
                or dash_depth(stripped)
            )
            if not is_new_record:
                extra = clean(stripped)
                pending["description"] = clean(pending["description"] + " " + extra)
                pending["full_description"] = clean(
                    pending["full_description"] + " " + extra
                )
                continue
            pending = None

        # Headings wrap across 2-3 lines in the PDF dump. Join them first.
        candidate = (buffer + " " + line).strip() if buffer else line.strip()

        m = HEADING_RE.match(candidate)
        if m:
            ch = m.group(1)
            # A heading line is complete when it ends with a full stop.
            if not candidate.rstrip().endswith("."):
                buffer = candidate
                continue
            buffer = ""
            in_scope = ch in CHAPTERS
            if in_scope:
                current_heading = f"{ch}{m.group(2)}"
                context = {}
                rows.append(
                    {
                        "code": current_heading,
                        "level": "heading",
                        "description": clean(m.group(3)),
                        "duty_cd": "",
                        "parent": ch,
                        "chapter": ch,
                        "heading": current_heading,
                        "tariff_version": TARIFF_VERSION,
                    }
                )
            continue

        buffer = ""
        if not in_scope:
            continue

        m = CODE_RE.match(line.strip())
        if m:
            code = f"{m.group(1)}.{m.group(2)}"
            chapter = m.group(1)[:2]
            if chapter not in CHAPTERS:
                in_scope = False
                continue
            # Some headings (e.g. 61.13) have a single tariff line and FBR
            # prints no separate heading row, so the dash context from the
            # previous heading must be dropped here.
            if m.group(1) != current_heading:
                current_heading = m.group(1)
                context = {}
            desc = clean(m.group(3))
            duty = m.group(4) or ""
            # Prepend the dash labels this code sits under, so the description
            # reads on its own: "Trousers... > Of cotton" instead of "Of cotton".
            # A code's own dash depth decides which labels apply: a "- - Of
            # cotton" line sits under depth-1 labels only, never under a
            # sibling "- - Of wool" label at its own depth.
            own_depth = dash_depth(desc) or 99
            parents = [
                context[d] for d in sorted(context) if d < own_depth and context[d]
            ]
            leaf = DASH_RE.match(desc).group(2).strip() if dash_depth(desc) else desc
            full_desc = " > ".join(parents + [leaf])
            row = {
                "code": code,
                "level": "tariff_line",
                "description": desc,
                "full_description": full_desc,
                "duty_cd": duty,
                "parent": m.group(1),
                "chapter": chapter,
                "heading": m.group(1),
                "tariff_version": TARIFF_VERSION,
            }
            rows.append(row)
            # Only watch for a continuation when the duty is still missing.
            pending = row if not duty else None
            continue

        d = dash_depth(line.strip())
        if d:
            label = clean(DASH_RE.match(line.strip()).group(2))
            context[d] = label
            # A new label at depth d invalidates anything deeper.
            for deeper in [k for k in context if k > d]:
                context.pop(deeper)

    return rows


def main():
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}. Put the FBR tariff text dump there.")

    rows = parse(RAW.read_text(encoding="utf-8", errors="ignore").splitlines())
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "code",
        "level",
        "description",
        "full_description",
        "duty_cd",
        "parent",
        "chapter",
        "heading",
        "tariff_version",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r.setdefault("full_description", r["description"])
            w.writerow(r)

    headings = [r for r in rows if r["level"] == "heading"]
    lines_ = [r for r in rows if r["level"] == "tariff_line"]
    ch61 = [r for r in lines_ if r["chapter"] == "61"]
    ch62 = [r for r in lines_ if r["chapter"] == "62"]
    no_duty = [r for r in lines_ if not r["duty_cd"]]

    print(f"Wrote {OUT}")
    print(f"  headings      : {len(headings)}")
    print(f"  tariff lines  : {len(lines_)}  (ch61={len(ch61)}, ch62={len(ch62)})")
    print(f"  missing duty  : {len(no_duty)}")
    if no_duty:
        print("  -> check these by hand against the PDF:")
        for r in no_duty[:5]:
            print(f"     {r['code']}  {r['description'][:60]}")


if __name__ == "__main__":
    main()
