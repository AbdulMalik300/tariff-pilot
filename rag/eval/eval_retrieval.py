"""
Step 5: measure retrieval quality.

Runs the test queries two ways and reports hit@k for each:

  blind      - plain semantic search over the whole corpus, no filter. This is
               what retrieval alone can do, with no help from the classifier.
  filtered   - the same queries with the chapter the classifier would have
               predicted, so Chapter 61 chunks cannot answer a Chapter 62
               question. This is the path the product actually uses.

The gap between the two numbers is the argument for the architecture: semantic
similarity cannot tell "knitted" from "woven", because the two read almost
identically, and that single distinction is what separates Chapter 61 from
Chapter 62. The classifier decides; retrieval then proves the decision.

Usage:
    python eval/eval_retrieval.py
    python eval/eval_retrieval.py -k 3
    python eval/eval_retrieval.py --verbose
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieve import search  # noqa: E402

QUERIES = Path(__file__).parent / "retrieval_queries.json"


def run(cases, k, use_chapter, verbose):
    hits, failures = 0, []
    for case in cases:
        chapter = case.get("chapter") or None if use_chapter else None
        results = search(case["query"], k=k, chapter=chapter)
        got = [r["id"] for r in results]
        ok = any(e in got for e in case["expect_ids"])
        if ok:
            hits += 1
        else:
            failures.append((case, got))
        if verbose or not ok:
            print(f"[{'PASS' if ok else 'FAIL'}] {case['query']}")
            if not ok:
                print(f"        expected : {case['expect_ids']}")
                print(f"        got      : {got[:k]}")
    return hits, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = json.loads(QUERIES.read_text(encoding="utf-8"))
    total = len(cases)

    print("=" * 62)
    print("BLIND: semantic search only, no chapter filter")
    print("=" * 62)
    blind_hits, blind_fail = run(cases, args.k, False, args.verbose)
    print(f"\nhit@{args.k} (blind)    : {blind_hits}/{total} = {blind_hits / total:.1%}")

    print("\n" + "=" * 62)
    print("FILTERED: chapter supplied by the classifier (the product path)")
    print("=" * 62)
    filt_hits, filt_fail = run(cases, args.k, True, args.verbose)
    print(f"\nhit@{args.k} (filtered) : {filt_hits}/{total} = {filt_hits / total:.1%}")

    gain = filt_hits - blind_hits
    print("\n" + "-" * 62)
    print(f"Chapter filtering recovers {gain} queries ({gain / total:+.1%}).")

    fixed = {c["query"] for c, _ in blind_fail} - {c["query"] for c, _ in filt_fail}
    if fixed:
        print("Recovered by knowing the chapter:")
        for q in sorted(fixed):
            print(f"  - {q}")

    still = [c for c, _ in filt_fail]
    if still:
        print("\nStill failing even with the chapter known:")
        for c in still:
            print(f"  - {c['query']}  (wanted {c['expect_ids']})")
        print(
            "\nThese are the real retrieval weaknesses. Note them in the report\n"
            "rather than tuning the eval set until they pass."
        )


if __name__ == "__main__":
    main()