"""
Step 4: retrieval. This is what the /evidence endpoint will call.

Two entry points:

  search(query, ...)              - plain semantic search over the corpus.
  evidence_for_code(code, ...)    - the one the product actually uses: given a
                                    code the classifier predicted, fetch the
                                    text that supports it.

Retrieval is metadata-filtered by chapter and heading, because a note about
Chapter 61 is never evidence for a Chapter 62 code. GRIs and section notes
apply to every code, so they are always allowed through.

Usage:
    python src/retrieve.py "men's cotton knitted t-shirt"
    python src/retrieve.py --code 6109.1000
"""

import argparse
import json
from pathlib import Path

DB_DIR = Path("data/processed/chroma")
COLLECTION = "pct_ch61_62"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=str(DB_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
        _collection = client.get_collection(COLLECTION, embedding_function=embed_fn)
    return _collection


def _format(results):
    """Chroma returns parallel lists; turn them into one list of dicts."""
    out = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append(
            {
                "id": i,
                "text": doc,
                "score": round(1 - dist, 4),  # cosine distance -> similarity
                **meta,
            }
        )
    return out


def search(query, k=5, chapter=None, chunk_type=None):
    """Plain semantic search, optionally narrowed by chapter or chunk type."""
    clauses = []
    if chapter:
        clauses.append({"chapter": {"$eq": str(chapter)}})
    if chunk_type:
        clauses.append({"chunk_type": {"$eq": chunk_type}})

    where = None
    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}

    res = get_collection().query(query_texts=[query], n_results=k, where=where)
    return _format(res)


def evidence_for_code(code, description="", k=6):
    """Evidence supporting one predicted code.

    Returns the tariff text for the code and its heading, plus the notes that
    could apply to it: notes for its own chapter, and chapter-agnostic rules
    (GRIs, Section XI notes) that apply to everything.
    """
    code = str(code)
    chapter = code[:2]
    heading = code[:4]
    query = description or f"heading {heading} {code}"

    coll = get_collection()
    hits = []

    # 1. The code's own tariff text and its heading: exact lookups, not search.
    exact_ids = [f"TARIFF-L-{code}", f"TARIFF-H-{heading}"]
    got = coll.get(ids=exact_ids)
    for i, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        hits.append({"id": i, "text": doc, "score": 1.0, "why": "exact", **meta})

    # 2. Notes that could bear on this code.
    note_where = {
        "$and": [
            {"chunk_type": {"$eq": "legal_note"}},
            {"$or": [{"chapter": {"$eq": chapter}}, {"chapter": {"$eq": ""}}]},
        ]
    }
    res = coll.query(query_texts=[query], n_results=k, where=note_where)
    for h in _format(res):
        h["why"] = "note"
        hits.append(h)

    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="free-text query")
    ap.add_argument("--code", help="fetch evidence for a predicted code")
    ap.add_argument("--chapter", help="restrict to chapter 61 or 62")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    if args.code:
        hits = evidence_for_code(args.code, description=args.query or "", k=args.k)
    elif args.query:
        hits = search(args.query, k=args.k, chapter=args.chapter)
    else:
        ap.error("give a query or --code")

    if not hits:
        print("No supporting text found.")
        return

    for h in hits:
        tag = h.get("why", h.get("chunk_type", ""))
        print(f"\n[{h['score']:.3f}] {h['id']}  ({tag}, {h.get('level','')})")
        print(f"  {h['text'][:300]}")
        print(f"  source: {h.get('source','')}")


if __name__ == "__main__":
    main()
