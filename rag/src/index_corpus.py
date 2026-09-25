"""
Step 3: embed the corpus and load it into ChromaDB.

Uses a multilingual embedding model so Roman Urdu queries ("sooti knitted
shirt") land near the English tariff text.

Usage:
    python src/index_corpus.py
    python src/index_corpus.py --reset     # rebuild from scratch
"""

import argparse
import json
from pathlib import Path

CORPUS = Path("data/processed/corpus.jsonl")
DB_DIR = Path("data/processed/chroma")
COLLECTION = "pct_ch61_62"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

METADATA_FIELDS = [
    "chunk_type",
    "level",
    "chapter",
    "headings",
    "code",
    "source",
    "note_type",
]


def load_corpus():
    if not CORPUS.exists():
        raise SystemExit(f"Missing {CORPUS}. Run src/build_corpus.py first.")
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete and rebuild")
    args = ap.parse_args()

    import chromadb
    from chromadb.utils import embedding_functions

    chunks = load_corpus()
    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    if args.reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"Deleted existing collection '{COLLECTION}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{k: c.get(k, "") for k in METADATA_FIELDS} for c in chunks],
    )

    print(f"Indexed {len(chunks)} chunks into '{COLLECTION}' at {DB_DIR}")
    print(f"Collection now holds {collection.count()} documents.")


if __name__ == "__main__":
    main()
