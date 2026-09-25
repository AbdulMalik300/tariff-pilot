# Pakistan Customs Classification Assistant

Takes a plain-language product description and returns the PCT/HS code, the
legal text that supports it, and the duty payable.

Scope for this phase: Chapters 61 and 62 (knitted and woven apparel).

## How the pieces fit

```
description ──► attribute extractor ──► classifier ──► top-3 codes
                                                           │
                                            ┌──────────────┴──────────────┐
                                            ▼                             ▼
                                    RAG evidence layer            duty engine
                                    (why this code)             (what it costs)
                                            │                             │
                                            └──────────────┬──────────────┘
                                                           ▼
                                                    FastAPI + Streamlit
```

The classifier picks the code. The RAG layer only proves it. Keeping those two
jobs separate is deliberate: retrieval cannot reliably tell "knitted" from
"woven", and that single distinction is what separates Chapter 61 from 62.

## Layout

| Folder | What lives there | Owner |
| --- | --- | --- |
| `rag/` | Legal corpus, ChromaDB retrieval, evidence for a predicted code | |
| `classifier/` | Attribute extractor, hierarchical model, clarification loop | |
| `duty/` | Tariff rates, SROs, duty and tax calculation | |
| `api/` | FastAPI endpoints and the Streamlit demo | |

Each folder has its own README and its own `requirements.txt`. Run commands
from inside the folder you are working in, not from the repo root.

## Getting started

```bash
git clone https://github.com/AbdulMalik300/tariff-pilot.git
cd tariff-pilot/rag
pip install -r requirements.txt
python src/build_corpus.py
python src/index_corpus.py
python eval/eval_retrieval.py
```

## Working agreements

- **Branch per piece of work.** Never commit straight to `main`. Open a pull
  request so the other two can see what changed.
- **Never commit client data.** Goods Declarations are stripped of importer and
  exporter names, NTN, invoice values and GD numbers *before* they reach this
  repo. `.gitignore` is a backstop, not the safeguard.
- **Never commit build artifacts.** The ChromaDB index is rebuilt from
  `corpus.jsonl` by anyone who needs it.
- **Record every metric in the results log**, with the date and the commit it
  came from. Numbers quoted in the report must be traceable to a run.

## Status

| Piece | State |
| --- | --- |
| Tariff table (Ch. 61/62) | Done: 33 headings, 252 tariff lines, FY2026-27 |
| Legal corpus | Done: 35 notes, marked unverified pending FBR check |
| RAG retrieval | Done: hit@5 76.2% blind, 90.5% with chapter filter |
| Attribute extractor | Not started |
| Classifier | Not started |
| Duty engine | Not started |
| API and UI | Not started |

## Known gaps

- The legal notes were transcribed from the US HTS because FBR's published
  tariff PDF is table-only. They are HS-level and therefore apply in Pakistan,
  but each one still needs checking against FBR's own First Schedule before the
  pilot.
- Pakistan's own "Pakistan Rules" from the First Schedule preamble, which
  govern the national 8-digit splits, are not in the corpus yet.
