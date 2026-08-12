# PyTorch Documentation Assistant (RAG)

A Retrieval-Augmented Generation (RAG) assistant that answers PyTorch questions
using **only** the official library documentation, and **cites its sources**.

Documentation is read directly from the installed `torch` package's docstrings
(the authoritative source the docs site is built from), embedded with a local
sentence-transformer, stored in Chroma, retrieved with a two-stage
(bi-encoder + cross-encoder) retriever, and answered by a locally hosted,
4-bit quantized **Qwen2.5-7B-Instruct** model, exposed through a Gradio UI.

## Architecture

**Indexing (run once):**
`docstrings → ingest → chunk → embed → Chroma`

**Querying (per question):**
`question → embed → retrieve (top-20) → rerank (top-8) → LLM (grounded) → answer + citations`

## Stack

| Component        | Choice                                | Reason (short)                                   |
|------------------|---------------------------------------|--------------------------------------------------|
| Document source  | Package docstrings (via `inspect`)    | Authoritative; no scraping; matches installed ver|
| Embedding model  | `BAAI/bge-small-en-v1.5`              | Beats MiniLM on retrieval; small, offline        |
| Vector DB        | Chroma (cosine / HNSW)               | Zero-setup; stores text+metadata+vectors together|
| Retrieval        | Bi-encoder + cross-encoder rerank    | Precision on a large technical corpus            |
| Chunking         | 500 tokens, 50 overlap, headers      | Fits BGE's 512 limit; keeps sections intact      |
| LLM              | Qwen2.5-7B-Instruct (4-bit)          | Strong grounded QA; Apache-2.0; fits a 16GB GPU  |
| UI               | Gradio                               | One-line public share link on Colab              |

See `report/` for the full report with justifications and findings.

## Where things run

- **Local (CPU is fine):** `ingest.py`, `chunker.py`, `embedder.py`, `build_index.py`, `retriever.py`
- **GPU required (use Google Colab T4):** `generate.py` (loads the 7B LLM) and `app.py`

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

**1. Build the index** (local or Colab):
```bash
python src/ingest.py        # -> pytorch_docs_raw.json
python src/chunker.py       # -> pytorch_chunks.json
python src/build_index.py   # -> chroma_db/
```

**2. Test retrieval** (local, no GPU needed):
```bash
python src/retriever.py
```

**3. Run the assistant** (on Colab with a GPU):
- Set the runtime to a T4 GPU (`Runtime → Change runtime type → T4 GPU`).
- Make sure `chroma_db/`, `src/`, and `requirements.txt` are present (rebuild the
  index on Colab with step 1, or upload the folder).
- Launch the UI:
```bash
python src/app.py
```
This prints a public `gradio.live` link.

## Notes
- Generated data (`chroma_db/`, JSON files) is **git-ignored** — it is regenerable
  by running the pipeline above.
- The corpus covers all **public** `torch` modules (auto-discovered; private/
  internal underscore modules are skipped).