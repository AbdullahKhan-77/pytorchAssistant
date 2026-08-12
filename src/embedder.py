"""
embedder.py - Loads BAAI/bge-small-en-v1.5 and exposes:
  embed_documents(texts) -> vectors  (no prefix; for indexing chunks)
  embed_query(query)     -> vector   (BGE query prefix; for search)
BGE uses a query/document asymmetry (prefix on queries only); embeddings are
normalised so cosine similarity == dot product. Model loads once on import.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

print(f"Loading embedding model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded.")


def embed_documents(texts):
    """Embed a list of document/chunk texts (NO prefix)."""
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )


def embed_query(query):
    """Embed a single user question (WITH the BGE query prefix)."""
    return model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    )


if __name__ == "__main__":
    v = embed_query("How do I create a linear layer?")
    print("Embedding dimension:", len(v))
    print("First 5 numbers:", v[:5])