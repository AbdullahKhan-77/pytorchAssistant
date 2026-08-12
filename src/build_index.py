"""
build_index.py - Stage 3: embed all chunks and store them in Chroma.
Stores vector + text + source metadata together, under cosine/HNSW.
Persists to ./chroma_db so embedding is a one-time cost.
Run:  python src/build_index.py
"""

import json

import chromadb

from embedder import embed_documents

CHUNKS_FILE = "pytorch_chunks.json"
DB_DIR = "chroma_db"
COLLECTION_NAME = "pytorch_docs"
ADD_BATCH = 2000   # Chroma limits how many items you can add per call


def main():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection (starting fresh)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    print("Embedding all chunks...")
    embeddings = embed_documents(texts)

    for i in range(0, len(chunks), ADD_BATCH):
        j = min(i + ADD_BATCH, len(chunks))
        collection.add(
            ids=[c["id"] for c in chunks[i:j]],
            documents=texts[i:j],
            embeddings=embeddings[i:j].tolist(),
            metadatas=[{"source": c["source"]} for c in chunks[i:j]],
        )
        print(f"Added {j} / {len(chunks)}")

    print(f"\nDone. Collection holds {collection.count()} chunks.")
    print(f"Database saved to ./{DB_DIR}/")


if __name__ == "__main__":
    main()