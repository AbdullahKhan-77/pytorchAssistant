"""
chunker.py - Stage 2: split docstrings into ~500-token, structure-aware chunks.
- Tokens counted with the embedding model's own tokenizer (matches what it sees).
- Target 500 (BGE limit is 512), 50-token overlap (~10%).
- Split on paragraph boundaries; short docstrings pass through whole.
- Long docstrings are split, and each piece gets the SIGNATURE prepended as a
  header (contextual chunking) so example-only chunks still identify their function.
Run:  python src/chunker.py
"""

import json

from embedder import model

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
HEADER_BUFFER = 8

tokenizer = model.tokenizer


def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_header(text):
    return text.split("\n")[0].strip()


def split_long_text(text, chunk_size, overlap, header=""):
    header_tokens = count_tokens(header) if header else 0
    effective_size = chunk_size - header_tokens - HEADER_BUFFER

    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_tokens = 0

    def finalize(parts):
        body = "\n\n".join(parts)
        if header and not body.startswith(header):
            return f"{header}\n\n{body}"
        return body

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > effective_size:
            if current:
                chunks.append(finalize(current))
                current = []
                current_tokens = 0
            token_ids = tokenizer.encode(para, add_special_tokens=False)
            step = effective_size - overlap
            for i in range(0, len(token_ids), step):
                piece_ids = token_ids[i:i + effective_size]
                chunks.append(finalize([tokenizer.decode(piece_ids)]))
            continue

        if current_tokens + para_tokens > effective_size:
            chunks.append(finalize(current))
            current = []
            current_tokens = 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append(finalize(current))
    return chunks


def chunk_documents(docs):
    chunked = []
    for doc in docs:
        text = doc["text"]
        if count_tokens(text) <= CHUNK_SIZE:
            chunked.append({
                "id": doc["id"],
                "source": doc["source"],
                "text": text,
            })
        else:
            header = make_header(text)
            pieces = split_long_text(text, CHUNK_SIZE, CHUNK_OVERLAP, header)
            for i, piece in enumerate(pieces):
                chunked.append({
                    "id": f"{doc['id']}__chunk{i}",
                    "source": doc["source"],
                    "text": piece,
                })
    return chunked


def main():
    with open("pytorch_docs_raw.json", encoding="utf-8") as f:
        docs = json.load(f)

    chunks = chunk_documents(docs)
    token_counts = [count_tokens(c["text"]) for c in chunks]
    print(f"Raw docs:   {len(docs)}")
    print(f"Chunks:     {len(chunks)}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"Avg tokens: {sum(token_counts) // len(token_counts)}")
    print(f"Split-derived chunks: {len([c for c in chunks if '__chunk' in c['id']])}")

    with open("pytorch_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"\nSaved {len(chunks)} chunks to pytorch_chunks.json")


if __name__ == "__main__":
    main()