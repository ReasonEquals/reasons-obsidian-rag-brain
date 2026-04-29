#!/usr/bin/env python3
"""
Semantic search over the vault ChromaDB index.

Embeds a natural language query via Voyage AI, finds the most similar
chunks in ChromaDB, synthesizes an answer via Claude, and prints results.

Usage:
    python brain/search.py "what chunking strategy did I decide on?"
    python brain/search.py "embedding model tradeoffs" --top 3
    python brain/search.py "vector store options" --no-synth

Env vars (via .env):
    VOYAGE_API_KEY    Voyage AI API key (required)
    ANTHROPIC_API_KEY Anthropic API key (required for synthesis)
"""

import argparse
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


try:
    import chromadb
except ImportError:
    print("chromadb not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# Reuse the embedding function and config from index.py
from index import (
    CHROMA_DIR,
    COLLECTION_NAME,
    embed_texts,
    ensure_search_tables,
    get_db_conn,
    log_search,
)

# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(query: str, chunks: list[tuple[str, str, str]]) -> str:
    """Call Claude Haiku to synthesize an answer from retrieved chunks.

    chunks: list of (text, source, section) tuples
    """
    import os

    try:
        import anthropic
    except ImportError:
        return "(synthesis unavailable — run: pip install anthropic)"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "(synthesis unavailable — set ANTHROPIC_API_KEY in .env)"

    context_parts = []
    for text, source, section in chunks:
        context_parts.append(f"[{source} § {section}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Answer the question using only the vault excerpts below. "
                    f"Be concise. If the excerpts don't contain a clear answer, say so.\n\n"
                    f"Question: {query}\n\n"
                    f"Excerpts:\n{context}"
                ),
            }
        ],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_vault(query: str, top_k: int = 5, full: bool = False, synth: bool = True) -> None:
    """Embed a query and retrieve the most similar vault chunks.

    The pipeline:
    1. Embed the query with input_type="query" (not "document")
       — the model is trained to put queries near matching documents
    2. ChromaDB finds the top-k nearest neighbors by cosine similarity
    3. Format and print results with source, section, score, and text

    Cosine similarity refresher:
    - 1.0 = identical direction (perfect match)
    - 0.0 = orthogonal (completely unrelated)
    - ChromaDB returns *distance* (1 - similarity), so lower = better

    In practice, for Voyage embeddings:
    - Distance < 0.3 = strong match
    - Distance 0.3-0.5 = related
    - Distance > 0.5 = weak / probably irrelevant
    """
    # Guard: empty queries waste an API call and return meaningless results
    if not query.strip():
        print("ERROR: Empty query. Usage: python brain/search.py \"your question here\"")
        sys.exit(1)

    # Connect to existing ChromaDB
    if not CHROMA_DIR.exists():
        print("No index found. Run `python brain/index.py` first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("Collection not found. Run `python brain/index.py` first.")
        sys.exit(1)

    total_chunks = collection.count()
    if total_chunks == 0:
        print("Index is empty. Run `python brain/index.py` first.")
        sys.exit(1)

    # Don't request more results than exist
    n_results = min(top_k, total_chunks)

    # Embed the query — note input_type="query", not "document"
    query_embedding = embed_texts([query], input_type="query")[0]

    # Find nearest neighbors
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Print results
    print(f"\n  Query: \"{query}\"")
    print(f"  Searching {total_chunks} chunks...\n")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    # Track for logging
    result_ids = []
    scores = []

    # Check if all results are weak
    all_weak = all(d > 0.5 for d in distances)

    # Synthesize answer from retrieved chunks
    if synth and documents:
        chunks = [
            (doc, meta.get("source", "?"), meta.get("section", "?"))
            for doc, meta in zip(documents, metadatas, strict=False)
        ]
        answer = synthesize(query, chunks)
        print(f"  {answer}\n")
        print("  " + "─" * 60 + "\n")

    for i, (doc, meta, dist, chunk_id) in enumerate(zip(documents, metadatas, distances, ids, strict=False), 1):
        similarity = 1 - dist  # convert distance to similarity
        source = meta.get("source", "?")
        section = meta.get("section", "?")
        tags = meta.get("tags", "")
        status = meta.get("status", "")

        result_ids.append(chunk_id)
        scores.append(similarity)

        # Header line
        print(f"  {i}. {source} § {section}")

        # Metadata line
        meta_parts = [f"score: {similarity:.3f}"]
        if status:
            meta_parts.append(f"status: {status}")
        if tags:
            meta_parts.append(f"tags: {tags}")
        print(f"     {' | '.join(meta_parts)}")

        # Content
        if full:
            # Show full chunk text, indented
            for line in doc.split("\n"):
                print(f"     {line}")
        else:
            # Truncated preview — first 200 chars
            preview = doc[:200].replace("\n", " ")
            if len(doc) > 200:
                preview += "..."
            print(f"     {preview}")

        print()

    if all_weak:
        print("  ⚠ All results scored below 0.5 — nothing strongly matched.")
        print("    Try rephrasing, or the topic may not be in the vault.\n")

    # Log the search to SQLite
    try:
        db_conn = get_db_conn()
        ensure_search_tables(db_conn)
        log_search(db_conn, query, top_k, result_ids, scores)
        db_conn.close()
    except Exception:  # noqa: S110
        pass  # logging failure shouldn't break search


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic search over your vault."
    )
    parser.add_argument(
        "query",
        help="Natural language search query",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full chunk text instead of truncated preview",
    )
    parser.add_argument(
        "--no-synth",
        action="store_true",
        help="Skip Claude synthesis, return raw chunks only",
    )
    args = parser.parse_args()
    search_vault(args.query, top_k=args.top, full=args.full, synth=not args.no_synth)
