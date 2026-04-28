---
title: RAG Architecture Notes
tags: [rag, architecture, ai, retrieval]
summary: Reference notes on RAG design decisions — chunking, retrieval, reranking, context design.
---

# RAG Architecture Notes

Running notes on RAG system design, collected while building the semantic search system. These are opinionated takes based on what actually worked, not a neutral survey.

## Chunking strategies

The chunking decision has more impact on retrieval quality than most people expect. Three main approaches:

**Fixed-token windows** — split every N tokens (128, 256, 512), optionally with overlap. Easy to implement, easy to reason about computationally. The problem: they're semantically blind. A 256-token window will happily split a paragraph in the middle, producing two chunks that each capture half of a complete thought. The embeddings for those chunks are weaker than a chunk containing the full argument.

**Paragraph/sentence splitting** — split on blank lines or sentence boundaries. Better than fixed-token for prose-heavy documents. Falls apart on structured notes with headers, bullets, and code blocks — the "paragraph" concept doesn't map cleanly.

**Structural splitting (H2 headers)** — split on semantic section markers. For an Obsidian vault where notes are organized around H2 sections, this is the right call. Each chunk corresponds to a coherent topic: "## Architecture", "## Tradeoffs", "## Next steps". The embedding captures a complete idea, not an arbitrary window.

The right strategy depends on your content type. Structural splitting works best when the documents have consistent heading structure. For dense prose (academic papers, long-form articles), paragraph or sentence splitting with a max-size cap may work better.

**Minimum chunk size matters.** A heading with one sentence beneath it ("## Status\nActive.") produces a nearly useless embedding. The fix is to merge short sections into the adjacent one before embedding.

## Embedding model choices

The embedding model is where most of the semantic quality comes from. Key considerations:

**Dimensions matter for storage and speed**, not quality. 512-dimensional vectors (voyage-3-lite) are adequate for a single-user vault of hundreds of documents. 1536-dimensional vectors (OpenAI) aren't meaningfully better at this scale and cost more to store and query.

**Asymmetric models** train separate representations for queries and documents, so "What approach did I use?" can land near "decided to use H2 structural splitting" even when the words are completely different. Symmetric models use the same representation for both, which works fine for "find similar documents" but struggles with natural language questions.

**Cost at small scale** is negligible. Voyage AI's free tier covers thousands of documents. The indexing cost for a personal vault is effectively zero. The expensive part is running your own embedding model, not using an API.

## Vector store options

For a local single-user system:

**ChromaDB** — embedded, no server to run, Python-native. The right choice for a personal knowledge system. Persists to disk. Supports cosine similarity, metadata filtering, and batch operations. The main limitation is it's not designed for multi-user or distributed use — but that's not the use case here.

**Qdrant** — more production-ready than ChromaDB, supports payload filtering with a richer query language. Worth switching to if you're building a multi-user product or need advanced filtering (e.g., range queries on numeric metadata).

**Pinecone / Weaviate** — hosted options. Removes the operational burden but adds a dependency on an external service. For a personal tool that should work offline, this is the wrong direction.

## Retrieval patterns

**Pure vector search** works well for "find content similar to this query". The weakness is precision on specific factual lookups — "what did I decide about X" may return broadly related chunks instead of the exact decision.

**Hybrid search** (vector + BM25 keyword) improves precision on specific terms, names, and exact phrases. ChromaDB doesn't support this natively; you'd need to run a separate BM25 index alongside it.

**Reranking** is the high-ROI upgrade for an existing vector search system. The pattern: retrieve top-20 chunks via vector similarity, then pass all 20 + the query to a cross-encoder model that scores them in context. The cross-encoder sees the query and each chunk together, so it can detect subtle relevance signals that the embedding model missed. Cross-encoders are slower than embedding lookup but you're only running them on 20 candidates, not the whole corpus.

**Pre-computed context** is an underrated pattern. Instead of querying at agent start time, run a context generation script after each sync and write a snapshot file. The agent loads the snapshot at session start — instant, no embedding call, always current. Best for "what's the state of the system right now" queries where freshness matters but precision doesn't.
