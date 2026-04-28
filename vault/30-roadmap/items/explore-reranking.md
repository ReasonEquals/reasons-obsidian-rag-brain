---
title: Explore reranking
status: planned
project: semantic-search-system
priority: medium
---

# Explore reranking

## Why

Semantic search via vector similarity is a coarse filter. The top-5 results are almost always in the right neighborhood, but they're not necessarily in the right order. A cross-encoder reranker sees the query and each candidate together in the same context window — it can detect relevance signals that the bi-encoder embedding model missed.

The pattern: retrieve top-20 from ChromaDB via vector similarity, pass all 20 + the query to the reranker, return the top-5 by reranker score. The reranker is slower than embedding lookup but you're only running it on 20 candidates, not the whole corpus.

## Options

**Voyage AI reranking API** — the simplest integration. Voyage offers a `/rerank` endpoint that takes a query + list of documents and returns relevance scores. One additional API call per search. No local model needed. Cost is per token, comparable to embeddings.

**Cohere Rerank** — similar API, strong benchmark performance. Another provider dependency. Free tier exists.

**Cross-encoder via sentence-transformers** — run a local cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`). No API call, works offline, but requires a Python environment with PyTorch and enough RAM to load the model (~300MB). CPU inference is fast enough for 20 candidates (< 1 second).

**Recommendation:** Start with the Voyage reranking API since we're already using Voyage for embeddings (one provider, one API key). If latency or cost becomes a concern, switch to the local cross-encoder.

The implementation is a thin wrapper around the existing search.py — retrieve `top_k * 4` from ChromaDB, rerank, return `top_k`. The reranking step should be opt-in via a `--rerank` flag so the existing behavior is unchanged.
