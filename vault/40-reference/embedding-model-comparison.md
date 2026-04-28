---
title: Embedding Model Comparison
tags: [embeddings, ai, voyage, openai, models]
summary: Comparison of embedding models for local RAG systems — voyage-3-lite, OpenAI text-embedding-3-small, and local options.
---

# Embedding Model Comparison

Notes comparing embedding models for the personal RAG system. Focused on models that are either free-tier viable or cheap enough to not think about at personal vault scale.

## voyage-3-lite

Voyage AI's smallest model. 512-dimensional vectors, fast, cheap (or free on the free tier). Supports asymmetric input types (`input_type="query"` vs `input_type="document"`).

**Why I use it:** The asymmetric input type support is the killer feature. When you embed a query with `input_type="query"`, the model knows to represent it as *something being searched for*, not a document to be retrieved. Voyage trains the model so query embeddings land near document embeddings that answer the query, even when the vocabulary is completely different.

**Practical quality:** Excellent for a personal knowledge base. Searches like "what did I decide about the vector store?" reliably surface the relevant project notes. "What went wrong during the homelab setup?" finds the lessons-learned section. The 512-dimensional representation is more than sufficient for a corpus of hundreds of documents.

**Cost:** Free tier covers a large number of tokens per month — more than enough for a personal vault even with aggressive reindexing. Paid tier starts at $0.02/1M tokens, which is effectively free at this scale.

**Limitation:** No batch embedding endpoint with callback — you have to poll or wait synchronously. Fine for a personal tool, would need redesign for a high-throughput service.

## OpenAI text-embedding-3-small

OpenAI's small embedding model. 1536 dimensions by default (can be truncated). No asymmetric input type support.

**Quality:** Competitive with voyage-3-lite on standard benchmarks. The lack of asymmetric input types means query-document retrieval relies on the model having learned symmetric similarity — "What framework did I choose?" needs to land near "decided on Astro" via general semantic similarity rather than explicit query-vs-document training.

**Cost:** $0.02/1M tokens. Comparable to Voyage at this scale.

**Why I didn't choose it:** The asymmetric input type is worth more than the marginal quality difference at personal vault scale. If I were using OpenAI for everything else and wanted to minimize providers, this would be the default choice.

## When to use asymmetric input types

The asymmetric input type distinction matters most for question-answering retrieval — when the user's query is phrased differently from how the answer is phrased in the document.

**High benefit scenarios:**
- Natural language questions against note-style content ("What were the tradeoffs I considered for X?")
- User-facing search where queries are short and conversational
- Domain-specific content where synonyms and paraphrases are common

**Low benefit scenarios:**
- "More like this" / similar document retrieval — query and document are both document-style
- Keyword-heavy search where the user types terms that appear verbatim in documents
- Very small corpora (< 50 documents) where vector distance is less precise

## Local embedding models

Running embeddings locally via Ollama or sentence-transformers is viable for offline use or data-sensitivity requirements. The tradeoffs:

**CPU inference is too slow for interactive use.** A 13B parameter local model on CPU takes several seconds per query. Acceptable for batch indexing, not for interactive search.

**GPU inference is fast but requires hardware.** An RTX 3090 or similar can run nomic-embed-text or similar models at practical speeds. If you have a GPU, local embeddings are a real option and remove the Voyage API dependency entirely.

**Quality gap exists but is shrinking.** nomic-embed-text and similar open models are competitive with the smaller commercial models. For a personal use case where you control the content, the quality gap is unlikely to matter.

**Recommendation:** Use Voyage AI API unless you have a GPU available for local inference or have a strong reason to avoid external APIs.
