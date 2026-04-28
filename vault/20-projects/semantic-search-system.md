---
title: Semantic Search System
status: active
tags: [rag, embeddings, python, chromadb]
started: 2024-01-10
summary: Semantic search over a local markdown vault using Voyage AI embeddings and ChromaDB.
repo: https://github.com/yourusername/obsidian-rag-brain
---

# Semantic Search System

A three-layer knowledge OS: structured markdown notes → SQLite for queryable metadata → ChromaDB for semantic search. The key insight is that markdown is always the source of truth; the databases are derived and can be rebuilt from scratch at any time.

## What it is

A local RAG system that treats an Obsidian vault as a knowledge base. You write notes in markdown, run two scripts, and can then ask natural language questions against everything you've written.

The system doesn't use a hosted service for search or storage. Everything runs locally: ChromaDB stores vectors on disk, SQLite stores metadata, and the only external call is to Voyage AI for embeddings (which could be swapped for a local model).

## Architecture

Three layers, each derived from the previous:

1. **Vault** — markdown files with YAML frontmatter. Source of truth.
2. **SQLite** — sync.py parses frontmatter and upserts to brain.db. Enables structured queries (active projects, roadmap status).
3. **ChromaDB** — index.py chunks each file on H2 boundaries, embeds via Voyage AI, stores vectors. Enables semantic queries.

The context_snapshot.py script bridges layer 2 and the agent layer: it reads SQLite and writes vault-state.md, a pre-computed summary that an AI agent can load at session start without scanning all 200+ files.

## Status

Core pipeline complete and working:
- sync.py: parsing frontmatter from all vault dirs, upserting to SQLite
- index.py: incremental indexing via MD5 hash, stale chunk cleanup on delete
- search.py: CLI query interface with score interpretation and SQLite logging
- context_snapshot.py: generates vault-state.md from project/roadmap tables

## Next steps

- Add ChromaDB metadata filters to search.py (e.g., `--status active` to narrow results to active projects before ranking)
- Experiment with reranking: use a cross-encoder to re-score top-20 results from ChromaDB before returning top-5
- Add a scheduled agent that loads vault-state.md at session start and surfaces anything stale or overdue
