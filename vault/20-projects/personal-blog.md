---
title: Personal Blog
status: planned
tags: [writing, web, astro]
summary: Public writing on LLMOps, personal AI tooling, and building in public.
---

# Personal Blog

A place to write publicly about what I'm building and learning. The audience is other builders — people who want to understand *why* design decisions get made, not just what the output looks like.

## Why

Most technical writing either stays too high-level ("here's what RAG is") or too implementation-specific ("here's the code"). The gap I want to fill is the design-decision layer: why H2 chunking instead of fixed-token windows, why SQLite as a metadata mirror alongside ChromaDB, why pre-computed context snapshots instead of live queries at agent start time.

Writing forces me to verify that I actually understand something. If I can't explain the tradeoffs, I don't understand the system yet.

## Stack options

Three viable approaches, ordered by setup overhead:

**Astro** — static site generator, markdown-native, good component support if I want interactive elements later. Low runtime cost (deploy to Cloudflare Pages or Vercel free tier). Current favorite.

**Obsidian Publish** — lowest friction, uses the vault directly as content. Limited customization. Good for a first version, easy to migrate away from.

**Custom SvelteKit** — most flexibility, highest setup cost. Worth it only if I need server-side functionality (search, dynamic content). Overkill for a blog that's mostly static.

## Open questions

- How frequently do I realistically write? Monthly is achievable. Weekly is aspirational.
- Should posts live in the vault (and get indexed by the RAG system) or in a separate content repo?
- What's the canonical post format — long-form deep dive or short technical notes with code?
