# obsidian-rag-brain

Semantic search and structured querying over a local Obsidian vault. A personal knowledge OS template demonstrating LLMOps patterns: H2-aware chunking, asymmetric embedding, incremental indexing, SQLite as a queryable metadata mirror, and pre-computed agent context.

**Stack:** Python · Voyage AI · ChromaDB · SQLite · Obsidian (or any markdown folder)  
**Cost:** ~$0/mo (Voyage AI free tier covers a personal vault; everything else is local)

---

## What this is

The problem: you accumulate decisions, notes, and context across dozens of markdown files. Finding the right note at the right moment requires either memorizing your own system or manually hunting through files. Giving an AI agent useful context requires the same manual hunting, at every session start.

The solution is a three-layer pipeline where markdown is always the source of truth:

1. **Vault** — markdown files with YAML frontmatter (title, status, tags). The only thing you write to.
2. **SQLite** — `sync.py` parses frontmatter and upserts to `brain.db`. Structured metadata, queryable with SQL. Everything in the database can be rebuilt from the markdown at any time.
3. **ChromaDB** — `index.py` chunks each file on H2 boundaries, embeds via Voyage AI, stores vectors locally. Natural language queries against everything you've written.

A fourth component bridges layers 2 and 3 and the agent layer: `context_snapshot.py` reads SQLite and writes `vault-state.md` — a pre-computed summary an AI agent loads at session start instead of scanning all files.

This is a template, not a product. It's designed to be cloned, pointed at your own vault, and extended.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Vault                            │
│   20-projects/*.md  30-roadmap/*.md  40-reference/*.md  │
│             (YAML frontmatter + markdown body)          │
└────────────┬────────────────────────┬───────────────────┘
             │                        │
        sync.py                  index.py
             │                        │
             ▼                        ▼
      ┌─────────────┐        ┌──────────────────┐
      │   SQLite    │        │    ChromaDB       │
      │  brain.db   │        │  (local vectors)  │
      │  projects   │        │  voyage-3-lite    │
      │  roadmap    │        │  512-dim cosine   │
      │  daily_notes│        └────────┬─────────┘
      │  sync_log   │                 │
      └──────┬──────┘           search.py
             │                        │
    context_snapshot.py          natural language
             │                   query + scores
             ▼
      vault-state.md
    (agent context file)
```

**Data flow:** Write markdown → `sync.py --once` → SQLite current → `index.py` → ChromaDB current → `search.py "query"` → results. Run `context_snapshot.py` after sync to refresh the agent context file.

---

## Design decisions worth understanding

### H2 chunking, not fixed-token windows

`chunker.py` splits each markdown file on `##` headers, not on token count. This is the highest-leverage design decision in the system.

Fixed-token windows (128/256/512 tokens) are the default because they're easy to implement. The problem: they're semantically blind. A 256-token window will split a paragraph mid-sentence if that's where the boundary falls, producing chunks that each capture half of a complete thought. The embedding for each chunk represents a fragment.

H2 structural splitting produces chunks that correspond to complete ideas — "## Architecture", "## Tradeoffs", "## Next steps". Each embedding represents a coherent topic. When a query asks about architecture, the "Architecture" section embeds close to it. When the window happened to split "Architecture" across two chunks, neither one does.

The tradeoff: structural splitting only works when documents have consistent heading structure. For dense prose without headers, paragraph splitting may work better. For this vault use case, H2 is the right boundary.

See `brain/chunker.py` for the implementation, including the minimum-word-count merge logic that handles the common case of `## Status\nActive.` — a one-word section that would produce a useless embedding.

### Frontmatter as metadata, not embedded text

YAML frontmatter fields (`status`, `tags`, `title`, `started`) are stored as ChromaDB metadata — not included in the embedded text. This is a deliberate separation.

Embedding "status: active, tags: rag, embeddings, python" alongside prose about the project would dilute the semantic signal. The embedding should capture what the text is *about*, and structured fields should be used for structured filtering (filter by `status: active` before semantic ranking, not during).

This separation also enables hybrid queries: narrow by metadata first (only active projects), then rank by semantic similarity within that filtered set.

See `brain/chunker.py:parse_frontmatter()` for the frontmatter stripping and normalization logic.

### Asymmetric input types (query vs. document)

`index.py` passes `input_type="document"` when embedding vault content, and `input_type="query"` when embedding a search query. Most embedding tutorials don't do this.

Voyage AI trains `voyage-3-lite` with two separate representations: one for documents (emphasize content meaning) and one for queries (emphasize what the user is looking for). A *query* embedding lands near *document* embeddings that answer that query, even when the words are completely different.

Practical example: "What went wrong with the homelab setup?" lands near "## Lessons learned — Lost a week of Gitea data when a NVMe developed bad sectors." The vocabulary doesn't overlap, but the semantic intent does. This works because the model learned that questions about problems are semantically close to descriptions of those problems.

Using `input_type="document"` for both query and index — the symmetric case — works fine for "find similar documents" but degrades on natural language questions. The asymmetric approach is a small change with a meaningful quality impact.

See `brain/index.py:embed_texts()` for the implementation and a more detailed explanation in the docstring.

### SQLite as queryable metadata mirror

ChromaDB stores vectors and retrieves nearest neighbors. It doesn't do SQL. SQLite does SQL but doesn't do vectors. The system runs both in parallel, with ChromaDB as the source of truth for vectors and SQLite as a queryable view of the metadata.

This enables two classes of queries that neither store handles alone:

- **Structured:** "Which source files produce the most chunks?" — `SELECT source, COUNT(*) FROM search_chunks GROUP BY source ORDER BY COUNT(*) DESC`
- **Observability:** "What have I been searching for?" — `SELECT query, searched_at FROM search_log ORDER BY searched_at DESC LIMIT 20`

Every search is logged to the `search_log` table (query text, result IDs, scores, timestamp). Over time this becomes a record of retrieval quality — you can see which queries return strong matches and which return weak ones, and use that to improve your notes or your chunking strategy.

The `search_chunks` table mirrors ChromaDB chunk metadata. If both stores drift out of sync, `index.py --force` rebuilds both from scratch.

See `brain/index.py:ensure_search_tables()` and `brain/index.py:sync_chunks_to_sqlite()`.

### Incremental indexing via MD5 hash

`index.py` stores an MD5 hash of each file's content in ChromaDB chunk metadata. On subsequent runs, it pulls stored hashes from ChromaDB, compares them to current file content, and skips unchanged files.

This matters for personal vault scale: you might have 200+ files but only edit 3 today. Re-embedding all 200 files on every run would be slow and would burn API quota unnecessarily.

Two edge cases handled explicitly:

1. **Deleted files:** If a file was removed since the last index run, its chunks are still in ChromaDB and would keep appearing in search results. The cleanup pass diffs `{stored_hashes.keys()}` against `{current vault files}` and deletes stale chunks before indexing.

2. **Restructured files:** If a file's H2 structure changes (a heading renamed, a section split into two), the old chunk IDs no longer exist. Upsert alone would leave orphan chunks under the old IDs. The fix: before upserting new chunks for a changed file, delete all existing chunks for that source path from ChromaDB.

See `brain/index.py:run_index()` for the full incremental pipeline.

---

## Quickstart

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/yourusername/obsidian-rag-brain.git
cd obsidian-rag-brain
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env: set VAULT_PATH, DB_PATH, and VOYAGE_API_KEY

# 4. Get a Voyage AI API key (free tier, no credit card required)
# https://www.voyageai.com → Sign up → API Keys

# 5. Sync vault frontmatter to SQLite
python brain/sync.py --once
# Expected: ✓ Sync complete: 8 upserted, 0 skipped, 0 deleted | DB: ~/vault/_data/brain.db

# 6. Index vault into ChromaDB
python brain/index.py
# Expected: Found 8 markdown files / Changed: 8 files → N chunks / Embedding N chunks... done

# 7. Search
python brain/search.py "what chunking strategy did I decide on?"
# Expected: results from vault/40-reference/rag-architecture-notes.md with score < 0.35

# 8. Generate agent context snapshot (optional)
python brain/context_snapshot.py
# Expected: ✓ vault-state.md updated (3 projects, 2 roadmap items)
```

To run incrementally after editing a note:
```bash
python brain/sync.py --once && python brain/index.py && python brain/context_snapshot.py
```

Or use watch mode during active writing sessions:
```bash
python brain/sync.py --watch   # re-syncs SQLite on every .md file save
```

---

## Using your own vault

Point `VAULT_PATH` in `.env` at your Obsidian vault directory (or any folder of markdown files):

```
VAULT_PATH=/Users/you/your-obsidian-vault
DB_PATH=/Users/you/your-obsidian-vault/_data/brain.db
```

**Frontmatter fields the system uses:**

| Field | Used by | Notes |
|-------|---------|-------|
| `title` | sync.py, context_snapshot.py | Falls back to filename if absent |
| `status` | sync.py, context_snapshot.py, search.py | `active`, `planned`, `complete` |
| `tags` | sync.py, search.py | List or space-separated string |
| `started` | sync.py | ISO date string |
| `completed` | sync.py | ISO date string |
| `summary` | sync.py, context_snapshot.py | One-sentence description |
| `repo` | sync.py, context_snapshot.py | URL |
| `priority` | sync.py | `high`, `medium`, `low` (roadmap items) |
| `project` | sync.py | Slug of parent project (roadmap items) |

All fields are optional. Files without frontmatter still get indexed and searched — they just won't have metadata to filter on.

**Skip patterns** (files/directories excluded from indexing):

```python
SKIP_PATTERNS = [".obsidian/", ".git/", ".claude/", "_assets/", "_data/"]
```

Add your own patterns to `SKIP_PATTERNS` in `brain/index.py` if you have directories you don't want indexed (templates, archive folders, etc.).

---

## Extending this

A few natural next steps, ordered by complexity:

- **Add metadata filters to search:** Pass a ChromaDB `where` dict to `collection.query()` — filter by `status`, `tags`, or `source` prefix before semantic ranking. Useful for "search only active projects" or "search only reference notes".

- **Pipe search output into Claude:** `search.py` returns the most relevant chunks. Pass them to the Claude API as context for a question-answering step. The retrieval is already done; wiring it to an LLM call is a few lines.

- **Scheduled agent context refresh:** Wire `context_snapshot.py` to run after every sync (git post-commit hook, launchd/cron). The agent context file stays current automatically without manual runs.

- **Swap the embedding model:** Only `embed_texts()` in `brain/index.py` touches Voyage AI. To switch to a local model (Ollama, sentence-transformers) or a different API (OpenAI), replace that one function. The rest of the pipeline is unchanged.

- **Scale up the vector store:** ChromaDB is local and single-process. For multi-user or hosted deployments, swap `chromadb.PersistentClient` for Qdrant, Weaviate, or Pinecone. The `collection.upsert()` / `collection.query()` calls are the only ChromaDB-specific surface.

---

## Tech stack

| Component | Tool | Cost |
|-----------|------|------|
| Note storage | Obsidian (or any markdown) | Free |
| Version control | Git | Free |
| Sync pipeline | Python + python-frontmatter | Free |
| Structured store | SQLite (stdlib) | Free |
| Scheduler (macOS) | launchd | Free |
| Embeddings | Voyage AI voyage-3-lite | Free tier / $0.02/1M tokens |
| Vector store | ChromaDB (local) | Free |
| Watch mode | watchdog | Free |

The only external dependency is Voyage AI for embeddings. Everything else is local, offline-capable, and has no ongoing cost.

---

## File reference

| File | What it demonstrates |
|------|---------------------|
| `brain/chunker.py` | H2 structural chunking, frontmatter separation, minimum-word merge, duplicate slug deduplication |
| `brain/index.py` | Asymmetric embedding (query vs. document), incremental indexing via MD5 hash, stale chunk cleanup, SQLite mirror, rate-limit retry backoff |
| `brain/search.py` | Cosine distance → similarity conversion, score interpretation thresholds, query logging to SQLite |
| `brain/sync.py` | Frontmatter → SQLite upsert-on-conflict, orphan cleanup, watch mode, idempotent schema creation |
| `brain/context_snapshot.py` | Pre-computed agent context pattern, write-if-changed to avoid watcher churn |
| `vault/` | Sample knowledge base with realistic content — run the demo against this before pointing at your own vault |
