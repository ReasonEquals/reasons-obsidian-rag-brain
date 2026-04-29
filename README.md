# reasons-obsidian-rag-brain

A working RAG pipeline over a local Obsidian vault. Five explicit LLMOps decisions made and defended: structural chunking over fixed-token windows, frontmatter separation, asymmetric embedding input types, SQLite as a queryable observability layer, and incremental indexing via content hashing.

**Stack:** Python · Voyage AI · ChromaDB · SQLite · Obsidian (or any markdown folder)
**Cost:** ~$0/mo (Voyage AI free tier; everything else is local)

---

## LLMOps decisions

Every RAG pipeline makes these choices. Most tutorials make them silently. This one makes them explicitly.

### 1. Structural chunking over fixed-token windows

`chunker.py` splits on `##` headers, not on token count.

Fixed-token windows (128/256/512 tokens) are the default because they're trivial to implement. The problem: they're semantically blind. A 256-token window splits mid-sentence when that's where the boundary falls, producing chunks where each embedding represents half of two different thoughts.

H2 structural splitting produces chunks that correspond to complete ideas — "## Architecture", "## Tradeoffs", "## Next steps". Each embedding represents a coherent topic. Retrieval improves because the query lands near a semantically complete chunk, not a fragment.

The tradeoff: structural splitting requires consistent heading discipline. For dense prose without headers, paragraph splitting may work better. For this vault use case, H2 is the correct boundary.

Handled: short sections (a `## Status` heading with one sentence) get merged into the next section rather than producing a useless single-word embedding. Duplicate headings within a file get deduplicated via slug counting to prevent silent ChromaDB overwrites.

See [`brain/chunker.py`](brain/chunker.py).

### 2. Frontmatter as metadata, not embedded text

YAML frontmatter fields (`status`, `tags`, `title`, `started`) are stored as ChromaDB metadata — not included in the embedded text.

Embedding `"status: active, tags: rag, embeddings, python"` alongside prose would dilute the semantic signal. The embedding should capture what the text is *about*. Structured fields belong in a structured store where they can be used for structured filtering — narrow by `status: active` before semantic ranking, not during.

This separation enables hybrid queries: filter by metadata first, then rank by semantic similarity within the filtered set. ChromaDB supports this via `where` dicts on `collection.query()`.

See [`brain/chunker.py:parse_frontmatter()`](brain/chunker.py).

### 3. Asymmetric input types (query vs. document)

`index.py` passes `input_type="document"` when embedding vault content and `input_type="query"` when embedding a search query. Most tutorials don't do this.

Voyage AI trains `voyage-3-lite` with two separate representations: one optimized for document content, one for query intent. A query embedding lands near document embeddings that *answer* that query, even when the vocabulary doesn't overlap.

Practical example: "What went wrong with the homelab setup?" lands near "## Lessons learned — Lost a week of Gitea data when a NVMe developed bad sectors." The model learned that questions about problems are semantically close to descriptions of those problems.

Using `input_type="document"` for both (the symmetric case) works for "find similar documents" but degrades on natural language questions. Asymmetric is a small change with a meaningful quality impact.

See [`brain/index.py:embed_texts()`](brain/index.py).

### 4. SQLite as a queryable observability layer

ChromaDB finds nearest neighbors. It doesn't do SQL. SQLite does SQL but doesn't do vectors. Both run in parallel.

`search_chunks` mirrors ChromaDB chunk metadata into SQLite. `search_log` records every query — text, result IDs, scores, timestamp.

This enables two things ChromaDB can't do alone:

```sql
-- Which source files produce the most chunks?
SELECT source, COUNT(*) FROM search_chunks GROUP BY source ORDER BY COUNT(*) DESC;

-- What has been searched, and how strong were the matches?
SELECT query, scores, searched_at FROM search_log ORDER BY searched_at DESC LIMIT 20;
```

The search log is retrieval observability: over time you can see which queries return strong matches and which return weak ones, and use that signal to improve chunking strategy or note structure.

See [`brain/index.py:ensure_search_tables()`](brain/index.py) and [`brain/index.py:sync_chunks_to_sqlite()`](brain/index.py).

### 5. Incremental indexing via content hashing

`index.py` stores an MD5 hash of each file's content in ChromaDB chunk metadata. On subsequent runs, it pulls stored hashes, compares against current file content, and skips unchanged files.

At personal vault scale (200+ files, 3 edited today), re-embedding everything on every run is slow and wastes API quota. The incremental path only embeds what changed.

Two edge cases handled explicitly:

**Deleted files:** If a file is removed since the last index run, its chunks persist in ChromaDB and keep appearing in search results. A diff of `stored_hashes.keys()` against current vault files identifies stale sources and deletes their chunks before indexing.

**Restructured files:** If a file's H2 structure changes (a heading renamed, a section split into two), old chunk IDs no longer exist. Upsert alone would leave orphan chunks under the old IDs. The fix: before upserting new chunks for a changed file, delete all existing chunks for that source path from ChromaDB.

See [`brain/index.py:run_index()`](brain/index.py).

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

**Data flow:** Write markdown → `sync.py --once` → SQLite current → `index.py` → ChromaDB current → `search.py "query"` → results.

`context_snapshot.py` is the agent layer bridge: reads SQLite and writes `vault-state.md`, a pre-computed summary an AI agent loads at session start instead of scanning all files. The pre-computed context pattern — generate once after sync, load at agent start — avoids per-session retrieval latency and keeps the context window predictable.

---

## Run it

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/yourusername/reasons-obsidian-rag-brain.git
cd reasons-obsidian-rag-brain
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

Incremental run after editing a note:
```bash
python brain/sync.py --once && python brain/index.py && python brain/context_snapshot.py
```

Watch mode during active writing sessions:
```bash
python brain/sync.py --watch   # re-syncs SQLite on every .md save, including deletions
```

Optional shell function (add to `~/.zshrc` or `~/.bashrc`) to search from anywhere:
```bash
brainsearch() { (cd /path/to/reasons-obsidian-rag-brain && .venv/bin/python brain/search.py "$@") }
```
Then `brainsearch "what did I decide about X?"` works in any directory.

---

## Point it at your own vault

Set `VAULT_PATH` in `.env` to your Obsidian vault directory (or any folder of markdown files):

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

All fields are optional. Files without frontmatter still get indexed and searched.

**Skip patterns** (files/directories excluded from indexing):

```python
SKIP_PATTERNS = [".obsidian/", ".git/", ".claude/", "_assets/", "_data/"]
```

Add patterns to `SKIP_PATTERNS` in `brain/index.py` for directories you don't want indexed.

---

## Extend it

Natural next steps, ordered by complexity:

- **Add metadata filters to search:** Pass a `where` dict to `collection.query()` — filter by `status`, `tags`, or `source` prefix before semantic ranking.
- **Wire search output to Claude:** `search.py` returns the most relevant chunks. Pass them to the Claude API as context for a question-answering step — the retrieval is done, wiring to an LLM call is a few lines.
- **Scheduled context refresh:** Wire `context_snapshot.py` to run after every sync (git post-commit hook, launchd/cron). The agent context file stays current automatically.
- **Swap the embedding model:** Only `embed_texts()` in `brain/index.py` touches Voyage AI. To switch to a local model (Ollama, sentence-transformers) or a different API (OpenAI), replace that one function.
- **Scale the vector store:** ChromaDB is local and single-process. For multi-user or hosted deployments, swap `chromadb.PersistentClient` for Qdrant, Weaviate, or Pinecone — `collection.upsert()` and `collection.query()` are the only ChromaDB-specific surface.

---

## File reference

| File | What it demonstrates |
|------|---------------------|
| [`brain/chunker.py`](brain/chunker.py) | H2 structural chunking, frontmatter separation, minimum-word merge, duplicate slug deduplication |
| [`brain/index.py`](brain/index.py) | Asymmetric embedding (query vs. document), incremental indexing via MD5 hash, stale chunk cleanup, SQLite mirror, rate-limit retry backoff |
| [`brain/search.py`](brain/search.py) | Cosine distance → similarity conversion, score interpretation thresholds, query logging to SQLite |
| [`brain/sync.py`](brain/sync.py) | Frontmatter → SQLite upsert-on-conflict, orphan cleanup, watch mode with delete handling, idempotent schema creation |
| [`brain/context_snapshot.py`](brain/context_snapshot.py) | Pre-computed agent context pattern, write-if-changed to avoid watcher churn |
| [`vault/`](vault/) | Sample knowledge base — run the demo against this before pointing at your own vault |
