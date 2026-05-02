#!/usr/bin/env python3
"""
Index vault markdown files into ChromaDB via Voyage AI embeddings.

Walks the vault, chunks each markdown file, embeds the chunks via
Voyage AI's HTTP API, and stores them in a local ChromaDB collection.
Also mirrors chunk metadata to SQLite (brain.db) for SQL querying.

Usage:
    python brain/index.py           # incremental (skip unchanged files)
    python brain/index.py --force   # full reindex

Env vars (via .env):
    VAULT_PATH      Path to the vault directory (default: ~/vault)
    DB_PATH         Path to brain.db (default: ~/vault/_data/brain.db)
    VOYAGE_API_KEY  Voyage AI API key (required)
"""

import argparse
import hashlib
import sqlite3
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional deps — fail with clear instructions
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # env vars can come from shell

import os

try:
    import chromadb
except ImportError:
    print("chromadb not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

from chunker import chunk_markdown, word_count

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VAULT_PATH = Path(os.getenv("VAULT_PATH", "~/vault")).expanduser()
DB_PATH = Path(os.getenv("DB_PATH", "~/vault/_data/brain.db")).expanduser()
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
CHROMA_DIR = Path(__file__).parent / ".chroma"
COLLECTION_NAME = "vault"

# Voyage AI embedding config
VOYAGE_MODEL = "voyage-3-lite"
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

# Files/dirs to skip — these aren't useful for semantic search
SKIP_PATTERNS = [
    ".obsidian/",
    ".git/",
    ".claude/",
    "_assets/",
    "_data/",
    "_checkpoints/",
]

# Voyage API batch limit — keep batches small to stay under rate limits.
# Free tier (even with payment method) starts at 10K TPM; this avoids 429s.
EMBED_BATCH_SIZE = 20
EMBED_BATCH_DELAY = 1.0  # seconds between batches

# ChromaDB upsert batch limit (their internal max is ~5461)
CHROMA_BATCH_SIZE = 5000


# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------

def load_vault_files() -> list[tuple[str, str]]:
    """Walk the vault and return (relative_path, content) pairs.

    Skips binary files, hidden dirs, and patterns in SKIP_PATTERNS.
    """
    files = []
    for md_file in sorted(VAULT_PATH.rglob("*.md")):
        rel = str(md_file.relative_to(VAULT_PATH))

        # Check skip patterns — use path-component-aware matching:
        # ".obsidian/" matches the directory component, not a substring
        # "/CLAUDE.md" matches files named exactly CLAUDE.md (leading /)
        skip = False
        for pattern in SKIP_PATTERNS:
            if pattern.startswith("/"):
                # Exact filename match (strip leading /)
                if rel.endswith(pattern[1:]) and ("/" not in rel[:-len(pattern[1:])] or rel == pattern[1:]):
                    skip = True
                    break
            elif pattern.endswith("/"):
                # Directory prefix match
                if rel.startswith(pattern) or f"/{pattern}" in f"/{rel}":
                    skip = True
                    break
            else:
                # Exact filename match anywhere in path
                if rel.endswith(pattern) or f"/{pattern}" in f"/{rel}":
                    skip = True
                    break
        if skip:
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        if content.strip():
            files.append((rel, content))

    return files


# ---------------------------------------------------------------------------
# Hashing (for incremental indexing)
# ---------------------------------------------------------------------------

def file_hash(content: str) -> str:
    """MD5 hash of file content. Used to detect changes between index runs.

    Why MD5 and not SHA-256? Speed doesn't matter at this scale, and we're
    not using it for security — just change detection. MD5 is fine.
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()  # noqa: S324


def get_stored_hashes(collection) -> dict[str, str]:
    """Pull file hashes from ChromaDB metadata.

    Each chunk stores its source file's hash. We only need one hash per
    file (they're all the same), so we deduplicate by source path.
    """
    try:
        all_meta = collection.get(include=["metadatas"])
        hashes = {}
        for meta in all_meta["metadatas"]:
            source = meta.get("source", "")
            h = meta.get("file_hash", "")
            if source and h:
                hashes[source] = h
        return hashes
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Embedding via Voyage AI HTTP API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed a list of texts via Voyage AI's REST API.

    This is the raw HTTP call — no SDK. Here's what's happening:

    1. We POST a JSON body with the texts and model name
    2. Voyage returns an array of embedding objects, each containing a
       vector of 512 floats (for voyage-3-lite)
    3. Each float represents one dimension in the embedding space

    The `input_type` parameter matters:
    - "document" — used when indexing. Tells the model to emphasize
      the content's meaning.
    - "query" — used when searching. Tells the model to emphasize
      what the user is looking for.

    Voyage trains the model so that a *query* embedding lands near
    *document* embeddings that answer that query, even when they use
    completely different words. "What framework should I use?" will be
    near a chunk about "decided on Astro for static generation" because
    the model learned that queries about framework choices are
    semantically close to descriptions of framework decisions.

    This asymmetry is why we don't just use "document" for both.
    """
    if not VOYAGE_API_KEY:
        print("ERROR: VOYAGE_API_KEY not set. Add it to .env")
        sys.exit(1)

    all_embeddings = []

    # Batch to stay under Voyage's per-request limit
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]

        # Delay between batches to respect rate limits
        if i > 0:
            time.sleep(EMBED_BATCH_DELAY)

        # Retry with backoff on 429 (rate limit)
        max_retries = 4
        for attempt in range(max_retries):
            response = requests.post(
                VOYAGE_URL,
                headers={
                    "Authorization": f"Bearer {VOYAGE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": VOYAGE_MODEL,
                    "input": batch,
                    "input_type": input_type,
                },
                timeout=30,
            )

            if response.status_code == 429:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
                print(f"\n  Rate limited, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
                continue

            break

        if response.status_code != 200:
            print(f"ERROR: Voyage API returned {response.status_code}")
            print(response.text)
            sys.exit(1)

        data = response.json()
        # Response shape: { "data": [{"embedding": [...], "index": 0}, ...] }
        # Sort by index to guarantee order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        all_embeddings.extend([item["embedding"] for item in sorted_data])

    return all_embeddings


# ---------------------------------------------------------------------------
# SQLite — chunk metadata mirror + search log
# ---------------------------------------------------------------------------

def get_db_conn() -> sqlite3.Connection:
    """Open a connection to brain.db with WAL mode for concurrent read safety."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def ensure_search_tables(conn: sqlite3.Connection) -> None:
    """Create search-related tables if they don't exist.

    Two tables:
    - search_chunks: mirrors ChromaDB chunk metadata into SQLite so you
      can do SQL queries like "which files produce the most chunks?" or
      "show me all chunks from active projects". ChromaDB can't do this.
    - search_log: records every search query with results and scores.
      Useful for tracking retrieval quality over time, and as a SQL
      learning playground.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS search_chunks (
            chunk_id    TEXT PRIMARY KEY,
            source      TEXT NOT NULL,
            section     TEXT NOT NULL,
            doc_title   TEXT NOT NULL,
            status      TEXT,
            tags        TEXT,
            word_count  INTEGER NOT NULL,
            file_hash   TEXT NOT NULL,
            indexed_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_search_chunks_source
            ON search_chunks(source);

        CREATE TABLE IF NOT EXISTS search_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT NOT NULL,
            top_k       INTEGER NOT NULL,
            result_ids  TEXT NOT NULL,
            scores      TEXT NOT NULL,
            searched_at TEXT NOT NULL
        );
    """)


def sync_chunks_to_sqlite(conn: sqlite3.Connection, chunks: list[dict]) -> None:
    """Upsert chunk metadata into the search_chunks table.

    This is a one-way mirror: ChromaDB is the source of truth for vectors,
    SQLite is a queryable view of the metadata. If they drift, --force
    reindex resets both.
    """
    now = _now_iso()
    conn.executemany(
        """
        INSERT INTO search_chunks (chunk_id, source, section, doc_title,
                                   status, tags, word_count, file_hash, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            source=excluded.source, section=excluded.section,
            doc_title=excluded.doc_title, status=excluded.status,
            tags=excluded.tags, word_count=excluded.word_count,
            file_hash=excluded.file_hash, indexed_at=excluded.indexed_at
        """,
        [
            (
                chunk["id"],
                chunk["metadata"].get("source", ""),
                chunk["metadata"].get("section", ""),
                chunk["metadata"].get("doc_title", ""),
                chunk["metadata"].get("status", ""),
                chunk["metadata"].get("tags", ""),
                word_count(chunk["text"]),
                chunk["metadata"].get("file_hash", ""),
                now,
            )
            for chunk in chunks
        ],
    )
    conn.commit()


def delete_chunks_from_sqlite(conn: sqlite3.Connection, source: str) -> None:
    """Remove all chunks for a given source file from SQLite."""
    conn.execute("DELETE FROM search_chunks WHERE source = ?", (source,))
    conn.commit()


def clear_all_chunks_sqlite(conn: sqlite3.Connection) -> None:
    """Wipe the search_chunks table (used during --force reindex)."""
    conn.execute("DELETE FROM search_chunks")
    conn.commit()


def log_search(conn: sqlite3.Connection, query: str, top_k: int,
               result_ids: list[str], scores: list[float]) -> None:
    """Log a search query and its results to SQLite.

    Storing result_ids and scores as comma-separated strings keeps the
    schema simple. For a learning project this is fine — you can always
    split them in SQL with json_each() or in Python.
    """
    conn.execute(
        """
        INSERT INTO search_log (query, top_k, result_ids, scores, searched_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            query,
            top_k,
            ",".join(result_ids),
            ",".join(f"{s:.4f}" for s in scores),
            _now_iso(),
        ),
    )
    conn.commit()


def _now_iso() -> str:
    """UTC timestamp in ISO 8601."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def run_index(force: bool = False) -> None:
    """Full indexing pipeline: load → chunk → diff → embed → store.

    With --force: re-embed and upsert everything, wipe stale data.
    Without: skip files whose content hash hasn't changed, clean up
    chunks from deleted files.
    """
    print(f"Vault: {VAULT_PATH}")
    print(f"Model: {VOYAGE_MODEL}")
    print(f"Store: {CHROMA_DIR}")
    print(f"DB:    {DB_PATH}")
    print()

    # Connect to ChromaDB (creates the directory if needed)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Connect to SQLite and ensure tables exist
    db_conn = get_db_conn()
    ensure_search_tables(db_conn)

    # Load vault files
    vault_files = load_vault_files()
    print(f"Found {len(vault_files)} markdown files")

    # Get stored hashes for incremental indexing
    stored_hashes = {} if force else get_stored_hashes(collection)

    # --- Clean up chunks from deleted files ---
    # If a vault file was removed since the last index, its chunks are
    # still in ChromaDB. Without this cleanup, deleted files would keep
    # appearing in search results forever.
    current_sources = {rel for rel, _ in vault_files}
    stale_sources = set(stored_hashes.keys()) - current_sources
    if stale_sources:
        print(f"Removing {len(stale_sources)} deleted file(s) from index:")
        for source in sorted(stale_sources):
            print(f"  ✗ {source}")
            try:
                existing = collection.get(where={"source": source}, include=[])
                if existing["ids"]:
                    collection.delete(ids=existing["ids"])
            except Exception as e:
                print(f"    Warning: could not remove from ChromaDB: {e}")
            delete_chunks_from_sqlite(db_conn, source)
        print()

    # Chunk all files, tracking which are new/changed
    all_chunks = []
    changed_files = []
    skipped_files = []
    parse_errors = []

    for rel_path, content in vault_files:
        current_hash = file_hash(content)
        try:
            chunks = chunk_markdown(content, rel_path)
        except Exception as exc:
            # Malformed YAML frontmatter or other chunker failure — skip the
            # file but keep indexing the rest of the vault. One bad file
            # should not abort the whole pass.
            parse_errors.append((rel_path, str(exc).splitlines()[0]))
            continue

        if not force and stored_hashes.get(rel_path) == current_hash:
            skipped_files.append(rel_path)
            continue

        changed_files.append(rel_path)

        # Tag each chunk with the file hash (for future incremental runs)
        for chunk in chunks:
            chunk["metadata"]["file_hash"] = current_hash
            all_chunks.append(chunk)

    print(f"Changed: {len(changed_files)} files → {len(all_chunks)} chunks")
    print(f"Skipped: {len(skipped_files)} files (unchanged)")
    if parse_errors:
        print(f"Parse errors: {len(parse_errors)} file(s) skipped")
        for rel, err in parse_errors:
            print(f"  ⚠ {rel}: {err}")
    print()

    if not all_chunks:
        print("Nothing to index.")
        db_conn.close()
        return

    # Remove old chunks from changed files before upserting new ones.
    # This handles the case where a file's H2 structure changed (old
    # chunk IDs no longer exist, so upsert alone would leave orphans).
    if not force:
        for rel_path in changed_files:
            try:
                existing = collection.get(
                    where={"source": rel_path},
                    include=[],
                )
                if existing["ids"]:
                    collection.delete(ids=existing["ids"])
            except Exception as e:
                print(f"  Warning: could not clean old chunks for {rel_path}: {e}")
            delete_chunks_from_sqlite(db_conn, rel_path)
    else:
        # Force mode: wipe the whole collection and recreate
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        clear_all_chunks_sqlite(db_conn)

    # Embed all chunks
    texts = [chunk["text"] for chunk in all_chunks]
    print(f"Embedding {len(texts)} chunks via Voyage AI...", end=" ", flush=True)
    t0 = time.time()
    embeddings = embed_texts(texts, input_type="document")
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    # Upsert into ChromaDB in batches (their internal limit is ~5461)
    for i in range(0, len(all_chunks), CHROMA_BATCH_SIZE):
        batch = all_chunks[i:i + CHROMA_BATCH_SIZE]
        batch_embeddings = embeddings[i:i + CHROMA_BATCH_SIZE]
        batch_texts = texts[i:i + CHROMA_BATCH_SIZE]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=[chunk["metadata"] for chunk in batch],
        )

    # Mirror to SQLite
    sync_chunks_to_sqlite(db_conn, all_chunks)

    total = collection.count()
    print(f"\nCollection '{COLLECTION_NAME}' now has {total} chunks total.")
    print("\nIndexed files:")
    for f in changed_files:
        print(f"  ✓ {f}")

    db_conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index vault markdown into ChromaDB for semantic search."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Full reindex (ignore hashes, re-embed everything)",
    )
    args = parser.parse_args()
    run_index(force=args.force)
