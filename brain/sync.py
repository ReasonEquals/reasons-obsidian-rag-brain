#!/usr/bin/env python3
"""
sync.py — Vault → SQLite sync pipeline.

Reads markdown files from the vault and upserts their frontmatter data
into brain.db. Markdown is always the source of truth — the database is
a derived, queryable mirror that can be rebuilt at any time.

Usage:
    python brain/sync.py --once    Single pass then exit
    python brain/sync.py --watch   Re-run on every file save (local dev)

Environment variables (via .env or shell):
    VAULT_PATH   Path to the vault directory (default: ~/vault)
    DB_PATH      Path to the SQLite database file (default: ~/vault/_data/brain.db)
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import frontmatter
except ImportError:
    sys.exit("Error: python-frontmatter is not installed.\nRun: pip install python-frontmatter")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VAULT_PATH = Path(os.environ.get("VAULT_PATH", "~/vault")).expanduser()
DB_PATH    = Path(os.environ.get("DB_PATH", "~/vault/_data/brain.db")).expanduser()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_write_conn(db_path: Path) -> sqlite3.Connection:
    """Open a read-write connection with safe PRAGMA settings."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist. Idempotent — safe to call on every run."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            slug        TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'planned',
            started     TEXT,
            completed   TEXT,
            summary     TEXT,
            repo        TEXT,
            url         TEXT,
            tags        TEXT NOT NULL DEFAULT '[]',
            body_md     TEXT,
            synced_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS roadmap_items (
            slug        TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'planned',
            project     TEXT REFERENCES projects(slug) ON DELETE SET NULL,
            priority    TEXT NOT NULL DEFAULT 'medium',
            due         TEXT,
            body_md     TEXT,
            synced_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daily_notes (
            date        TEXT PRIMARY KEY,
            word_count  INTEGER NOT NULL DEFAULT 0,
            body_md     TEXT,
            synced_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sync_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at      TEXT NOT NULL,
            file_path   TEXT NOT NULL,
            action      TEXT NOT NULL,
            note        TEXT
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug_from_path(path: Path) -> str:
    return path.stem


def tags_list(raw) -> list[str]:
    """Normalize tags: handles list, space-separated string, or None."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t).lstrip("#") for t in raw]
    if isinstance(raw, str):
        return [t.lstrip("#") for t in raw.split()]
    return []


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def slugify(text: str) -> str:
    """Convert a display name to a kebab-case slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s]+", "-", text)
    return text.strip("-")


# ---------------------------------------------------------------------------
# Sync functions
# ---------------------------------------------------------------------------

def sync_projects(conn: sqlite3.Connection, vault: Path) -> list[dict]:
    """Sync 20-projects/*.md → projects table.

    Demonstrates: frontmatter parsing, upsert-on-conflict, orphan cleanup
    (rows deleted from DB when corresponding markdown file is removed).
    Markdown is source of truth — the DB row lives and dies with the file.
    """
    results = []
    project_dir = vault / "20-projects"
    if not project_dir.exists():
        return results

    seen_slugs = set()

    for md_file in sorted(project_dir.glob("*.md")):
        slug = slug_from_path(md_file)
        seen_slugs.add(slug)
        try:
            post = frontmatter.load(str(md_file))
        except Exception as e:
            results.append({"file": str(md_file), "action": "skip", "note": str(e)})
            continue

        fm = post.metadata
        tag_list = tags_list(fm.get("tags"))

        conn.execute(
            """
            INSERT INTO projects
                (slug, title, status, started, completed, summary, repo, url,
                 tags, body_md, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                title     = excluded.title,
                status    = excluded.status,
                started   = excluded.started,
                completed = excluded.completed,
                summary   = excluded.summary,
                repo      = excluded.repo,
                url       = excluded.url,
                tags      = excluded.tags,
                body_md   = excluded.body_md,
                synced_at = excluded.synced_at
            """,
            (
                slug,
                fm.get("title") or slug,
                fm.get("status") or "planned",
                fm.get("started") or None,
                fm.get("completed") or None,
                fm.get("summary") or None,
                fm.get("repo") or None,
                fm.get("url") or None,
                json.dumps(tag_list),
                post.content or None,
                now_iso(),
            ),
        )
        results.append({"file": str(md_file), "action": "upsert", "note": None})

    # Orphan cleanup — markdown is source of truth.
    # If a file is deleted, remove its row from the DB.
    existing_slugs = {
        row[0] for row in conn.execute("SELECT slug FROM projects").fetchall()
    }
    for gone_slug in existing_slugs - seen_slugs:
        conn.execute("DELETE FROM projects WHERE slug = ?", (gone_slug,))
        results.append({"file": gone_slug, "action": "delete", "note": "file removed"})

    return results


def sync_daily_notes(conn: sqlite3.Connection, vault: Path) -> list[dict]:
    """Sync 10-daily/YYYY-MM-DD.md → daily_notes table."""
    results = []
    daily_dir = vault / "10-daily"
    if not daily_dir.exists():
        return results

    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    for md_file in sorted(daily_dir.glob("*.md")):
        slug = slug_from_path(md_file)
        if not date_pattern.match(slug):
            continue  # skip templates or other non-date files

        try:
            post = frontmatter.load(str(md_file))
        except Exception as e:
            results.append({"file": str(md_file), "action": "skip", "note": str(e)})
            continue

        conn.execute(
            """
            INSERT INTO daily_notes (date, word_count, body_md, synced_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                word_count = excluded.word_count,
                body_md    = excluded.body_md,
                synced_at  = excluded.synced_at
            """,
            (
                slug,
                word_count(post.content),
                post.content or None,
                now_iso(),
            ),
        )
        results.append({"file": str(md_file), "action": "upsert", "note": None})

    return results


def sync_roadmap(conn: sqlite3.Connection, vault: Path) -> list[dict]:
    """Sync 30-roadmap/items/*.md → roadmap_items table."""
    results = []
    roadmap_dir = vault / "30-roadmap" / "items"
    if not roadmap_dir.exists():
        return results

    seen_slugs = set()

    for md_file in sorted(roadmap_dir.glob("*.md")):
        slug = slug_from_path(md_file)
        seen_slugs.add(slug)
        try:
            post = frontmatter.load(str(md_file))
        except Exception as e:
            results.append({"file": str(md_file), "action": "skip", "note": str(e)})
            continue

        fm = post.metadata
        # project field is a FK reference to projects(slug) — use raw value
        raw_project = fm.get("project") or None

        conn.execute(
            """
            INSERT INTO roadmap_items
                (slug, title, status, project, priority, due, body_md, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                title     = excluded.title,
                status    = excluded.status,
                project   = excluded.project,
                priority  = excluded.priority,
                due       = excluded.due,
                body_md   = excluded.body_md,
                synced_at = excluded.synced_at
            """,
            (
                slug,
                fm.get("title") or slug,
                fm.get("status") or "planned",
                raw_project,
                fm.get("priority") or "medium",
                fm.get("due") or None,
                post.content or None,
                now_iso(),
            ),
        )
        results.append({"file": str(md_file), "action": "upsert", "note": None})

    # Orphan cleanup
    existing_slugs = {
        row[0] for row in conn.execute("SELECT slug FROM roadmap_items").fetchall()
    }
    for gone_slug in existing_slugs - seen_slugs:
        conn.execute("DELETE FROM roadmap_items WHERE slug = ?", (gone_slug,))
        results.append({"file": gone_slug, "action": "delete", "note": "file removed"})

    return results


# ---------------------------------------------------------------------------
# Sync log
# ---------------------------------------------------------------------------

def write_sync_log(conn: sqlite3.Connection, results: list[dict]) -> None:
    run_at = now_iso()
    conn.executemany(
        "INSERT INTO sync_log (run_at, file_path, action, note) VALUES (?, ?, ?, ?)",
        [(run_at, r["file"], r["action"], r["note"]) for r in results],
    )


# ---------------------------------------------------------------------------
# Main sync pass
# ---------------------------------------------------------------------------

def run_sync() -> None:
    if not VAULT_PATH or not VAULT_PATH.exists():
        sys.exit(f"Error: VAULT_PATH not set or does not exist: {VAULT_PATH!r}")
    if not DB_PATH:
        sys.exit("Error: DB_PATH is not set.")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_write_conn(DB_PATH)
    ensure_schema(conn)

    all_results: list[dict] = []
    all_results += sync_projects(conn, VAULT_PATH)
    all_results += sync_daily_notes(conn, VAULT_PATH)
    all_results += sync_roadmap(conn, VAULT_PATH)

    write_sync_log(conn, all_results)
    conn.commit()
    conn.close()

    upserted = sum(1 for r in all_results if r["action"] == "upsert")
    skipped  = sum(1 for r in all_results if r["action"] == "skip")
    deleted  = sum(1 for r in all_results if r["action"] == "delete")

    print(f"✓ Sync complete: {upserted} upserted, {skipped} skipped, {deleted} deleted | DB: {DB_PATH}")
    for r in all_results:
        if r["action"] == "skip":
            print(f"  ⚠ Skipped {r['file']}: {r['note']}")


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def run_watch() -> None:
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        sys.exit("Error: watchdog is not installed.\nRun: pip install watchdog")

    import time

    class VaultHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if event.src_path.endswith(".md"):
                print(f"\n→ Changed: {event.src_path}")
                run_sync()

        def on_created(self, event):
            if event.src_path.endswith(".md"):
                print(f"\n→ Created: {event.src_path}")
                run_sync()

        def on_deleted(self, event):
            if event.src_path.endswith(".md"):
                print(f"\n→ Deleted: {event.src_path}")
                run_sync()

    observer = Observer()
    observer.schedule(VaultHandler(), str(VAULT_PATH), recursive=True)
    observer.start()
    print(f"Watching {VAULT_PATH} for changes (Ctrl+C to stop)…")
    run_sync()  # initial pass on start
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync vault markdown → SQLite")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once",  action="store_true", help="Single pass then exit")
    group.add_argument("--watch", action="store_true", help="Watch mode (re-run on file save)")
    args = parser.parse_args()

    if args.once:
        run_sync()
    else:
        run_watch()
