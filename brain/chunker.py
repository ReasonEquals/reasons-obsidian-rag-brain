#!/usr/bin/env python3
"""Markdown-aware chunking for vault semantic search.

Splits vault markdown files on H2 (##) headers, producing chunks that
align with the natural semantic boundaries in the notes. Frontmatter
becomes metadata (for filtering), not embedded text (which would dilute
semantic signal).

Usage (standalone test):
    python chunker.py [path-to-markdown-file]
    python chunker.py  # defaults to vault/40-reference/rag-architecture-notes.md
"""

import re
import sys
from pathlib import Path

import frontmatter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Chunks with fewer words than this get merged into the next chunk.
# Why: A heading like "## Status" with one word underneath is useless as a
# standalone embedding — it has no semantic content. Merging it with the
# next section gives the embedding model enough text to work with.
MIN_CHUNK_WORDS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Turn a heading into a URL-safe slug for chunk IDs.

    'What it is' -> 'what-it-is'
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


def word_count(text: str) -> int:
    """Split-based word count. Good enough for thresholding."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Core chunking
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, body_text).

    Frontmatter is structured data (title, status, tags, dates). We store
    it as ChromaDB metadata for filtering — NOT in the embedded text.

    Why? Embedding "status: active, tags: ai, web" alongside prose about
    the project would confuse the model. The embedding should capture what
    the text is *about*, and metadata fields should be used for structured
    filtering (e.g., "show me only active projects").
    """
    post = frontmatter.loads(text)

    # Normalize metadata values to strings (ChromaDB metadata must be
    # str, int, float, or bool — no lists, no None)
    meta = {}
    for key, value in post.metadata.items():
        if isinstance(value, list):
            meta[key] = ", ".join(str(v) for v in value)
        elif value is None:
            continue  # skip nulls entirely
        else:
            meta[key] = str(value)

    return meta, post.content


def chunk_markdown(text: str, file_path: str) -> list[dict]:
    """Split a markdown file into chunks on H2 boundaries.

    Returns a list of chunk dicts, each with:
        - id:       deterministic ID like "20-projects/vault-search.md::status"
        - text:     the chunk content (what gets embedded)
        - metadata: frontmatter fields + source/section info for ChromaDB

    Strategy:
        1. Strip frontmatter → metadata dict + body text
        2. Split body on H2 headers (## ...)
        3. Each section becomes a chunk, with the H1 title prepended
           (so the embedding knows what file this section belongs to)
        4. Sections shorter than MIN_CHUNK_WORDS merge into the next one
        5. Files with no H2 headers become a single chunk

    Why H2, not H3 or paragraphs?
        - H2 is the primary structural marker in this vault
        - H3 splits would create too-small chunks (most vault notes don't use H3)
        - Paragraph splits would lose the heading context that tells you
          *what section* you're reading
        - Fixed token windows (128/256/512) ignore document structure entirely —
          they'd split mid-sentence or mid-section, producing chunks where the
          embedding captures half of two different topics instead of one complete one
    """
    metadata, body = parse_frontmatter(text)

    # FIX: Empty body after frontmatter extraction → no chunks to embed.
    # A file that is only frontmatter (e.g. a project stub with no prose)
    # would produce an empty embedding, wasting an API call and adding
    # noise to search results.
    if not body.strip():
        return []

    # Relative path from vault root for display and IDs
    rel_path = file_path  # caller should pass relative path

    # Extract H1 title (first # heading) — prepended to every chunk
    # so the embedding knows the document context
    h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    doc_title = h1_match.group(1).strip() if h1_match else Path(rel_path).stem

    # Split on H2 headers. re.split keeps the delimiter if it's in a group.
    # Pattern: match lines starting with ## (but not ### or more)
    parts = re.split(r"^(##\s+.+)$", body, flags=re.MULTILINE)

    # Build raw sections: list of (heading, content) tuples
    # parts[0] is text before the first H2 (intro/H1 area)
    # Then alternating: [heading, content, heading, content, ...]
    sections: list[tuple[str, str]] = []

    # Intro section (before first H2)
    intro = parts[0].strip()
    # Remove the H1 line from intro (we use doc_title separately)
    intro = re.sub(r"^#\s+.+\n*", "", intro).strip()
    if intro:
        sections.append(("intro", intro))

    # Remaining sections come in pairs: heading, content
    for i in range(1, len(parts), 2):
        heading_line = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Extract heading text (strip the ## prefix)
        heading = re.sub(r"^##\s+", "", heading_line).strip()
        sections.append((heading, content))

    # Merge short sections into the next one
    merged: list[tuple[str, str]] = []
    carry_heading = None
    carry_text = ""

    for heading, content in sections:
        if carry_heading is not None:
            # We're carrying a short section — merge into this one
            combined_text = f"{carry_text}\n\n{heading}\n{content}".strip()
            heading = carry_heading  # keep the first heading as the label
            content = combined_text
            carry_heading = None
            carry_text = ""

        if word_count(content) < MIN_CHUNK_WORDS:
            # Too short — carry forward
            carry_heading = heading
            carry_text = content
        else:
            merged.append((heading, content))

    # If the last section was short and got carried, append it anyway
    if carry_heading is not None:
        if merged:
            # Merge into the previous section
            prev_heading, prev_content = merged[-1]
            merged[-1] = (prev_heading, f"{prev_content}\n\n{carry_heading}\n{carry_text}".strip())
        else:
            # Only section in the file
            merged.append((carry_heading, carry_text))

    # If no sections at all (no H2s, no intro), use the whole body
    if not merged:
        merged.append(("full", body.strip()))

    # Build chunk dicts
    # FIX: Track slugs to handle duplicate H2 headings (e.g., two "## Status"
    # sections in the same file). Without deduplication, the second chunk would
    # silently overwrite the first in ChromaDB, causing data loss.
    chunks = []
    slug_counts: dict[str, int] = {}

    for heading, content in merged:
        slug = slugify(heading)

        # Deduplicate: first "status" stays as-is, second becomes "status-2"
        slug_counts[slug] = slug_counts.get(slug, 0) + 1
        if slug_counts[slug] > 1:
            slug = f"{slug}-{slug_counts[slug]}"

        chunk_id = f"{rel_path}::{slug}"

        # Prepend doc title so the embedding has file-level context
        # "RAG Architecture — Chunking strategies" is more meaningful than just "Chunking strategies"
        if heading in ("intro", "full"):
            chunk_text = f"{doc_title}\n\n{content}"
        else:
            chunk_text = f"{doc_title} — {heading}\n\n{content}"

        chunk_meta = {
            **metadata,
            "source": rel_path,
            "section": heading,
            "doc_title": doc_title,
        }

        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": chunk_meta,
        })

    return chunks


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    sample_vault = script_dir.parent / "vault"

    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1]).expanduser()
        try:
            vault_path = test_file.parent.parent.parent  # best-effort for vault root
        except Exception:
            vault_path = test_file.parent
    else:
        vault_path = sample_vault
        test_file = vault_path / "40-reference" / "rag-architecture-notes.md"

    if not test_file.exists():
        print(f"File not found: {test_file}")
        sys.exit(1)

    text = test_file.read_text()
    try:
        rel = str(test_file.relative_to(vault_path))
    except ValueError:
        rel = test_file.name

    chunks = chunk_markdown(text, rel)

    print(f"\n{'='*60}")
    print(f"File: {rel}")
    print(f"Chunks: {len(chunks)}")
    print(f"{'='*60}\n")

    for chunk in chunks:
        preview = chunk["text"][:120].replace("\n", " ")
        wc = word_count(chunk["text"])
        print(f"  [{chunk['id']}]")
        print(f"  {wc} words | {preview}...")
        print(f"  meta: {chunk['metadata']}")
        print()
