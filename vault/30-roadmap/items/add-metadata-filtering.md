---
title: Add metadata filtering to search
status: planned
project: semantic-search-system
priority: high
---

# Add metadata filtering to search

## Why

Right now search.py returns results from the entire vault. If I want to search only active projects, or only reference notes, I have to manually filter the output. ChromaDB supports pre-filter on metadata fields before vector scoring — this is the right place to add it.

The value: "what are my open questions about the blog?" should be able to filter `status: planned` and `project: personal-blog` before doing the semantic search, so it doesn't surface completed-project notes that happen to mention blogging.

## Approach

Add optional CLI flags to search.py:

```
python brain/search.py "open questions" --status planned
python brain/search.py "architecture decisions" --tags rag,architecture
python brain/search.py "lessons learned" --source 20-projects/
```

ChromaDB's `where` parameter supports equality filters and `$in` for list membership. The implementation is a dict built from whichever flags are set:

```python
where = {}
if args.status:
    where["status"] = args.status
if args.tags:
    where["tags"] = {"$in": args.tags.split(",")}
```

Pass `where` to `collection.query()` when non-empty. When empty, omit it entirely (ChromaDB requires `where` to be non-empty if present).

One edge case: the `tags` field is stored as a comma-joined string (`"rag, embeddings, python"`) so the `$in` filter won't work directly — need to either store tags as a list (ChromaDB doesn't support list metadata) or use a `$contains` substring match, or normalize tags at index time into separate boolean fields.
