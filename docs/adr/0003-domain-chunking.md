# ADR-0003 — Chunking per document shape; stdlib `ast` before tree-sitter

**Status:** accepted

## Decision
- Prose: heading-aware paragraph packing (~400 tokens, ~60 overlap) with the section heading
  prepended; sentence fallback for oversized paragraphs; PDF page numbers kept.
- Transcripts: whole speaker turns only, packed to ~350 tokens with a one-turn overlap; speakers
  and start/end timestamps in metadata. A test asserts no turn is ever split.
- Resumes/JDs: section-aware (Skills, Experience, Requirements, …).
- Code: Python spans from the stdlib `ast` (exact function/class boundaries incl. decorators);
  other languages use a top-level-definition heuristic; every file also gets an overview chunk
  (path, symbols, imports) so "where is X" questions have a precise target.

## Why not tree-sitter now
Precise for every language, but adds native wheels and grammar packages to the image and more
code to maintain. The heuristic is good enough for TS/Angular files in this repo; tree-sitter is
the first upgrade if the Code agent is pointed at large polyglot repos.

## Consequences
+ Citations are meaningful (page/section, file::symbol lines, speaker@timestamp).
− Token counts are approximated (~1.3 tokens/word) rather than tokenized per model.
