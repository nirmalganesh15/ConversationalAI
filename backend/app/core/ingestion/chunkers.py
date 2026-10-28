"""Domain-specific chunking strategies.

One size does not fit all: prose wants heading-aware windows with overlap, code wants
symbol boundaries, transcripts must never split a speaker turn, and resumes/JDs are
naturally sectioned. Each strategy emits Chunks with metadata the agents can cite.
"""

import ast
import hashlib
import re
from typing import Any, Protocol

from app.core.ingestion.models import LoadedDocument
from app.core.retrieval.models import Chunk


def approx_tokens(text: str) -> int:
    # ~1.3 tokens per English word; good enough for budgeting without a tokenizer call.
    return int(len(text.split()) * 1.3) + 1


def chunk_id(owner: str, source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(
        f"{owner}|{source}|{index}|{text[:64]}".encode(), usedforsecurity=False
    ).hexdigest()
    return digest[:20]


class Chunker(Protocol):
    def chunk(self, doc: LoadedDocument, owner: str, extra: dict[str, Any]) -> list[Chunk]: ...


def _base_meta(doc: LoadedDocument, owner: str, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": doc.source,
        "title": doc.title,
        "doc_type": doc.doc_type,
        "owner": owner,
        **doc.metadata,
        **extra,
    }


# --- prose ----------------------------------------------------------------------
_HEADING = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 ,/&\-\(\)]{3,80}:?)$")


class RecursiveChunker:
    """Heading-aware paragraph packing with token overlap.

    Splits on blank lines, tracks the most recent heading so every chunk carries its
    section title (and prepends it, which measurably helps both BM25 and embeddings
    on drug labels where the answer text rarely repeats the section name).
    """

    def __init__(self, target_tokens: int = 400, overlap_tokens: int = 60) -> None:
        self.target = target_tokens
        self.overlap = overlap_tokens

    def _paragraphs(self, text: str) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        out: list[str] = []
        for p in paras:
            if approx_tokens(p) <= self.target:
                out.append(p)
                continue
            # Oversized paragraph: fall back to sentence packing.
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf: list[str] = []
            for s in sentences:
                if buf and approx_tokens(" ".join([*buf, s])) > self.target:
                    out.append(" ".join(buf))
                    buf = []
                buf.append(s)
            if buf:
                out.append(" ".join(buf))
        return out

    def chunk(self, doc: LoadedDocument, owner: str, extra: dict[str, Any]) -> list[Chunk]:
        base = _base_meta(doc, owner, extra)
        chunks: list[Chunk] = []
        heading = ""
        for section in doc.sections:
            buf: list[str] = []
            for para in self._paragraphs(section.text):
                lines = para.splitlines()
                if len(lines) <= 2 and _HEADING.match(lines[0].strip()):
                    # New section: close the current window without overlap.
                    self._emit(chunks, buf, heading, section.page, base, owner, doc.source)
                    heading = lines[0].strip()
                    buf = lines[1:]
                    continue
                if buf and approx_tokens("\n\n".join([*buf, para])) > self.target:
                    self._emit(chunks, buf, heading, section.page, base, owner, doc.source)
                    buf = self._overlap_tail(buf)
                buf.append(para)
            self._emit(chunks, buf, heading, section.page, base, owner, doc.source)
        return chunks

    def _overlap_tail(self, buf: list[str]) -> list[str]:
        tail: list[str] = []
        for para in reversed(buf):
            if approx_tokens(" ".join(tail)) >= self.overlap:
                break
            tail.insert(0, para)
        return tail if len(tail) < len(buf) else []

    @staticmethod
    def _emit(
        chunks: list[Chunk],
        buf: list[str],
        heading: str,
        page: int | None,
        base: dict[str, Any],
        owner: str,
        source: str,
    ) -> None:
        if not buf:
            return
        body = "\n\n".join(buf)
        text = f"{heading}\n{body}" if heading else body
        idx = len(chunks)
        meta = {**base, "section": heading.lstrip("# ").strip(), "chunk_index": idx}
        if page is not None:
            meta["page"] = page
        chunks.append(Chunk(id=chunk_id(owner, source, idx, text), text=text, metadata=meta))


# --- code ------------------------------------------------------------------------
_TOP_LEVEL_DEF = re.compile(
    r"^(export\s+)?(default\s+)?(async\s+)?"
    r"(function|class|interface|type|enum|const|def|func|public|private|protected)\b",
)


class CodeChunker:
    """Symbol-level chunks + one file-overview chunk per file.

    Python uses the stdlib `ast` for exact function/class boundaries; other languages
    use a top-level-definition heuristic with a line-window fallback. Tree-sitter would
    be more precise but adds native deps to the image - see ADR-0003.
    """

    def __init__(self, max_lines: int = 80) -> None:
        self.max_lines = max_lines

    def chunk(self, doc: LoadedDocument, owner: str, extra: dict[str, Any]) -> list[Chunk]:
        base = _base_meta(doc, owner, extra)
        source = doc.full_text
        lines = source.splitlines()
        path = doc.metadata.get("path", doc.source)
        lang = doc.metadata.get("lang", "")

        spans = self._python_spans(source) if lang == "py" else self._heuristic_spans(lines)
        chunks: list[Chunk] = []

        symbols = [name for name, _, _ in spans]
        imports = [ln.strip() for ln in lines[:60] if re.match(r"^\s*(import|from)\s", ln)][:25]
        overview = (
            f"File: {path}\nLanguage: {lang}\nLines: {len(lines)}\n"
            f"Symbols: {', '.join(symbols) or '(none)'}\nImports:\n" + "\n".join(imports)
        )
        chunks.append(self._make(owner, doc, base, 0, overview, "<file>", 1, len(lines)))

        for name, start, end in spans:
            for w_start in range(start, end + 1, self.max_lines):
                w_end = min(end, w_start + self.max_lines - 1)
                body = "\n".join(lines[w_start - 1 : w_end])
                text = f"# {path} :: {name} (lines {w_start}-{w_end})\n{body}"
                chunks.append(self._make(owner, doc, base, len(chunks), text, name, w_start, w_end))
        if not spans and len(lines) > 0:  # config files, markdown, etc.
            for w_start in range(1, len(lines) + 1, self.max_lines):
                w_end = min(len(lines), w_start + self.max_lines - 1)
                body = "\n".join(lines[w_start - 1 : w_end])
                text = f"# {path} (lines {w_start}-{w_end})\n{body}"
                chunks.append(
                    self._make(owner, doc, base, len(chunks), text, "<module>", w_start, w_end)
                )
        return chunks

    @staticmethod
    def _make(
        owner: str,
        doc: LoadedDocument,
        base: dict[str, Any],
        idx: int,
        text: str,
        symbol: str,
        start: int,
        end: int,
    ) -> Chunk:
        meta = {**base, "symbol": symbol, "line_start": start, "line_end": end, "chunk_index": idx}
        return Chunk(id=chunk_id(owner, doc.source, idx, text), text=text, metadata=meta)

    @staticmethod
    def _python_spans(source: str) -> list[tuple[str, int, int]]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        spans: list[tuple[str, int, int]] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                spans.append((node.name, start, node.end_lineno or node.lineno))
        return spans

    @staticmethod
    def _heuristic_spans(lines: list[str]) -> list[tuple[str, int, int]]:
        starts: list[tuple[str, int]] = []
        for i, line in enumerate(lines, start=1):
            if _TOP_LEVEL_DEF.match(line):
                name_match = re.search(
                    r"(?:function|class|interface|type|enum|const|def|func)\s+([A-Za-z_$][\w$]*)",
                    line,
                )
                starts.append((name_match.group(1) if name_match else line.strip()[:40], i))
        spans: list[tuple[str, int, int]] = []
        for idx, (name, start) in enumerate(starts):
            end = starts[idx + 1][1] - 1 if idx + 1 < len(starts) else len(lines)
            spans.append((name, start, end))
        return spans


# --- transcripts ----------------------------------------------------------------------
_TURN = re.compile(
    r"^\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-\u2013]?\s*(?P<speaker>[^:]{1,60}):\s*(?P<text>.*)$"
)


def parse_transcript(text: str) -> list[dict[str, str]]:
    """Parse `[hh:mm:ss] Speaker: text` lines; continuation lines join the previous turn."""
    turns: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.upper() == "WEBVTT" or "-->" in line:
            continue
        match = _TURN.match(line)
        if match:
            turns.append(
                {"ts": match["ts"], "speaker": match["speaker"].strip(), "text": match["text"]}
            )
        elif turns:
            turns[-1]["text"] += " " + line
    return turns


class TranscriptChunker:
    """Packs whole speaker turns into windows - a turn is never split across chunks."""

    def __init__(self, target_tokens: int = 350, overlap_turns: int = 1) -> None:
        self.target = target_tokens
        self.overlap_turns = overlap_turns

    def chunk(self, doc: LoadedDocument, owner: str, extra: dict[str, Any]) -> list[Chunk]:
        base = _base_meta(doc, owner, extra)
        turns = parse_transcript(doc.full_text)
        if not turns:
            return RecursiveChunker().chunk(doc, owner, extra)
        chunks: list[Chunk] = []
        window: list[dict[str, str]] = []

        def emit(window: list[dict[str, str]]) -> None:
            text = "\n".join(f"[{t['ts']}] {t['speaker']}: {t['text']}" for t in window)
            idx = len(chunks)
            meta = {
                **base,
                "meeting_id": doc.source,
                "speakers": sorted({t["speaker"] for t in window}),
                "ts_start": window[0]["ts"],
                "ts_end": window[-1]["ts"],
                "chunk_index": idx,
            }
            chunks.append(
                Chunk(id=chunk_id(owner, doc.source, idx, text), text=text, metadata=meta)
            )

        for turn in turns:
            candidate = [*window, turn]
            text_len = approx_tokens(" ".join(t["text"] for t in candidate))
            if window and text_len > self.target:
                emit(window)
                window = window[-self.overlap_turns :] if self.overlap_turns else []
            window.append(turn)
        if window:
            emit(window)
        return chunks


# --- resumes & job descriptions -----------------------------------------------------------
_SECTION_HINTS = re.compile(
    r"^(#+\s*)?(summary|profile|experience|work experience|employment|education|skills|"
    r"technical skills|projects|certifications|publications|responsibilities|requirements|"
    r"qualifications|what you.ll do|what you bring|about (us|the role)|bonus|nice to have|"
    r"preferred|benefits|what we offer|mission|your mission)\b.*$",
    re.IGNORECASE,
)


class SectionChunker:
    """Splits resumes / JDs on their natural sections so 'Skills' or 'Requirements'
    arrive as one coherent unit for gap analysis."""

    def __init__(self, max_tokens: int = 450) -> None:
        self.max_tokens = max_tokens

    def chunk(self, doc: LoadedDocument, owner: str, extra: dict[str, Any]) -> list[Chunk]:
        base = _base_meta(doc, owner, extra)
        sections: list[tuple[str, list[str]]] = [("Header", [])]
        for line in doc.full_text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) < 60 and _SECTION_HINTS.match(stripped):
                sections.append((stripped.lstrip("# ").rstrip(":"), []))
            else:
                sections[-1][1].append(line)

        chunks: list[Chunk] = []
        for name, body_lines in sections:
            body = "\n".join(body_lines).strip()
            if not body:
                continue
            # Long sections (e.g. experience) are sub-split on blank lines.
            parts = [body]
            if approx_tokens(body) > self.max_tokens:
                parts = RecursiveChunker(self.max_tokens, 40)._paragraphs(body)
            for part in parts:
                text = f"{doc.title} — {name}\n{part}"
                idx = len(chunks)
                meta = {**base, "section": name, "chunk_index": idx}
                chunks.append(
                    Chunk(id=chunk_id(owner, doc.source, idx, text), text=text, metadata=meta)
                )
        return chunks
