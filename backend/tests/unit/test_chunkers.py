from app.core.ingestion.chunkers import (
    CodeChunker,
    RecursiveChunker,
    SectionChunker,
    TranscriptChunker,
    parse_transcript,
)
from app.core.ingestion.models import LoadedDocument, Section
from tests.conftest import METFORMIN, TRANSCRIPT


def _doc(text: str, **meta: str) -> LoadedDocument:
    return LoadedDocument(
        source="s", title="T", doc_type="text", sections=[Section(text=text)], metadata=dict(meta)
    )


def test_recursive_chunks_carry_their_heading() -> None:
    chunks = RecursiveChunker(target_tokens=40).chunk(_doc(METFORMIN), "system", {})
    contraindication = next(c for c in chunks if "renal" in c.text)
    assert contraindication.metadata["section"] == "Contraindications"
    assert contraindication.text.startswith("# Contraindications")


def test_recursive_respects_target_size() -> None:
    text = "\n\n".join(f"Paragraph {i} " + "word " * 50 for i in range(10))
    chunks = RecursiveChunker(target_tokens=120, overlap_tokens=0).chunk(_doc(text), "o", {})
    assert len(chunks) > 3
    assert all(len(c.text.split()) < 200 for c in chunks)


def test_transcript_parser_handles_continuations() -> None:
    turns = parse_transcript("[00:00:01] A: hello\ncontinued line\n[00:00:09] B: hi")
    assert turns == [
        {"ts": "00:00:01", "speaker": "A", "text": "hello continued line"},
        {"ts": "00:00:09", "speaker": "B", "text": "hi"},
    ]


def test_transcript_never_splits_a_turn() -> None:
    chunks = TranscriptChunker(target_tokens=12, overlap_turns=0).chunk(_doc(TRANSCRIPT), "o", {})
    turns = parse_transcript(TRANSCRIPT)
    for turn in turns:
        assert any(turn["text"] in c.text for c in chunks)  # every turn intact in some chunk
    assert chunks[0].metadata["ts_start"] == "00:00:05"
    assert "Dr. Rao" in chunks[0].metadata["speakers"]


def test_python_code_chunked_on_symbols() -> None:
    source = "import os\n\n\ndef alpha():\n    return 1\n\n\nclass Beta:\n    x = 2\n"
    doc = _doc(source, path="pkg/mod.py", lang="py")
    chunks = CodeChunker().chunk(doc, "o", {})
    symbols = [c.metadata["symbol"] for c in chunks]
    assert symbols == ["<file>", "alpha", "Beta"]
    assert chunks[1].metadata["line_start"] == 4
    assert "Symbols: alpha, Beta" in chunks[0].text


def test_typescript_heuristic_spans() -> None:
    source = "export class ChatService {\n  send() {}\n}\n\nexport function helper() {}\n"
    chunks = CodeChunker().chunk(_doc(source, path="chat.service.ts", lang="ts"), "o", {})
    assert {"ChatService", "helper"} <= {c.metadata["symbol"] for c in chunks}


def test_section_chunker_splits_resume_sections() -> None:
    resume = "Jane Doe\n\nSkills\nPython, FastAPI\n\nExperience\nBuilt RAG systems at Acme."
    chunks = SectionChunker().chunk(_doc(resume), "o", {"doc_role": "resume"})
    sections = [c.metadata["section"] for c in chunks]
    assert sections == ["Header", "Skills", "Experience"]
    assert chunks[1].metadata["doc_role"] == "resume"
