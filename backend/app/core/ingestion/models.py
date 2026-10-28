from typing import Any

from pydantic import BaseModel, Field


class Section(BaseModel):
    text: str
    page: int | None = None


class LoadedDocument(BaseModel):
    source: str  # stable identifier: filename or repo path
    title: str
    doc_type: str  # pdf | text | markdown | docx | code | transcript
    sections: list[Section]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)
