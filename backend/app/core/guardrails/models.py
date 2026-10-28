from typing import Literal

from pydantic import BaseModel

Action = Literal["pass", "flag", "redact", "block"]


class GuardrailResult(BaseModel):
    name: str
    stage: Literal["input", "retrieval", "output"]
    action: Action
    detail: str = ""

    @property
    def blocked(self) -> bool:
        return self.action == "block"
