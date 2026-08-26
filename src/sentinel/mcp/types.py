from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LaceMemoryQuery(BaseModel):
    query: str
    scope: str = "auto"
    category: str = "decision"
    max_results: int = 5


class LaceRememberPayload(BaseModel):
    content: str
    category: Literal["pattern", "decision", "debug", "reference", "preference"] = "decision"
    tags: list[str] = Field(default_factory=list)
    scope: str = "auto"
