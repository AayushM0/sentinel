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


class LaceMemoryItem(BaseModel):
    id: str = ""
    title: str = ""
    category: str = "decision"
    content: str = ""
    tags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    scope: str = "project"
    raw: str | None = None


class LaceContextResponse(BaseModel):
    status: str = "active"
    project: str = ""
    cwd: str = ""
    instructions: str = ""
    message: str = ""
