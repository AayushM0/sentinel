from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Regex to match YAML frontmatter between --- delimiters
_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


class ADR(BaseModel):
    """Architectural Decision Record model (MADR format)."""

    id: str = ""
    title: str
    status: Literal["proposed", "accepted", "rejected", "deprecated", "superseded"] = "accepted"
    date: str = Field(default_factory=lambda: datetime.now(UTC).date().isoformat())

    superseded_by: str | None = None
    category: str = "decision"
    tags: list[str] = Field(default_factory=list)
    scope: str = "project"
    constraints: list[str] = Field(default_factory=list)
    code_pattern: str | None = None
    confidence: float = 1.0
    body: str = ""

    @classmethod
    def from_markdown(cls, raw_md: str) -> ADR:
        """Parse MADR markdown containing YAML frontmatter and markdown body."""
        match = _FRONTMATTER_PATTERN.match(raw_md.strip())
        if not match:
            raise ValueError("Invalid MADR format: missing YAML frontmatter (--- delimiters)")

        frontmatter_str, body = match.groups()
        data = yaml.safe_load(frontmatter_str) or {}
        if not isinstance(data, dict):
            raise TypeError("Invalid MADR frontmatter: must be a YAML dictionary")

        return cls(**data, body=body.strip())

    def to_markdown(self) -> str:
        """Serialize ADR to Markdown string with YAML frontmatter."""
        meta = {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "date": self.date,
            "superseded_by": self.superseded_by,
            "category": self.category,
            "tags": self.tags,
            "scope": self.scope,
            "constraints": self.constraints,
            "code_pattern": self.code_pattern,
            "confidence": self.confidence,
        }
        # Filter out keys with None or empty values if not id/title/status
        cleaned_meta = {k: v for k, v in meta.items() if v is not None}
        frontmatter = yaml.safe_dump(cleaned_meta, sort_keys=False, allow_unicode=True)
        return f"---\n{frontmatter}---\n\n{self.body}\n"


if __name__ == "__main__":
    # Framework-free self-check (Rule 2903681)
    adr = ADR(id="ADR-001", title="Test Policy", status="accepted", body="Policy body.")
    md = adr.to_markdown()
    reparsed = ADR.from_markdown(md)
    assert reparsed.id == "ADR-001"
    assert reparsed.title == "Test Policy"
    assert reparsed.body == "Policy body."
    print("adr.py standalone self-check passed successfully.")

