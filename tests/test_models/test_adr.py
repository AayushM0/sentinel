import pytest

from sentinel.models.adr import ADR


def test_madr_parse_from_markdown(sample_madr_markdown: str):
    adr = ADR.from_markdown(sample_madr_markdown)
    assert adr.id == "ADR-014"
    assert adr.title == "Encrypted State Persistence Policy"
    assert adr.status == "accepted"
    assert adr.category == "decision"
    assert adr.scope == "global"
    assert len(adr.tags) == 3
    assert "storage" in adr.tags
    assert len(adr.constraints) == 2
    assert "NEVER use raw window.localStorage for auth tokens" in adr.constraints
    assert adr.code_pattern == "SecureEncryptedStore.setItem(key, value)"
    assert adr.confidence == 0.95
    assert "# Context & Problem Statement" in adr.body
    assert "# Decision Outcome" in adr.body


def test_madr_to_markdown_roundtrip(sample_madr_markdown: str):
    adr = ADR.from_markdown(sample_madr_markdown)
    serialized_md = adr.to_markdown()

    # Re-parse serialized markdown
    re_parsed = ADR.from_markdown(serialized_md)
    assert re_parsed.id == adr.id
    assert re_parsed.title == adr.title
    assert re_parsed.status == adr.status
    assert re_parsed.constraints == adr.constraints
    assert re_parsed.code_pattern == adr.code_pattern
    assert re_parsed.body.strip() == adr.body.strip()


def test_madr_minimal_defaults():
    raw_md = """---
title: "Use Fast Path"
---
Just use fast path.
"""
    adr = ADR.from_markdown(raw_md)
    assert adr.title == "Use Fast Path"
    assert adr.status == "accepted"
    assert adr.category == "decision"
    assert adr.tags == []
    assert adr.constraints == []
    assert adr.body.strip() == "Just use fast path."


def test_madr_invalid_markdown():
    with pytest.raises(ValueError, match="Invalid MADR format"):
        ADR.from_markdown("No frontmatter here at all")


def test_madr_proposed_and_rejected_statuses():
    proposed_md = """---
id: "ADR-098"
title: "Proposed Feature"
status: "proposed"
---
Discussion ongoing.
"""
    adr_p = ADR.from_markdown(proposed_md)
    assert adr_p.status == "proposed"

    rejected_md = """---
id: "ADR-099"
title: "Rejected Feature"
status: "rejected"
---
Rejected after architecture review.
"""
    adr_r = ADR.from_markdown(rejected_md)
    assert adr_r.status == "rejected"
