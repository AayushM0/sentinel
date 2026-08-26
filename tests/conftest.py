import pytest


@pytest.fixture
def sample_madr_markdown() -> str:
    return """---
id: "ADR-014"
title: "Encrypted State Persistence Policy"
status: "accepted"
date: "2026-08-12"
superseded_by: null
category: "decision"
tags:
  - "storage"
  - "security"
  - "auth"
scope: "global"
constraints:
  - "NEVER use raw window.localStorage for auth tokens"
  - "MUST use SecureEncryptedStore wrapper"
code_pattern: "SecureEncryptedStore.setItem(key, value)"
confidence: 0.95
---

# Context & Problem Statement
Direct access to browser localStorage exposes authentication tokens.

# Decision Outcome
All persistent state must be routed through SecureEncryptedStore.
"""
