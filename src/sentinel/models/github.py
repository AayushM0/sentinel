"""Pydantic data models for GitHub API payloads."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PRMetadata(BaseModel):
    """Metadata for a GitHub Pull Request."""

    number: int
    title: str
    body: str | None = None
    author: str = Field(alias="user_login")
    state: str  # "open" | "closed" | "merged"
    head_sha: str = Field(alias="head_sha")
    base_branch: str
    head_branch: str

    model_config = ConfigDict(populate_by_name=True)


class PRFile(BaseModel):
    """A file changed in a Pull Request."""

    filename: str
    status: str  # "added" | "modified" | "removed" | "renamed"
    patch: str | None = None
    additions: int = 0
    deletions: int = 0


class ReviewComment(BaseModel):
    """A review comment posted on a Pull Request."""

    id: int
    html_url: str
    body: str
    event: str  # "APPROVE" | "REQUEST_CHANGES" | "COMMENT"


class CheckRun(BaseModel):
    """A CI check run on a Pull Request."""

    name: str
    status: str  # "queued" | "in_progress" | "completed"
    conclusion: str | None = None  # "success" | "failure" | "cancelled" | "skipped"
    output_summary: str | None = None


class SecurityVulnerability(BaseModel):
    """A single security vulnerability found by audit tools."""

    package: str
    installed_version: str
    fixed_version: str | None = None
    severity: str  # "critical" | "high" | "medium" | "low"
    advisory: str
    url: str | None = None


class SecurityAuditResult(BaseModel):
    """Result from a security audit tool."""

    tool: str  # "pip-audit" | "npm-audit" | "ruff"
    vulnerabilities: list[SecurityVulnerability] = Field(default_factory=list)
    summary: str
    exit_code: int


if __name__ == "__main__":
    # Rule 2903681 self-checks
    pr = PRMetadata(
        number=1,
        title="Test PR",
        body=None,
        author="test",
        state="open",
        head_sha="abc",
        base_branch="main",
        head_branch="feat/test",
    )
    assert pr.number == 1
    assert pr.body is None

    f = PRFile(filename="test.py", status="added", patch="+test", additions=1, deletions=0)
    assert f.filename == "test.py"

    rc = ReviewComment(id=1, html_url="https://example.com", body="LGTM", event="APPROVE")
    assert rc.event == "APPROVE"

    cr = CheckRun(name="CI", status="completed", conclusion="success", output_summary="OK")
    assert cr.conclusion == "success"

    vuln = SecurityVulnerability(
        package="pkg",
        installed_version="1.0",
        fixed_version="1.1",
        severity="high",
        advisory="CVE-123",
        url="https://example.com",
    )
    assert vuln.severity == "high"

    result = SecurityAuditResult(
        tool="pip-audit", vulnerabilities=[vuln], summary="1 vuln", exit_code=1
    )
    assert result.tool == "pip-audit"

    print("github.py self-check passed.")
