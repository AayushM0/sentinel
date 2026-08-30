"""Tests for GitHub data models."""

from sentinel.models.github import (
    CheckRun,
    PRFile,
    PRMetadata,
    ReviewComment,
    SecurityAuditResult,
    SecurityVulnerability,
)


def test_pr_metadata_creation():
    pr = PRMetadata(
        number=42,
        title="Add auth module",
        body="Implements user authentication",
        author="aayushm0",
        state="open",
        head_sha="abc123",
        base_branch="main",
        head_branch="feat/auth",
    )
    assert pr.number == 42
    assert pr.state == "open"


def test_pr_file_creation():
    f = PRFile(
        filename="src/auth.py",
        status="added",
        patch="+import auth\n+def login(): pass",
        additions=10,
        deletions=0,
    )
    assert f.filename == "src/auth.py"
    assert f.additions == 10


def test_review_comment_creation():
    rc = ReviewComment(
        id=12345,
        html_url="https://github.com/owner/repo/pull/42#issuecomment-12345",
        body="## Review\nAll tests pass",
        event="APPROVE",
    )
    assert rc.event == "APPROVE"


def test_check_run_creation():
    cr = CheckRun(
        name="CI Build",
        status="completed",
        conclusion="success",
        output_summary="All checks passed",
    )
    assert cr.conclusion == "success"


def test_security_vulnerability():
    vuln = SecurityVulnerability(
        package="requests",
        installed_version="2.28.0",
        fixed_version="2.31.0",
        severity="high",
        advisory="CVE-2023-32681",
        url="https://github.com/advisories/GHSA-xxxx",
    )
    assert vuln.severity == "high"
    assert vuln.fixed_version == "2.31.0"


def test_security_audit_result():
    result = SecurityAuditResult(
        tool="pip-audit",
        vulnerabilities=[
            SecurityVulnerability(
                package="flask",
                installed_version="2.3.0",
                fixed_version=None,
                severity="medium",
                advisory="CVE-2023-XXXX",
                url="https://example.com",
            )
        ],
        summary="Found 1 medium vulnerability",
        exit_code=1,
    )
    assert result.tool == "pip-audit"
    assert len(result.vulnerabilities) == 1
