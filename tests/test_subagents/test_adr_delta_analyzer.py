"""Tests for Subagent B (ADR-Delta Reasoning Analyzer) - Phase 4."""

from unittest.mock import AsyncMock

import pytest

from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.adr import ADR
from sentinel.models.diff import parse_git_diff
from sentinel.models.review_state import (
    SubagentStatus,
    SubagentType,
)
from sentinel.session_store import SessionStore
from sentinel.subagents.adr_delta_analyzer import (
    ADRDeltaAnalyzer,
    DeltaReport,
    DeltaRequest,
)


@pytest.fixture
def mock_lace_client() -> LaceMcpClient:
    """Fixture providing a mock LaceMcpClient."""
    client = AsyncMock(spec=LaceMcpClient)
    client.get_relevant_adrs = AsyncMock(return_value=[])
    return client


def test_delta_request_validation(mock_lace_client: LaceMcpClient) -> None:
    """DeltaRequest must validate session_id and timeout_seconds."""
    empty_diff = parse_git_diff("")

    # Valid request
    req = DeltaRequest(
        session_id="sess_001",
        git_diff=empty_diff,
        touched_files=[],
        lace_client=mock_lace_client,
    )
    assert req.session_id == "sess_001"
    assert req.timeout_seconds == 30

    # Empty session_id must fail
    with pytest.raises(ValueError, match="session_id must not be empty"):
        DeltaRequest(
            session_id="",
            git_diff=empty_diff,
            touched_files=[],
            lace_client=mock_lace_client,
        )

    # Negative/zero timeout must fail
    with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
        DeltaRequest(
            session_id="sess_001",
            git_diff=empty_diff,
            touched_files=[],
            lace_client=mock_lace_client,
            timeout_seconds=0,
        )


def test_delta_report_to_dict() -> None:
    """DeltaReport.to_dict() must format payload matching ApprovalGate schema."""
    report = DeltaReport(
        session_id="sess_001",
        violations=["auth/session.py violates ADR-014"],
        modified_adrs=["ADR-014"],
        proposed_adrs=[],
        summary="1 architectural violation detected.",
        duration_ms=45,
    )
    data = report.to_dict()
    assert data["session_id"] == "sess_001"
    assert data["violations"] == ["auth/session.py violates ADR-014"]
    assert data["modified_adrs"] == ["ADR-014"]
    assert data["proposed_adrs"] == []
    assert data["summary"] == "1 architectural violation detected."
    assert data["duration_ms"] == 45


@pytest.mark.asyncio
async def test_adr_delta_analyzer_empty_diff(mock_lace_client: LaceMcpClient) -> None:
    """Empty or whitespace diffs must return clean DeltaReport with 0 violations."""
    empty_diff = parse_git_diff("")
    req = DeltaRequest(
        session_id="sess_empty",
        git_diff=empty_diff,
        touched_files=[],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert report.session_id == "sess_empty"
    assert report.violations == []
    assert report.modified_adrs == []
    assert report.proposed_adrs == []
    assert "No architectural changes" in report.summary
    assert report.duration_ms >= 0


@pytest.mark.asyncio
async def test_adr_delta_analyzer_persists_to_session_store(
    mock_lace_client: LaceMcpClient, tmp_path
) -> None:
    """ADRDeltaAnalyzer must persist SubagentTask with SubagentStatus.COMPLETED to SQLite."""
    db_path = tmp_path / "test_sentinel.db"
    store = SessionStore(db_path=db_path)
    store.create_session(
        session_id="sess_store_01",
        branch_name="main",
        commit_sha="1234567abcdef",
        diff_summary="clean diff",
    )

    empty_diff = parse_git_diff("")
    req = DeltaRequest(
        session_id="sess_store_01",
        git_diff=empty_diff,
        touched_files=[],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req, session_store=store)

    assert report.session_id == "sess_store_01"

    session = store.get_session("sess_store_01")
    assert session is not None
    assert len(session.tasks) == 1
    task = session.tasks[0]
    assert task.subagent_type == SubagentType.ADR_DELTA_ANALYZER
    assert task.status == SubagentStatus.COMPLETED
    assert task.result_payload is not None
    assert task.result_payload["violations"] == []


@pytest.mark.asyncio
async def test_adr_delta_analyzer_detects_prohibited_pattern(
    mock_lace_client: LaceMcpClient,
) -> None:
    """ADRDeltaAnalyzer detects prohibited pattern in + lines and emits exact citation."""
    raw_diff = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -10,3 +10,4 @@\n"
        " def save_token(token):\n"
        "+    localStorage.setItem('auth_token', token)\n"
    )
    diff = parse_git_diff(raw_diff)

    adr_14 = ADR(
        id="ADR-014",
        title="Encrypted State Persistence Policy",
        status="accepted",
        category="decision",
        tags=["storage", "security"],
        constraints=["NEVER use raw window.localStorage for auth tokens"],
        code_pattern="localStorage.setItem",
        body="Direct access to localStorage exposes tokens.",
    )
    mock_lace_client.get_relevant_adrs = AsyncMock(return_value=[adr_14])

    req = DeltaRequest(
        session_id="sess_viol_01",
        git_diff=diff,
        touched_files=["src/auth.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.violations) == 1
    assert "src/auth.py" in report.violations[0]
    assert "ADR-014" in report.violations[0]
    assert "Encrypted State Persistence Policy" in report.violations[0]
    assert "1 architectural violation(s) detected" in report.summary


@pytest.mark.asyncio
async def test_adr_delta_analyzer_confidence_threshold_filtering(
    mock_lace_client: LaceMcpClient,
) -> None:
    """ADRs with confidence below confidence_threshold must not be enforced."""
    raw_diff = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -10,3 +10,4 @@\n"
        "+    lowConfPattern.execute()\n"
    )
    diff = parse_git_diff(raw_diff)

    low_conf_adr = ADR(
        id="ADR-099",
        title="Low Confidence Rule",
        status="accepted",
        confidence=0.6,
        code_pattern="lowConfPattern.execute",
        body="",
    )
    mock_lace_client.get_relevant_adrs = AsyncMock(return_value=[low_conf_adr])

    req = DeltaRequest(
        session_id="sess_conf_test",
        git_diff=diff,
        touched_files=["src/auth.py"],
        lace_client=mock_lace_client,
        confidence_threshold=0.8,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.violations) == 0


@pytest.mark.asyncio
async def test_adr_delta_analyzer_unhandled_error_records_failed_status(
    mock_lace_client: LaceMcpClient, tmp_path
) -> None:
    """When LACE MCP fails, analyzer records SubagentStatus.FAILED in SQLite and returns error report."""
    db_path = tmp_path / "test_sentinel.db"
    store = SessionStore(db_path=db_path)
    store.create_session(
        session_id="sess_err_01",
        branch_name="main",
        commit_sha="1234567abcdef",
        diff_summary="error diff",
    )

    raw_diff = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -1,1 +1,2 @@\n"
        "+x = 1\n"
    )
    diff = parse_git_diff(raw_diff)

    # Force LACE client to raise an exception
    mock_lace_client.get_relevant_adrs = AsyncMock(
        side_effect=RuntimeError("LACE MCP connection died")
    )

    req = DeltaRequest(
        session_id="sess_err_01",
        git_diff=diff,
        touched_files=["src/auth.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req, session_store=store)

    assert "Analysis encountered an error" in report.summary
    assert report.violations == []

    session = store.get_session("sess_err_01")
    assert session is not None
    assert len(session.tasks) == 1
    task = session.tasks[0]
    assert task.status == SubagentStatus.FAILED
    assert task.result_payload is not None
    assert "error" in task.result_payload


def test_comment_stripping_preserves_literals() -> None:
    """_strip_comments strips inline comments (# or //) while preserving string literals and URLs."""
    analyzer = ADRDeltaAnalyzer()

    # Hex colors in string literals
    assert analyzer._strip_comments("const color = '#ff0000';") == "const color = '#ff0000';"
    assert analyzer._strip_comments('bg_color = "#123456"') == 'bg_color = "#123456"'

    # URLs
    assert (
        analyzer._strip_comments("const endpoint = 'https://api.example.com/v1';")
        == "const endpoint = 'https://api.example.com/v1';"
    )
    assert (
        analyzer._strip_comments("url = http://localhost:8000/docs")
        == "url = http://localhost:8000/docs"
    )

    # Inline comments without preceding whitespace
    assert analyzer._strip_comments("x=1#comment") == "x=1"
    assert analyzer._strip_comments("foo()//comment") == "foo()"
    assert analyzer._strip_comments("let a = 2; // trailing comment") == "let a = 2;"


@pytest.mark.asyncio
async def test_detect_architectural_pivot_and_draft_madr_adr(
    mock_lace_client: LaceMcpClient,
) -> None:
    """Introducing a novel 3rd-party library (e.g. duckdb) auto-drafts a MADR 3.0 proposed ADR."""
    raw_diff = (
        "diff --git a/src/analytics/pipeline.py b/src/analytics/pipeline.py\n"
        "--- a/src/analytics/pipeline.py\n"
        "+++ b/src/analytics/pipeline.py\n"
        "@@ -1,3 +1,4 @@\n"
        "+import duckdb\n"
        "+from duckdb import connect\n"
        " def run_query(q):\n"
    )
    diff = parse_git_diff(raw_diff)

    req = DeltaRequest(
        session_id="sess_novel_01",
        git_diff=diff,
        touched_files=["src/analytics/pipeline.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.proposed_adrs) == 1
    draft = report.proposed_adrs[0]
    assert draft.id == "ADR-001"
    assert "duckdb" in draft.title.lower()
    assert draft.status == "draft"
    assert "duckdb" in draft.code_pattern
    assert "duckdb" in draft.tags

    # Verify MADR 3.0 markdown roundtrip
    md = draft.to_markdown()
    assert "---" in md
    assert "id: ADR-001" in md
    rehydrated = ADR.from_markdown(md)
    assert rehydrated.id == draft.id
    assert rehydrated.title == draft.title


@pytest.mark.asyncio
async def test_known_project_dependencies_do_not_trigger_redundant_proposed_adrs(
    mock_lace_client: LaceMcpClient,
) -> None:
    """Declared packages like pydantic, mcp, yaml, pytest, and stdlib modules are not flagged as novel."""
    raw_diff = (
        "diff --git a/src/core.py b/src/core.py\n"
        "--- a/src/core.py\n"
        "+++ b/src/core.py\n"
        "@@ -1,3 +1,7 @@\n"
        "+import pydantic\n"
        "+import yaml\n"
        "+import subprocess\n"
        "+import socket\n"
        "+from mcp import ClientSession\n"
    )
    diff = parse_git_diff(raw_diff)

    req = DeltaRequest(
        session_id="sess_declared_dep",
        git_diff=diff,
        touched_files=["src/core.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.proposed_adrs) == 0
