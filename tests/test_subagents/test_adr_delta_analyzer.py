"""Tests for Subagent B (ADR-Delta Reasoning Analyzer) - Phase 4."""

from unittest.mock import AsyncMock, patch

import pytest

from sentinel.approval_gate import ApprovalGate
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
    assert "No architectural changes or violations detected." in report.summary
    assert report.duration_ms >= 0


@pytest.mark.asyncio
async def test_adr_delta_analyzer_persists_to_session_store(
    mock_lace_client: LaceMcpClient, tmp_path
) -> None:
    """ADRDeltaAnalyzer must automatically persist results to SessionStore."""
    db_path = tmp_path / "test_sentinel.db"
    store = SessionStore(db_path=db_path)
    store.create_session(
        session_id="sess_persist",
        branch_name="feat/phase-4",
        commit_sha="abcdef123456",
        diff_summary="empty diff",
    )

    empty_diff = parse_git_diff("")
    req = DeltaRequest(
        session_id="sess_persist",
        git_diff=empty_diff,
        touched_files=[],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req, session_store=store)
    assert report.session_id == "sess_persist"

    session = store.get_session("sess_persist")
    assert session is not None
    task = next(t for t in session.tasks if t.subagent_type == SubagentType.ADR_DELTA_ANALYZER)
    assert task.status == SubagentStatus.COMPLETED
    assert task.result_payload is not None
    assert task.result_payload["violations"] == []


@pytest.mark.asyncio
async def test_adr_delta_analyzer_detects_prohibited_pattern(
    mock_lace_client: LaceMcpClient,
) -> None:
    """ADRDeltaAnalyzer must detect prohibited code pattern in added lines."""
    diff = parse_git_diff(
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -1,3 +1,4 @@\n"
        "+localStorage.setItem('key', val)\n"
    )
    adr = ADR(
        id="ADR-014",
        title="Encrypted Store",
        status="accepted",
        code_pattern="localStorage.setItem",
        body="Use SecureStore",
    )
    mock_lace_client.get_relevant_adrs = AsyncMock(return_value=[adr])

    req = DeltaRequest(
        session_id="sess_viol",
        git_diff=diff,
        touched_files=["src/auth.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.violations) == 1
    assert "ADR-014" in report.violations[0]
    assert "src/auth.py" in report.violations[0]


@pytest.mark.asyncio
async def test_adr_delta_analyzer_ignores_deleted_lines(
    mock_lace_client: LaceMcpClient,
) -> None:
    """Deleting a prohibited line must NOT trigger a violation alert."""
    diff = parse_git_diff(
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -1,4 +1,3 @@\n"
        "-localStorage.setItem('key', val)\n"
        "+SecureStore.setItem('key', val)\n"
    )
    adr = ADR(
        id="ADR-014",
        title="Encrypted Store",
        status="accepted",
        code_pattern="localStorage.setItem",
        body="Use SecureStore",
    )
    mock_lace_client.get_relevant_adrs = AsyncMock(return_value=[adr])

    req = DeltaRequest(
        session_id="sess_del",
        git_diff=diff,
        touched_files=["src/auth.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.violations) == 0
    assert "No architectural changes" in report.summary


@pytest.mark.asyncio
async def test_adr_delta_analyzer_ignores_deprecated_adr(
    mock_lace_client: LaceMcpClient,
) -> None:
    """Deprecated ADR constraints must be ignored."""
    diff = parse_git_diff(
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -1,3 +1,4 @@\n"
        "+localStorage.setItem('key', val)\n"
    )
    adr = ADR(
        id="ADR-010",
        title="Old Policy",
        status="deprecated",
        code_pattern="localStorage.setItem",
        body="Old body",
    )
    mock_lace_client.get_relevant_adrs = AsyncMock(return_value=[adr])

    req = DeltaRequest(
        session_id="sess_dep",
        git_diff=diff,
        touched_files=["src/auth.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.violations) == 0


@pytest.mark.asyncio
async def test_adr_delta_analyzer_unhandled_error_records_failed_status(
    mock_lace_client: LaceMcpClient, tmp_path
) -> None:
    """ADRDeltaAnalyzer must record SubagentStatus.FAILED on unhandled exception."""
    db_path = tmp_path / "test_sentinel.db"
    store = SessionStore(db_path=db_path)
    store.create_session(
        session_id="sess_err",
        branch_name="feat/phase-4",
        commit_sha="abcdef123456",
        diff_summary="error diff",
    )

    diff = parse_git_diff(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,2 @@\n+x = 1\n"
    )
    req = DeltaRequest(
        session_id="sess_err",
        git_diff=diff,
        touched_files=["a.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    with patch.object(
        analyzer, "_extract_query_context", side_effect=RuntimeError("Simulated engine fault")
    ):
        report = await analyzer.run(req, session_store=store)

    assert "Simulated engine fault" in report.summary
    session = store.get_session("sess_err")
    assert session is not None
    task = next(t for t in session.tasks if t.subagent_type == SubagentType.ADR_DELTA_ANALYZER)
    assert task.status == SubagentStatus.FAILED
    assert "Simulated engine fault" in task.result_payload["error"]


def test_comment_stripping_preserves_literals() -> None:
    """Comment stripping must not truncate hex colors or URLs."""
    analyzer = ADRDeltaAnalyzer()
    assert analyzer._strip_comments("const color = '#ff0000';") == "const color = '#ff0000';"
    assert (
        analyzer._strip_comments("const url = 'https://api.example.com';")
        == "const url = 'https://api.example.com';"
    )
    assert analyzer._strip_comments("// comment line") == ""
    assert analyzer._strip_comments("# python comment") == ""
    assert analyzer._strip_comments("const x = 1; // trailing") == "const x = 1;"
    assert analyzer._strip_comments("x = 1 # trailing") == "x = 1"


@pytest.mark.asyncio
async def test_adr_delta_analyzer_approval_gate_integration(
    mock_lace_client: LaceMcpClient,
) -> None:
    """DeltaReport output must seamlessly format through ApprovalGate.format_approval_card."""
    empty_diff = parse_git_diff("")
    req = DeltaRequest(
        session_id="sess_gate",
        git_diff=empty_diff,
        touched_files=[],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    gate = ApprovalGate()
    card = gate.format_approval_card(
        session_id="sess_gate",
        branch_name="feat/phase-4",
        commit_sha="abcdef123456",
        test_result={
            "sandbox_status": "completed",
            "exit_code": 0,
            "tests_passed": 3,
            "tests_failed": 0,
        },
        delta_report=report.to_dict(),
    )

    assert "Zero violations detected." in card
    assert "sess_gate" in card
