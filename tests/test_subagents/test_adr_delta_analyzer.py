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


@pytest.mark.asyncio
async def test_substring_collision_does_not_trigger_false_violation(
    mock_lace_client: LaceMcpClient,
) -> None:
    """Word boundary matching must prevent substring collision (localStore vs localStorage)."""
    diff = parse_git_diff(
        "diff --git a/src/store.ts b/src/store.ts\n"
        "--- a/src/store.ts\n"
        "+++ b/src/store.ts\n"
        "@@ -1,2 +1,3 @@\n"
        "+const localStore = new CustomStore();\n"
    )
    adr = ADR(
        id="ADR-014",
        title="Encrypted Store",
        status="accepted",
        code_pattern="localStorage",
        body="Use SecureStore",
    )
    mock_lace_client.get_relevant_adrs = AsyncMock(return_value=[adr])

    req = DeltaRequest(
        session_id="sess_word_boundary",
        git_diff=diff,
        touched_files=["src/store.ts"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.violations) == 0


@pytest.mark.asyncio
async def test_batch_multi_file_violations(
    mock_lace_client: LaceMcpClient,
) -> None:
    """Analyzer must report multiple violations across multiple files without halting early."""
    diff = parse_git_diff(
        "diff --git a/src/auth.ts b/src/auth.ts\n"
        "--- a/src/auth.ts\n"
        "+++ b/src/auth.ts\n"
        "@@ -1,2 +1,3 @@\n"
        "+window.localStorage.setItem('auth', token);\n"
        "diff --git a/src/db.py b/src/db.py\n"
        "--- a/src/db.py\n"
        "+++ b/src/db.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+sqlite3.connect('raw.db')\n"
    )
    adr1 = ADR(
        id="ADR-014",
        title="Auth Encryption",
        status="accepted",
        code_pattern="localStorage.setItem",
        body="Use SecureStore",
    )
    adr2 = ADR(
        id="ADR-022",
        title="Async DB Pool",
        status="accepted",
        code_pattern="sqlite3.connect",
        body="Use AsyncSession",
    )
    mock_lace_client.get_relevant_adrs = AsyncMock(return_value=[adr1, adr2])

    req = DeltaRequest(
        session_id="sess_multi_viol",
        git_diff=diff,
        touched_files=["src/auth.ts", "src/db.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.violations) == 2
    paths_in_violations = [v for v in report.violations if "src/auth.ts" in v or "src/db.py" in v]
    assert len(paths_in_violations) == 2


@pytest.mark.asyncio
async def test_query_context_extracts_imports_and_symbols(
    mock_lace_client: LaceMcpClient,
) -> None:
    """_extract_query_context must extract imports and symbols from added lines."""
    diff = parse_git_diff(
        "diff --git a/src/cache/redis_mgr.py b/src/cache/redis_mgr.py\n"
        "--- a/src/cache/redis_mgr.py\n"
        "+++ b/src/cache/redis_mgr.py\n"
        "@@ -1,2 +1,4 @@\n"
        "+import redis\n"
        "+from cryptography.fernet import Fernet\n"
    )
    req = DeltaRequest(
        session_id="sess_imports",
        git_diff=diff,
        touched_files=["src/cache/redis_mgr.py"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    query = analyzer._extract_query_context(req)

    assert "redis" in query
    assert "cache" in query
    assert "cryptography" in query or "fernet" in query.lower()


@pytest.mark.asyncio
async def test_multiline_block_comments_do_not_trigger_violations(
    mock_lace_client: LaceMcpClient,
) -> None:
    """Block comments /* ... */ must not trigger constraint violations."""
    diff = parse_git_diff(
        "diff --git a/src/auth.ts b/src/auth.ts\n"
        "--- a/src/auth.ts\n"
        "+++ b/src/auth.ts\n"
        "@@ -1,2 +1,5 @@\n"
        "+/* \n"
        "+ * Example migration:\n"
        "+ * localStorage.setItem('token', val)\n"
        "+ */\n"
        "+const safeStore = new SecureStore();\n"
    )
    adr = ADR(
        id="ADR-014",
        title="Auth Encryption",
        status="accepted",
        code_pattern="localStorage.setItem",
        body="Use SecureStore",
    )
    mock_lace_client.get_relevant_adrs = AsyncMock(return_value=[adr])

    req = DeltaRequest(
        session_id="sess_block_comment",
        git_diff=diff,
        touched_files=["src/auth.ts"],
        lace_client=mock_lace_client,
    )

    analyzer = ADRDeltaAnalyzer()
    report = await analyzer.run(req)

    assert len(report.violations) == 0
