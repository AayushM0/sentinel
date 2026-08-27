"""Tests for ReviewOrchestrator (Concurrent Multi-Subagent Orchestration) - Phase 4 Issue 4."""

from unittest.mock import AsyncMock, patch

import pytest

from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.adr import ADR
from sentinel.models.diff import parse_git_diff
from sentinel.models.review_state import (
    SessionStatus,
    SubagentStatus,
    SubagentType,
)
from sentinel.orchestrator import OrchestratorRequest, ReviewOrchestrator
from sentinel.session_store import SessionStore
from sentinel.subagents.adr_delta_analyzer import DeltaReport
from sentinel.subagents.sandbox_runner import SandboxResult


@pytest.fixture
def mock_lace_client() -> LaceMcpClient:
    """Fixture providing a mock LaceMcpClient."""
    client = AsyncMock(spec=LaceMcpClient)
    client.get_relevant_adrs = AsyncMock(return_value=[])
    client.commit_adr = AsyncMock(return_value=True)
    return client


@pytest.mark.asyncio
async def test_orchestrator_runs_subagents_concurrently(
    mock_lace_client: LaceMcpClient, tmp_path
) -> None:
    """Orchestrator must execute SandboxRunner and ADRDeltaAnalyzer concurrently and persist both."""
    db_path = tmp_path / "test_sentinel.db"
    store = SessionStore(db_path=db_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "a.py").write_text("print('hello')", encoding="utf-8")

    diff = parse_git_diff(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,2 @@\n+print('hello')\n"
    )

    req = OrchestratorRequest(
        session_id="sess_orch_01",
        branch_name="feat/concurrent",
        commit_sha="1234567abcdef",
        diff_summary="add print",
        git_diff=diff,
        touched_files=["a.py"],
        workspace_root=workspace_dir,
        lace_client=mock_lace_client,
        session_store=store,
        interactive=False,
    )

    mock_sandbox_result = SandboxResult(
        sandbox_status="completed",
        exit_code=0,
        tests_passed=5,
        tests_failed=0,
        linter_errors=[],
        duration_ms=120,
        logs="All tests passed",
    )
    mock_delta_report = DeltaReport(
        session_id="sess_orch_01",
        violations=[],
        modified_adrs=[],
        proposed_adrs=[],
        summary="Zero violations detected.",
        duration_ms=40,
    )

    orchestrator = ReviewOrchestrator()
    with (
        patch("sentinel.orchestrator.SandboxRunner.run", new_callable=AsyncMock) as mock_sb_run,
        patch("sentinel.orchestrator.ADRDeltaAnalyzer.run", new_callable=AsyncMock) as mock_adr_run,
    ):
        mock_sb_run.return_value = mock_sandbox_result
        mock_adr_run.return_value = mock_delta_report

        session = await orchestrator.run_review(req)

    assert session.session_id == "sess_orch_01"
    assert session.status == SessionStatus.PENDING_HUMAN_APPROVAL

    # Verify both subagent tasks recorded in session store
    db_session = store.get_session("sess_orch_01")
    assert db_session is not None
    task_types = {t.subagent_type for t in db_session.tasks}
    assert SubagentType.SANDBOX_RUNNER in task_types
    assert SubagentType.ADR_DELTA_ANALYZER in task_types


@pytest.mark.asyncio
async def test_orchestrator_subagent_failure_isolation(
    mock_lace_client: LaceMcpClient, tmp_path
) -> None:
    """If SandboxRunner crashes, ADRDeltaAnalyzer completes and session reflects both states."""
    db_path = tmp_path / "test_sentinel.db"
    store = SessionStore(db_path=db_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    diff = parse_git_diff("")
    req = OrchestratorRequest(
        session_id="sess_fault_iso",
        branch_name="feat/concurrent",
        commit_sha="1234567abcdef",
        diff_summary="empty diff",
        git_diff=diff,
        touched_files=[],
        workspace_root=workspace_dir,
        lace_client=mock_lace_client,
        session_store=store,
        interactive=False,
    )

    mock_delta_report = DeltaReport(
        session_id="sess_fault_iso",
        violations=[],
        summary="Zero violations.",
        duration_ms=10,
    )

    orchestrator = ReviewOrchestrator()
    with (
        patch(
            "sentinel.orchestrator.SandboxRunner.run",
            side_effect=RuntimeError("Daytona cluster offline"),
        ),
        patch("sentinel.orchestrator.ADRDeltaAnalyzer.run", new_callable=AsyncMock) as mock_adr_run,
    ):
        mock_adr_run.return_value = mock_delta_report

        session = await orchestrator.run_review(req)

    assert session.session_id == "sess_fault_iso"
    db_session = store.get_session("sess_fault_iso")
    assert db_session is not None

    sb_task = next(t for t in db_session.tasks if t.subagent_type == SubagentType.SANDBOX_RUNNER)
    assert sb_task.status == SubagentStatus.FAILED

    adr_task = next(
        t for t in db_session.tasks if t.subagent_type == SubagentType.ADR_DELTA_ANALYZER
    )
    assert adr_task.status == SubagentStatus.COMPLETED


@pytest.mark.asyncio
async def test_orchestrator_approval_commits_proposed_adrs_to_lace(
    mock_lace_client: LaceMcpClient, tmp_path
) -> None:
    """When human approves, proposed ADRs in delta report are committed to LACE memory vault."""
    db_path = tmp_path / "test_sentinel.db"
    store = SessionStore(db_path=db_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    diff = parse_git_diff(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,2 @@\n+import duckdb\n"
    )
    req = OrchestratorRequest(
        session_id="sess_appr_commit",
        branch_name="feat/novel-db",
        commit_sha="1234567abcdef",
        diff_summary="add duckdb",
        git_diff=diff,
        touched_files=["a.py"],
        workspace_root=workspace_dir,
        lace_client=mock_lace_client,
        session_store=store,
        interactive=True,
    )

    proposed_adr = ADR(
        id="ADR-015",
        title="Adopt DuckDB",
        status="draft",
        code_pattern="duckdb",
        body="## Context\n\nDuckDB used.",
    )
    mock_sandbox_result = SandboxResult(
        sandbox_status="completed",
        exit_code=0,
        tests_passed=10,
        tests_failed=0,
        linter_errors=[],
        duration_ms=250,
        logs="Pass",
    )
    mock_delta_report = DeltaReport(
        session_id="sess_appr_commit",
        violations=[],
        proposed_adrs=[proposed_adr],
        summary="1 new ADR proposed.",
    )

    orchestrator = ReviewOrchestrator()
    with (
        patch("sentinel.orchestrator.SandboxRunner.run", new_callable=AsyncMock) as mock_sb,
        patch("sentinel.orchestrator.ADRDeltaAnalyzer.run", new_callable=AsyncMock) as mock_adr,
        patch("builtins.input", return_value="a"),
    ):
        mock_sb.return_value = mock_sandbox_result
        mock_adr.return_value = mock_delta_report

        session = await orchestrator.run_review(req)

    assert session.status == SessionStatus.APPROVED
    mock_lace_client.commit_adr.assert_called_once_with(adr=proposed_adr)


@pytest.mark.asyncio
async def test_orchestrator_rejection_aborts_without_committing_adrs(
    mock_lace_client: LaceMcpClient, tmp_path
) -> None:
    """When human rejects, session transitions to REJECTED and no ADRs are committed to LACE."""
    db_path = tmp_path / "test_sentinel.db"
    store = SessionStore(db_path=db_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    req = OrchestratorRequest(
        session_id="sess_reject",
        branch_name="feat/novel-db",
        commit_sha="1234567abcdef",
        diff_summary="add duckdb",
        git_diff=parse_git_diff(""),
        touched_files=[],
        workspace_root=workspace_dir,
        lace_client=mock_lace_client,
        session_store=store,
        interactive=True,
    )

    proposed_adr = ADR(id="ADR-015", title="Adopt DuckDB", status="draft", body="")
    mock_sandbox_result = SandboxResult(
        sandbox_status="completed",
        exit_code=0,
        tests_passed=1,
        tests_failed=0,
        linter_errors=[],
        duration_ms=100,
        logs="Pass",
    )
    mock_delta_report = DeltaReport(session_id="sess_reject", proposed_adrs=[proposed_adr])

    orchestrator = ReviewOrchestrator()
    with (
        patch("sentinel.orchestrator.SandboxRunner.run", new_callable=AsyncMock) as mock_sb,
        patch("sentinel.orchestrator.ADRDeltaAnalyzer.run", new_callable=AsyncMock) as mock_adr,
        patch("builtins.input", return_value="r"),
    ):
        mock_sb.return_value = mock_sandbox_result
        mock_adr.return_value = mock_delta_report

        session = await orchestrator.run_review(req)

    assert session.status == SessionStatus.REJECTED
    mock_lace_client.commit_adr.assert_not_called()
