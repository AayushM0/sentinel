"""Unit and integration tests for TrueForgeAdapter and harness contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sentinel.approval_gate import ApprovalDecision
from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.adr import ADR
from sentinel.models.review_state import (
    ApprovalActionType,
    ReviewSession,
    SessionStatus,
    SubagentStatus,
    SubagentTask,
    SubagentType,
)
from sentinel.session_store import SessionStore
from sentinel.subagents.sandbox_runner import SandboxResult
from sentinel.trueforge_adapter import TrueForgeAdapter


def test_trueforge_config_loading() -> None:
    """TrueForgeAdapter loads and parses config from workspace."""
    adapter = TrueForgeAdapter()
    assert adapter.config.version == "1.0.0"
    assert adapter.config.agent.name == "sentinel"
    assert adapter.config.storage.backend == "sqlite"
    assert adapter.config.subagents.sandbox_runner.provider == "daytona"


def test_is_tool_gated_and_approval_policies() -> None:
    """Verifies tool gating and approval policy checks."""
    adapter = TrueForgeAdapter()

    # Gated tools
    assert adapter.is_tool_gated("remember") is True
    assert adapter.is_tool_gated("git_push") is True
    assert adapter.is_tool_gated("sentinel_check_diff") is False

    # Approval policies
    assert adapter.requires_approval(ApprovalActionType.PRE_PUSH_COMMIT) is True
    assert adapter.requires_approval("LACE_ADR_UPDATE") is True
    assert adapter.requires_approval("UNKNOWN_ACTION") is True


def test_get_tool_definitions() -> None:
    """Verifies TrueForge tool definitions schema format."""
    adapter = TrueForgeAdapter()
    defs = adapter.get_tool_definitions()

    assert len(defs) == 4
    names = [d["name"] for d in defs]
    assert "sentinel_check_diff" in names
    assert "sentinel_query_adrs" in names
    assert "sentinel_run_sandbox" in names
    assert "sentinel_resolve_approval" in names

    for d in defs:
        assert "name" in d
        assert "description" in d
        assert "parameters" in d
        assert d["parameters"]["type"] == "object"


def test_resolve_approval_through_adapter(tmp_path: Path) -> None:
    """Resolves pending approval gate through TrueForgeAdapter and completes workflow."""
    db_path = str(tmp_path / "test_session.db")
    store = SessionStore(db_path=db_path)
    sess = store.create_session("feat/test", "commit123", "Test diff")

    appr = store.set_pending_approval(
        sess.session_id,
        ApprovalActionType.PRE_PUSH_COMMIT,
        {"summary": "Test approval"},
    )

    adapter = TrueForgeAdapter()
    res = adapter.resolve_approval(appr.approval_id, ApprovalDecision.APPROVED, db_path=db_path)

    assert res["status"] == "resolved"
    assert res["decision"] == "APPROVED"

    hydrated = store.get_session(sess.session_id)
    assert hydrated is not None
    assert hydrated.status == SessionStatus.COMPLETED

    # Resolving already decided approval raises ValueError
    with pytest.raises(ValueError, match="already been decided"):
        adapter.resolve_approval(appr.approval_id, ApprovalDecision.REJECTED, db_path=db_path)

    # Missing approval ID raises ValueError
    with pytest.raises(ValueError, match="not found"):
        adapter.resolve_approval("missing_id", ApprovalDecision.APPROVED, db_path=db_path)


@pytest.mark.asyncio
async def test_check_diff_through_adapter(tmp_path: Path) -> None:
    """check_diff returns standardized dict result from ReviewOrchestrator."""
    db_path = str(tmp_path / "test_session.db")
    store = SessionStore(db_path=db_path)

    mock_session = ReviewSession(
        session_id="sess_adapter_test",
        branch_name="main",
        commit_sha="0000000000000000000000000000000000000000",
        status=SessionStatus.COMPLETED,
        diff_summary="Clean repo test",
        tasks=[
            SubagentTask(
                task_id="t1",
                session_id="sess_adapter_test",
                subagent_type=SubagentType.SANDBOX_RUNNER,
                status=SubagentStatus.COMPLETED,
                result_payload={"exit_code": 0},
            )
        ],
    )

    with (
        patch("sentinel.trueforge_adapter.extract_git_context") as mock_git,
        patch(
            "sentinel.trueforge_adapter.ReviewOrchestrator.run_review", new_callable=AsyncMock
        ) as mock_review,
    ):
        mock_git.return_value.branch_name = "main"
        mock_git.return_value.commit_sha = "abc123"
        mock_git.return_value.raw_diff = ""
        mock_git.return_value.touched_files = []
        mock_review.return_value = mock_session

        adapter = TrueForgeAdapter()
        res = await adapter.check_diff(workspace_path=str(tmp_path), store=store)

        assert res["session_id"] == "sess_adapter_test"
        assert res["status"] == "COMPLETED"
        assert len(res["tasks"]) == 1
        assert res["tasks"][0]["subagent_type"] == "SANDBOX_RUNNER"


@pytest.mark.asyncio
async def test_check_diff_invalid_mode_raises(tmp_path: Path) -> None:
    """check_diff rejects invalid diff mode."""
    adapter = TrueForgeAdapter()
    with pytest.raises(ValueError, match="Invalid diff extraction mode"):
        await adapter.check_diff(workspace_path=str(tmp_path), mode="invalid")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_query_adrs_through_adapter() -> None:
    """query_adrs returns formatted ADR dictionary entries."""
    mock_adr = ADR(
        id="ADR-001",
        title="Use Encrypted Storage",
        status="accepted",
        context="Security requirement",
        consequences="Better security",
        code_pattern="use_encrypted_store()",
        constraints=["Do not use raw localStorage"],
    )

    mock_lace = AsyncMock(spec=LaceMcpClient)
    mock_lace.is_connected = True
    mock_lace.get_relevant_adrs = AsyncMock(return_value=[mock_adr])

    adapter = TrueForgeAdapter()
    results = await adapter.query_adrs(
        touched_files=["auth.ts"],
        query="storage",
        lace_client=mock_lace,
    )

    assert len(results) == 1
    assert results[0]["title"] == "Use Encrypted Storage"
    assert results[0]["status"] == "accepted"
    assert results[0]["constraints"] == ["Do not use raw localStorage"]


@pytest.mark.asyncio
async def test_run_sandbox_through_adapter(tmp_path: Path) -> None:
    """run_sandbox executes tests and returns serialized result."""
    mock_result = SandboxResult(
        sandbox_status="completed",
        exit_code=0,
        tests_passed=5,
        tests_failed=0,
        linter_errors=[],
        duration_ms=120,
        logs="All tests passed",
    )

    with patch("sentinel.trueforge_adapter.SandboxRunner.run", new_callable=AsyncMock) as mock_sb:
        mock_sb.return_value = mock_result

        adapter = TrueForgeAdapter()
        res = await adapter.run_sandbox(workspace_path=str(tmp_path))

        assert res["sandbox_status"] == "completed"
        assert res["exit_code"] == 0
        assert res["tests_passed"] == 5


@pytest.mark.asyncio
async def test_run_sandbox_zero_timeout_raises(tmp_path: Path) -> None:
    """run_sandbox does not bypass non-positive timeout validation."""
    adapter = TrueForgeAdapter()
    with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
        await adapter.run_sandbox(workspace_path=str(tmp_path), timeout_seconds=0)
