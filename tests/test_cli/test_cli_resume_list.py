"""Tests for Sentinel CLI Resume and List Subcommands (Phase 5 Issue 3)."""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sentinel.cli import main
from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.adr import ADR
from sentinel.models.review_state import (
    SessionStatus,
    SubagentStatus,
    SubagentType,
)
from sentinel.session_store import SessionStore


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Fixture to create an initialized git repository with initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    (repo / "README.md").write_text("# Initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"], cwd=repo, check=True, capture_output=True
    )
    return repo


def test_cli_list_sessions(temp_git_repo: Path, capsys) -> None:
    """Running 'sentinel list' prints formatted table of review sessions from SQLite."""
    db_path = str(temp_git_repo / ".sentinel" / "test.db")
    store = SessionStore(db_path=db_path)

    store.create_session(
        session_id="sess_hist_01",
        branch_name="feat/auth",
        commit_sha="1234567890ab",
        diff_summary="add auth tokens",
    )
    store.mark_completed("sess_hist_01")

    store.create_session(
        session_id="sess_hist_02",
        branch_name="feat/db",
        commit_sha="abcdef123456",
        diff_summary="migrate to duckdb",
    )

    exit_code = main(["list", "--db-path", db_path])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "sess_hist_01" in captured.out
    assert "sess_hist_02" in captured.out
    assert "feat/auth" in captured.out
    assert "feat/db" in captured.out


@pytest.mark.asyncio
async def test_cli_resume_pending_session_approved(temp_git_repo: Path, capsys) -> None:
    """Running 'sentinel resume' re-prompts pending approval session and commits ADRs without subagent re-runs."""
    db_path = str(temp_git_repo / ".sentinel" / "test.db")
    store = SessionStore(db_path=db_path)

    # Seed pending session
    session = store.create_session(
        session_id="sess_resume_01",
        branch_name="feat/redis",
        commit_sha="c0ffee123456",
        diff_summary="add redis cache",
    )
    store.transition_session(session.session_id, SessionStatus.PENDING_HUMAN_APPROVAL)

    proposed_adr = ADR(id="ADR-099", title="Use Redis", status="draft", body="")
    # Seed completed subagent tasks
    store.save_subagent_result(
        session_id="sess_resume_01",
        subagent_type=SubagentType.SANDBOX_RUNNER,
        status=SubagentStatus.COMPLETED,
        result_payload={
            "sandbox_status": "completed",
            "exit_code": 0,
            "tests_passed": 5,
            "tests_failed": 0,
        },
        task_id="sess_resume_01_sandbox",
    )
    store.save_subagent_result(
        session_id="sess_resume_01",
        subagent_type=SubagentType.ADR_DELTA_ANALYZER,
        status=SubagentStatus.COMPLETED,
        result_payload={
            "session_id": "sess_resume_01",
            "violations": [],
            "proposed_adrs": [proposed_adr.model_dump()],
        },
        task_id="sess_resume_01_adr",
    )

    mock_lace = AsyncMock(spec=LaceMcpClient)
    mock_lace.commit_adr = AsyncMock(return_value=True)

    with (
        patch("sentinel.cli._get_lace_client", return_value=mock_lace),
        patch("builtins.input", return_value="a"),
        patch("sentinel.orchestrator.SandboxRunner.run", new_callable=AsyncMock) as mock_sb,
        patch("sentinel.orchestrator.ADRDeltaAnalyzer.run", new_callable=AsyncMock) as mock_adr,
    ):
        exit_code = main(
            [
                "resume",
                "sess_resume_01",
                "--workspace",
                str(temp_git_repo),
                "--db-path",
                db_path,
            ]
        )

    assert exit_code == 0
    mock_sb.assert_not_called()
    mock_adr.assert_not_called()
    mock_lace.commit_adr.assert_called_once()

    updated = store.get_session("sess_resume_01")
    assert updated is not None
    assert updated.status == SessionStatus.COMPLETED
