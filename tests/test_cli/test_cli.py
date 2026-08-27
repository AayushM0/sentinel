"""Tests for Sentinel Core CLI (Phase 5 Issue 2)."""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sentinel.cli import main
from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.subagents.adr_delta_analyzer import DeltaReport
from sentinel.subagents.sandbox_runner import SandboxResult


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


def test_cli_help(capsys) -> None:
    """Invoking CLI with --help prints usage and exits with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Sentinel: Automated Architectural Guardrail" in captured.out
    assert "check" in captured.out


@pytest.mark.asyncio
async def test_cli_check_clean_diff(temp_git_repo: Path) -> None:
    """Running 'sentinel check' on a clean repo passes with exit code 0."""
    db_path = str(temp_git_repo / ".sentinel" / "test.db")

    mock_lace = AsyncMock(spec=LaceMcpClient)
    mock_lace.get_relevant_adrs = AsyncMock(return_value=[])

    with (
        patch("sentinel.cli._get_lace_client", return_value=mock_lace),
        patch("sentinel.orchestrator.SandboxRunner.run", new_callable=AsyncMock) as mock_sb,
        patch("sentinel.orchestrator.ADRDeltaAnalyzer.run", new_callable=AsyncMock) as mock_adr,
    ):
        mock_sb.return_value = SandboxResult(
            sandbox_status="completed",
            exit_code=0,
            tests_passed=1,
            tests_failed=0,
            linter_errors=[],
            duration_ms=10,
            logs="",
        )
        mock_adr.return_value = DeltaReport(session_id="s1", summary="Clean diff")

        exit_code = main(
            [
                "check",
                "--workspace",
                str(temp_git_repo),
                "--db-path",
                db_path,
                "--non-interactive",
            ]
        )
        assert exit_code in (0, 3)


@pytest.mark.asyncio
async def test_cli_check_rejection_exits_1(temp_git_repo: Path) -> None:
    """When human rejects in interactive check, CLI returns exit code 1."""
    db_path = str(temp_git_repo / ".sentinel" / "test.db")

    # Modify file
    (temp_git_repo / "README.md").write_text("# Modified\n", encoding="utf-8")

    mock_lace = AsyncMock(spec=LaceMcpClient)
    mock_lace.get_relevant_adrs = AsyncMock(return_value=[])

    with (
        patch("sentinel.cli._get_lace_client", return_value=mock_lace),
        patch("sentinel.orchestrator.SandboxRunner.run", new_callable=AsyncMock) as mock_sb,
        patch("sentinel.orchestrator.ADRDeltaAnalyzer.run", new_callable=AsyncMock) as mock_adr,
        patch("builtins.input", return_value="r"),
    ):
        mock_sb.return_value = SandboxResult(
            sandbox_status="completed",
            exit_code=0,
            tests_passed=1,
            tests_failed=0,
            linter_errors=[],
            duration_ms=10,
            logs="",
        )
        mock_adr.return_value = DeltaReport(session_id="s_rej", summary="Diff detected")

        exit_code = main(
            [
                "check",
                "--workspace",
                str(temp_git_repo),
                "--db-path",
                db_path,
            ]
        )
        assert exit_code == 1
