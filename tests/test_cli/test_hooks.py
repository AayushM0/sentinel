"""Tests for Git Pre-Push Hook Installer (Phase 5 Issue 4)."""

import subprocess
from pathlib import Path

import pytest

from sentinel.cli import main
from sentinel.hooks import (
    HookError,
    install_pre_push_hook,
    uninstall_pre_push_hook,
)


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


def test_install_and_uninstall_pre_push_hook(temp_git_repo: Path) -> None:
    """install_pre_push_hook creates executable hook and uninstall removes it."""
    hook_file = install_pre_push_hook(temp_git_repo)
    assert hook_file.exists()
    content = hook_file.read_text(encoding="utf-8")
    assert "sentinel check" in content
    assert "#!/bin/sh" in content

    # Test uninstall
    removed = uninstall_pre_push_hook(temp_git_repo)
    assert removed is True
    assert not hook_file.exists()


def test_install_hook_non_git_dir_raises(tmp_path: Path) -> None:
    """Installing hook in non-git directory raises HookError."""
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()

    with pytest.raises(HookError, match="Not a git repository"):
        install_pre_push_hook(plain_dir)


def test_cli_install_and_uninstall_commands(temp_git_repo: Path, capsys) -> None:
    """CLI subcommands 'install-hook' and 'uninstall-hook' execute successfully."""
    exit_code_install = main(["install-hook", "--workspace", str(temp_git_repo)])
    assert exit_code_install == 0
    captured = capsys.readouterr()
    assert "pre-push hook installed successfully" in captured.out

    hook_path = temp_git_repo / ".git" / "hooks" / "pre-push"
    assert hook_path.exists()

    exit_code_uninstall = main(["uninstall-hook", "--workspace", str(temp_git_repo)])
    assert exit_code_uninstall == 0
    captured_un = capsys.readouterr()
    assert "pre-push hook removed successfully" in captured_un.out
    assert not hook_path.exists()
