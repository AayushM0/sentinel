"""Tests for Git Context & Diff Extractor (Phase 5 Issue 1)."""

import subprocess
from pathlib import Path

import pytest

from sentinel.git_utils import (
    GitContext,
    GitError,
    extract_git_context,
    get_current_branch,
    get_head_commit_sha,
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

    # Initial commit on main
    (repo / "README.md").write_text("# Initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"], cwd=repo, check=True, capture_output=True
    )
    return repo


def test_extract_git_context_clean_repo(temp_git_repo: Path) -> None:
    """Clean git repo with no modified files returns empty diff and is_dirty=False."""
    ctx = extract_git_context(temp_git_repo)
    assert isinstance(ctx, GitContext)
    assert len(ctx.commit_sha) >= 7
    assert ctx.raw_diff == ""
    assert ctx.touched_files == []
    assert ctx.is_dirty is False


def test_extract_git_context_unstaged_changes(temp_git_repo: Path) -> None:
    """Unstaged modifications in working tree are extracted into raw_diff and touched_files."""
    file_path = temp_git_repo / "README.md"
    file_path.write_text("# Initial\n\nAdded line in working tree.\n", encoding="utf-8")

    ctx = extract_git_context(temp_git_repo, mode="unstaged")
    assert ctx.is_dirty is True
    assert "README.md" in ctx.touched_files
    assert "+Added line in working tree." in ctx.raw_diff


def test_extract_git_context_staged_changes(temp_git_repo: Path) -> None:
    """Staged changes are extracted when mode='staged'."""
    new_file = temp_git_repo / "new_module.py"
    new_file.write_text("import duckdb\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "new_module.py"], cwd=temp_git_repo, check=True, capture_output=True
    )

    ctx = extract_git_context(temp_git_repo, mode="staged")
    assert "new_module.py" in ctx.touched_files
    assert "+import duckdb" in ctx.raw_diff


def test_extract_git_context_branch_diff(temp_git_repo: Path) -> None:
    """Branch diff against base branch (main) extracts committed changes on feature branch."""
    # Ensure current branch is main
    subprocess.run(
        ["git", "branch", "-M", "main"], cwd=temp_git_repo, check=True, capture_output=True
    )

    # Create feature branch
    subprocess.run(
        ["git", "checkout", "-b", "feat/novel"], cwd=temp_git_repo, check=True, capture_output=True
    )
    (temp_git_repo / "feature.py").write_text("def feature(): pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "feature.py"], cwd=temp_git_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add feature"], cwd=temp_git_repo, check=True, capture_output=True
    )

    ctx = extract_git_context(temp_git_repo, mode="branch", base_branch="main")
    assert ctx.branch_name == "feat/novel"
    assert "feature.py" in ctx.touched_files
    assert "+def feature(): pass" in ctx.raw_diff


def test_extract_git_context_not_a_git_repo(tmp_path: Path) -> None:
    """Invoking extract_git_context on a non-git directory raises GitError."""
    non_repo = tmp_path / "plain_dir"
    non_repo.mkdir()

    with pytest.raises(GitError, match="Not a git repository"):
        extract_git_context(non_repo)


def test_get_current_branch_and_commit(temp_git_repo: Path) -> None:
    """Helper functions return valid branch name and commit SHA."""
    branch = get_current_branch(temp_git_repo)
    assert branch in ("main", "master")
    sha = get_head_commit_sha(temp_git_repo)
    assert len(sha) == 40
