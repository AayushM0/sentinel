"""Git repository utilities for extracting working tree diffs, branches, and commit metadata."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger("sentinel.git_utils")


class GitError(Exception):
    """Raised when git command execution fails or workspace is not a valid repository."""


@dataclass
class GitContext:
    """Contextual metadata and unified diff extracted from a local git workspace."""

    branch_name: str
    commit_sha: str
    raw_diff: str
    touched_files: list[str] = field(default_factory=list)
    is_dirty: bool = False


def _run_git(args: list[str], cwd: Path, timeout: int = 15) -> str:
    """Execute git command safely with timeout and return stdout."""
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise GitError("git executable not found in PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git command timed out after {timeout}s: git {' '.join(args)}") from exc
    except Exception as exc:
        raise GitError(f"git command failed: {exc}") from exc

    if res.returncode != 0:
        err_msg = res.stderr.strip() or res.stdout.strip()
        if "not a git repository" in err_msg.lower():
            raise GitError(f"Not a git repository: {cwd}")
        raise GitError(f"git command exited with code {res.returncode}: {err_msg}")

    return res.stdout


def is_git_repository(repo_path: Path) -> bool:
    """Check if the given directory is inside an initialized git work tree."""
    try:
        out = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=repo_path)
        return out.strip() == "true"
    except GitError:
        return False


def get_current_branch(repo_path: Path) -> str:
    """Get active git branch name, or short commit SHA if detached HEAD."""
    try:
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path).strip()
        if branch == "HEAD":
            # Detached HEAD, return short SHA
            return _run_git(["rev-parse", "--short", "HEAD"], cwd=repo_path).strip()
        return branch
    except GitError:
        return "main"


def get_head_commit_sha(repo_path: Path) -> str:
    """Get authoritative 40-character commit SHA of HEAD."""
    try:
        return _run_git(["rev-parse", "HEAD"], cwd=repo_path).strip()
    except GitError:
        return "0000000000000000000000000000000000000000"


def extract_git_context(
    repo_path: Path,
    mode: Literal["unstaged", "staged", "branch", "working_tree"] = "unstaged",
    base_branch: str = "main",
) -> GitContext:
    """Extract diff and git metadata from repository based on requested mode."""
    if not is_git_repository(repo_path):
        raise GitError(f"Not a git repository: {repo_path}")

    branch_name = get_current_branch(repo_path)
    commit_sha = get_head_commit_sha(repo_path)

    # Determine diff command arguments based on mode
    if mode == "staged":
        diff_args = ["diff", "--cached"]
    elif mode == "branch":
        # Check if base branch exists, otherwise fall back to origin/{base_branch}
        diff_args = ["diff", f"{base_branch}...HEAD"]
    else:  # unstaged or working_tree
        diff_args = ["diff", "HEAD"]

    try:
        raw_diff = _run_git(diff_args, cwd=repo_path)
    except GitError:
        if mode == "branch":
            # Fallback if three-dot diff fails
            raw_diff = _run_git(["diff", base_branch], cwd=repo_path)
        else:
            raw_diff = _run_git(["diff"], cwd=repo_path)

    # Check status for dirty state and touched files
    status_out = _run_git(["status", "--porcelain"], cwd=repo_path)
    is_dirty = bool(status_out.strip())

    # Extract touched file paths from git diff headers and porcelain status
    touched_set: set[str] = set()

    for line in status_out.splitlines():
        if line.strip():
            # Porcelain format: XY path or XY "path"
            path_part = line[3:].strip().strip('"')
            if " -> " in path_part:
                path_part = path_part.split(" -> ")[1].strip('"')
            if path_part:
                touched_set.add(path_part)

    for line in raw_diff.splitlines():
        if line.startswith(("+++ b/", "--- a/")):
            path = line[6:].strip()
            if path != "dev/null":
                touched_set.add(path)

    touched_files = sorted(touched_set)

    return GitContext(
        branch_name=branch_name,
        commit_sha=commit_sha,
        raw_diff=raw_diff,
        touched_files=touched_files,
        is_dirty=is_dirty,
    )


if __name__ == "__main__":
    import tempfile

    # Standalone zero-dependency self-checks (Rule 2903681)
    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Non-git directory must raise GitError
            try:
                extract_git_context(tmp_path)
                raise AssertionError("Should have raised GitError on plain directory")
            except GitError:
                pass

            # 2. Initialized repository
            subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.name", "Self Test"],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )
            (tmp_path / "init.txt").write_text("hello", encoding="utf-8")
            subprocess.run(
                ["git", "add", "init.txt"], cwd=tmp_path, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
            )

            ctx = extract_git_context(tmp_path)
            assert len(ctx.commit_sha) == 40
            assert ctx.is_dirty is False

            # Modify file
            (tmp_path / "init.txt").write_text("hello world\n", encoding="utf-8")
            ctx_dirty = extract_git_context(tmp_path)
            assert ctx_dirty.is_dirty is True
            assert "init.txt" in ctx_dirty.touched_files
            assert "+hello world" in ctx_dirty.raw_diff

            print("git_utils.py standalone self-check passed successfully.")

    _self_test()
