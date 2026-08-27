"""Git pre-push hook installer and manager for Sentinel guardrails."""

from __future__ import annotations

import logging
import stat
from pathlib import Path

from sentinel.git_utils import is_git_repository

logger = logging.getLogger("sentinel.hooks")

_PRE_PUSH_HOOK_TEMPLATE = """#!/bin/sh
# Sentinel Pre-Push Architectural Guardrail Hook
# Automatically installed by `sentinel install-hook`

echo "Sentinel: Running pre-push architectural check on branch..."
sentinel check --branch || {
    echo "Sentinel: Push aborted due to architectural violations, failed sandbox tests, or human rejection." >&2
    exit 1
}
"""


class HookError(Exception):
    """Raised when git hook installation or removal fails."""


def install_pre_push_hook(repo_path: Path) -> Path:
    """Install executable pre-push hook in the git repository."""
    workspace = repo_path.resolve()
    if not is_git_repository(workspace):
        raise HookError(f"Not a git repository: {workspace}")

    hooks_dir = workspace / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hook_file = hooks_dir / "pre-push"
    hook_file.write_text(_PRE_PUSH_HOOK_TEMPLATE, encoding="utf-8")

    # Set executable permissions (0o755)
    try:
        current_stat = hook_file.stat().st_mode
        hook_file.chmod(current_stat | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not set executable bit on %s: %s", hook_file, exc)

    return hook_file


def uninstall_pre_push_hook(repo_path: Path) -> bool:
    """Remove pre-push hook from git repository."""
    workspace = repo_path.resolve()
    hook_file = workspace / ".git" / "hooks" / "pre-push"

    if hook_file.exists():
        try:
            hook_file.unlink()
            return True
        except Exception as exc:
            raise HookError(f"Failed to remove hook at {hook_file}: {exc}") from exc
    return False


if __name__ == "__main__":
    import subprocess
    import tempfile

    # Standalone zero-dependency self-checks (Rule 2903681)
    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Non-git dir raises HookError
            try:
                install_pre_push_hook(tmp_path)
                raise AssertionError("Should raise HookError on non-git dir")
            except HookError:
                pass

            # 2. Initialized repo
            subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
            hook_path = install_pre_push_hook(tmp_path)
            assert hook_path.exists()
            assert "sentinel check" in hook_path.read_text(encoding="utf-8")

            assert uninstall_pre_push_hook(tmp_path) is True
            assert not hook_path.exists()

            print("hooks.py standalone self-check passed successfully.")

    _self_test()
