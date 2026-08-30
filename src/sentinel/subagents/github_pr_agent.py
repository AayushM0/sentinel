"""Subagent C: GitHub PR Integration.

Fetches PR context from GitHub and posts review comments after approval.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure src is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from sentinel.mcp.github_server import GitHubMCPServer

logger = logging.getLogger(__name__)


@dataclass
class GitHubPRRequest:
    """Input to GitHubPRAgent."""

    session_id: str
    pr_number: int
    repo: str


@dataclass
class GitHubPRResult:
    """Output of GitHubPRAgent."""

    session_id: str
    pr_number: int
    repo: str
    pr_metadata: dict = field(default_factory=dict)
    files: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    review_comment: dict | None = None
    duration_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "pr_number": self.pr_number,
            "repo": self.repo,
            "pr_metadata": self.pr_metadata,
            "files": self.files,
            "checks": self.checks,
            "review_comment": self.review_comment,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class GitHubPRAgent:
    """Subagent for GitHub PR integration."""

    def __init__(self, server: GitHubMCPServer | None = None) -> None:
        self.server = server or GitHubMCPServer()

    async def fetch_pr_context(self, request: GitHubPRRequest) -> GitHubPRResult:
        """Fetch PR metadata, files, and check status from GitHub."""
        start = time.monotonic()
        result = GitHubPRResult(
            session_id=request.session_id,
            pr_number=request.pr_number,
            repo=request.repo,
        )

        try:
            # Fetch PR metadata
            result.pr_metadata = await self.server.handle_tool(
                "get_pr", {"pr_number": request.pr_number}
            )

            # Fetch changed files
            result.files = await self.server.handle_tool(
                "get_pr_files", {"pr_number": request.pr_number}
            )

            # Fetch CI checks
            result.checks = await self.server.handle_tool(
                "get_pr_checks", {"pr_number": request.pr_number}
            )

        except Exception as exc:  # noqa: BLE001 — top-level agent error capture
            result.error = str(exc)
            logger.error("GitHubPRAgent failed: %s", exc)

        result.duration_ms = int((time.monotonic() - start) * 1000)
        return result

    async def post_review(self, pr_number: int, repo: str, body: str, event: str) -> dict:
        """Post a review comment on a PR."""
        return await self.server.handle_tool(
            "post_review_comment",
            {"pr_number": pr_number, "body": body, "event": event},
        )


if __name__ == "__main__":
    # Rule 2903681 self-checks
    import asyncio
    from unittest.mock import AsyncMock

    async def _self_check():
        mock_server = AsyncMock()
        mock_server.handle_tool = AsyncMock(
            side_effect=[
                {"number": 1, "title": "Test", "author": "test", "state": "open"},
                [{"filename": "test.py", "status": "added"}],
                [{"name": "CI", "conclusion": "success"}],
            ]
        )

        agent = GitHubPRAgent(server=mock_server)
        request = GitHubPRRequest(session_id="self_test", pr_number=1, repo="owner/repo")
        result = await agent.fetch_pr_context(request)

        assert result.pr_metadata["number"] == 1
        assert len(result.files) == 1
        assert len(result.checks) == 1
        assert result.error is None

        print("github_pr_agent.py self-check passed.")

    asyncio.run(_self_check())
