"""MCP server wrapping GitHub REST API tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from sentinel.mcp.github_client import GitHubClient

# Ensure src is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


class GitHubMCPServer:
    """MCP server providing GitHub PR tools."""

    def __init__(self, client: GitHubClient | None = None) -> None:
        if client is not None:
            self.client = client
        elif os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO"):
            self.client = GitHubClient()
        else:
            self.client = None  # Lazy: will raise on tool call if not configured

    def get_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "get_pr",
                "description": "Fetch PR metadata from GitHub",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pr_number": {"type": "integer", "description": "PR number to fetch"},
                    },
                    "required": ["pr_number"],
                },
            },
            {
                "name": "get_pr_files",
                "description": "Fetch files changed in a PR",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pr_number": {"type": "integer", "description": "PR number"},
                    },
                    "required": ["pr_number"],
                },
            },
            {
                "name": "post_review_comment",
                "description": "Post a review comment on a PR",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pr_number": {"type": "integer", "description": "PR number"},
                        "body": {"type": "string", "description": "Review body (Markdown)"},
                        "event": {
                            "type": "string",
                            "enum": ["APPROVE", "REQUEST_CHANGES", "COMMENT"],
                            "description": "Review event type",
                        },
                    },
                    "required": ["pr_number", "body", "event"],
                },
            },
            {
                "name": "get_pr_checks",
                "description": "Fetch CI check runs for a PR",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pr_number": {"type": "integer", "description": "PR number"},
                    },
                    "required": ["pr_number"],
                },
            },
        ]

    async def handle_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Route tool call to GitHub client method."""
        if self.client is None:
            raise RuntimeError(
                "GitHub MCP server not configured: set GITHUB_TOKEN and GITHUB_REPO in .env"
            )
        if tool_name == "get_pr":
            pr = await self.client.get_pr(arguments["pr_number"])
            return pr.model_dump()
        elif tool_name == "get_pr_files":
            files = await self.client.get_pr_files(arguments["pr_number"])
            return [f.model_dump() for f in files]
        elif tool_name == "post_review_comment":
            comment = await self.client.post_review(
                arguments["pr_number"],
                arguments["body"],
                arguments["event"],
            )
            return comment.model_dump()
        elif tool_name == "get_pr_checks":
            checks = await self.client.get_pr_checks(arguments["pr_number"])
            return [c.model_dump() for c in checks]
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    # Rule 2903681 self-checks
    import asyncio
    from unittest.mock import AsyncMock

    from sentinel.models.github import PRMetadata

    async def _self_check():
        mock_client = AsyncMock()
        mock_client.get_pr = AsyncMock(
            return_value=PRMetadata(
                number=1,
                title="Test",
                body=None,
                author="test",
                state="open",
                head_sha="abc",
                base_branch="main",
                head_branch="feat/test",
            )
        )

        server = GitHubMCPServer(client=mock_client)
        tools = server.get_tools()
        assert len(tools) == 4
        tool_names = [t["name"] for t in tools]
        assert "get_pr" in tool_names

        result = await server.handle_tool("get_pr", {"pr_number": 1})
        assert result["number"] == 1

        print("github_server.py self-check passed.")

    asyncio.run(_self_check())
