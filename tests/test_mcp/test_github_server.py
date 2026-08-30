"""Tests for GitHub MCP server tools."""

from unittest.mock import AsyncMock

import pytest

from sentinel.mcp.github_server import GitHubMCPServer
from sentinel.models.github import CheckRun, PRFile, PRMetadata, ReviewComment


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.get_pr = AsyncMock(
        return_value=PRMetadata(
            number=42,
            title="Test PR",
            body="Body",
            author="test",
            state="open",
            head_sha="abc",
            base_branch="main",
            head_branch="feat/test",
        )
    )
    client.get_pr_files = AsyncMock(
        return_value=[
            PRFile(filename="test.py", status="added", patch="+test", additions=1, deletions=0)
        ]
    )
    client.post_review = AsyncMock(
        return_value=ReviewComment(
            id=1,
            html_url="https://example.com",
            body="Review",
            event="APPROVE",
        )
    )
    client.get_pr_checks = AsyncMock(
        return_value=[
            CheckRun(name="CI", status="completed", conclusion="success", output_summary="OK")
        ]
    )
    return client


@pytest.fixture
def server(mock_client):
    return GitHubMCPServer(client=mock_client)


@pytest.mark.asyncio
async def test_get_pr_tool(server, mock_client):
    result = await server.handle_tool("get_pr", {"pr_number": 42})
    assert result["number"] == 42
    mock_client.get_pr.assert_called_once_with(42)


@pytest.mark.asyncio
async def test_get_pr_files_tool(server, mock_client):
    result = await server.handle_tool("get_pr_files", {"pr_number": 42})
    assert len(result) == 1
    assert result[0]["filename"] == "test.py"
    mock_client.get_pr_files.assert_called_once_with(42)


@pytest.mark.asyncio
async def test_post_review_comment_tool(server, mock_client):
    result = await server.handle_tool(
        "post_review_comment",
        {
            "pr_number": 42,
            "body": "## Review\nLGTM",
            "event": "APPROVE",
        },
    )
    assert result["event"] == "APPROVE"
    mock_client.post_review.assert_called_once_with(42, "## Review\nLGTM", "APPROVE")


@pytest.mark.asyncio
async def test_get_pr_checks_tool(server, mock_client):
    result = await server.handle_tool("get_pr_checks", {"pr_number": 42})
    assert len(result) == 1
    assert result[0]["conclusion"] == "success"
    mock_client.get_pr_checks.assert_called_once_with(42)


def test_server_tools():
    server = GitHubMCPServer(client=AsyncMock())
    tools = server.get_tools()
    tool_names = [t["name"] for t in tools]
    assert "get_pr" in tool_names
    assert "get_pr_files" in tool_names
    assert "post_review_comment" in tool_names
    assert "get_pr_checks" in tool_names
