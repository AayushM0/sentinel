"""Tests for GitHub PR integration subagent."""

from unittest.mock import AsyncMock

import pytest

from sentinel.subagents.github_pr_agent import GitHubPRAgent, GitHubPRRequest, GitHubPRResult


@pytest.fixture
def mock_server():
    server = AsyncMock()
    server.handle_tool = AsyncMock(
        side_effect=[
            # get_pr
            {"number": 42, "title": "Test PR", "author": "test", "state": "open"},
            # get_pr_files
            [{"filename": "test.py", "status": "added"}],
            # get_pr_checks
            [{"name": "CI", "conclusion": "success"}],
        ]
    )
    return server


@pytest.mark.asyncio
async def test_fetch_pr_context(mock_server):
    agent = GitHubPRAgent(server=mock_server)
    request = GitHubPRRequest(session_id="test", pr_number=42, repo="owner/repo")
    result = await agent.fetch_pr_context(request)
    assert isinstance(result, GitHubPRResult)
    assert result.pr_metadata["number"] == 42
    assert len(result.files) == 1
    assert len(result.checks) == 1


@pytest.mark.asyncio
async def test_post_review(mock_server):
    mock_server.handle_tool = AsyncMock(return_value={"id": 1, "event": "APPROVE"})
    agent = GitHubPRAgent(server=mock_server)
    comment = await agent.post_review(42, "owner/repo", "## Review\nLGTM", "APPROVE")
    assert comment["event"] == "APPROVE"


def test_agent_init():
    agent = GitHubPRAgent(server=AsyncMock())
    assert agent.server is not None


def test_result_to_dict():
    result = GitHubPRResult(
        session_id="test",
        pr_number=42,
        repo="owner/repo",
        pr_metadata={"number": 42},
        files=[{"filename": "test.py"}],
        checks=[{"name": "CI"}],
    )
    d = result.to_dict()
    assert d["session_id"] == "test"
    assert d["pr_number"] == 42
    assert len(d["files"]) == 1
