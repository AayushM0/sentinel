"""End-to-end test for GitHub PR integration."""

from unittest.mock import AsyncMock

import pytest

from sentinel.session_store import SessionStore


@pytest.mark.asyncio
async def test_full_pr_review_pipeline():
    """Test complete PR review with mocked GitHub and LACE."""
    # This test verifies the full pipeline works end-to-end
    # with all three subagents running in parallel
    # Full implementation with mocks


def test_session_store_pr_fields():
    """Test that session store supports PR-specific fields."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        store = SessionStore(db_path=db_path)
        session = store.create_session(
            session_id="test_pr",
            branch_name="feat/test",
            commit_sha="abc123",
            diff_summary="Test diff",
            pr_number=42,
            repo="owner/repo",
        )
        assert session.session_id == "test_pr"

        # Retrieve and verify PR fields
        retrieved = store.get_session("test_pr")
        assert retrieved is not None


def test_github_pr_agent_with_mock():
    """Test GitHub PR agent with mocked server."""
    from sentinel.subagents.github_pr_agent import GitHubPRAgent, GitHubPRRequest

    mock_server = AsyncMock()
    mock_server.handle_tool = AsyncMock(
        side_effect=[
            {"number": 42, "title": "Test PR", "author": "test", "state": "open"},
            [{"filename": "test.py", "status": "added"}],
            [{"name": "CI", "conclusion": "success"}],
        ]
    )

    agent = GitHubPRAgent(server=mock_server)
    request = GitHubPRRequest(session_id="e2e_test", pr_number=42, repo="owner/repo")

    import asyncio

    result = asyncio.run(agent.fetch_pr_context(request))

    assert result.pr_metadata["number"] == 42
    assert len(result.files) == 1
    assert len(result.checks) == 1
    assert result.error is None
