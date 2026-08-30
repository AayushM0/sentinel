"""Tests for GitHub API client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.mcp.github_client import GitHubClient
from sentinel.models.github import CheckRun, PRFile, PRMetadata, ReviewComment


@pytest.fixture
def client():
    return GitHubClient(token="test_token", repo="owner/repo")


@pytest.mark.asyncio
async def test_get_pr(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "number": 42,
        "title": "Add auth",
        "body": "Test PR",
        "user": {"login": "aayushm0"},
        "state": "open",
        "head": {"sha": "abc123", "ref": "feat/auth"},
        "base": {"ref": "main"},
    }

    with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
        pr = await client.get_pr(42)
        assert isinstance(pr, PRMetadata)
        assert pr.number == 42


@pytest.mark.asyncio
async def test_get_pr_files(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "filename": "src/auth.py",
            "status": "added",
            "patch": "+import auth",
            "additions": 10,
            "deletions": 0,
        }
    ]

    with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
        files = await client.get_pr_files(42)
        assert len(files) == 1
        assert isinstance(files[0], PRFile)
        assert files[0].filename == "src/auth.py"


@pytest.mark.asyncio
async def test_post_review_comment(client):
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "id": 12345,
        "html_url": "https://github.com/owner/repo/pull/42#issuecomment-12345",
        "body": "## Review\nAll tests pass",
        "state": "approved",
    }

    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
        comment = await client.post_review(42, "## Review\nAll tests pass", "APPROVE")
        assert isinstance(comment, ReviewComment)
        assert comment.event == "APPROVE"


@pytest.mark.asyncio
async def test_get_pr_checks(client):
    # Mock PR response to get head SHA
    mock_pr_response = MagicMock()
    mock_pr_response.status_code = 200
    mock_pr_response.json.return_value = {
        "head": {"sha": "abc123"},
    }

    # Mock check runs response
    mock_checks_response = MagicMock()
    mock_checks_response.status_code = 200
    mock_checks_response.json.return_value = {
        "check_runs": [
            {
                "name": "CI Build",
                "status": "completed",
                "conclusion": "success",
                "output": {"summary": "All checks passed"},
            }
        ]
    }

    async def mock_get(url, **kwargs):
        if "/pulls/" in url and "check-runs" not in url:
            return mock_pr_response
        return mock_checks_response

    with patch.object(client._client, "get", side_effect=mock_get):
        checks = await client.get_pr_checks(42)
        assert len(checks) == 1
        assert isinstance(checks[0], CheckRun)
        assert checks[0].conclusion == "success"


def test_client_init():
    c = GitHubClient(token="test", repo="owner/repo")
    assert c.token == "test"
    assert c.repo == "owner/repo"
