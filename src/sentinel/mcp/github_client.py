"""Async GitHub REST API client."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Self

import httpx

from sentinel.models.github import CheckRun, PRFile, PRMetadata, ReviewComment

# Ensure src is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


class GitHubClient:
    """Async client for GitHub REST API v3."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None, repo: str | None = None) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.repo = repo or os.environ.get("GITHUB_REPO", "")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        if not self.repo:
            raise ValueError("GITHUB_REPO environment variable is required")

        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def get_pr(self, pr_number: int) -> PRMetadata:
        """Fetch PR metadata from GitHub."""
        response = await self._client.get(f"/repos/{self.repo}/pulls/{pr_number}")
        response.raise_for_status()
        data = response.json()
        return PRMetadata(
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            author=data["user"]["login"],
            state=data["state"],
            head_sha=data["head"]["sha"],
            base_branch=data["base"]["ref"],
            head_branch=data["head"]["ref"],
        )

    async def get_pr_files(self, pr_number: int) -> list[PRFile]:
        """Fetch all files changed in a PR, following pagination."""
        all_files: list[PRFile] = []
        page = 1
        while True:
            response = await self._client.get(
                f"/repos/{self.repo}/pulls/{pr_number}/files",
                params={"per_page": 100, "page": page},
            )
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_files.extend(PRFile(**f) for f in data)
            if len(data) < 100:
                break
            page += 1
        return all_files

    async def post_review(self, pr_number: int, body: str, event: str) -> ReviewComment:
        """Post a review comment on a PR."""
        response = await self._client.post(
            f"/repos/{self.repo}/pulls/{pr_number}/reviews",
            json={"body": body, "event": event},
        )
        response.raise_for_status()
        data = response.json()
        return ReviewComment(
            id=data["id"],
            html_url=data["html_url"],
            body=data["body"],
            event=event,
        )

    async def get_pr_checks(self, pr_number: int) -> list[CheckRun]:
        """Fetch all CI check runs for a PR, following pagination."""
        # First get the PR to find the head SHA
        pr_response = await self._client.get(f"/repos/{self.repo}/pulls/{pr_number}")
        pr_response.raise_for_status()
        head_sha = pr_response.json()["head"]["sha"]

        # Then get check runs for that commit, paginated
        all_checks: list[CheckRun] = []
        page = 1
        while True:
            response = await self._client.get(
                f"/repos/{self.repo}/commits/{head_sha}/check-runs",
                params={"per_page": 100, "page": page},
            )
            response.raise_for_status()
            data = response.json()
            runs = data.get("check_runs", [])
            if not runs:
                break
            all_checks.extend(
                CheckRun(
                    name=cr["name"],
                    status=cr["status"],
                    conclusion=cr.get("conclusion"),
                    output_summary=cr.get("output", {}).get("summary"),
                )
                for cr in runs
            )
            if len(runs) < 100:
                break
            page += 1
        return all_checks


if __name__ == "__main__":
    # Rule 2903681 self-checks (with mocked HTTP)
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    async def _self_check():
        # Mock httpx client
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            # Test get_pr
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "number": 1,
                "title": "Test",
                "body": None,
                "user": {"login": "test"},
                "state": "open",
                "head": {"sha": "abc", "ref": "feat/test"},
                "base": {"ref": "main"},
            }
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.post = AsyncMock(return_value=mock_resp)

            client = GitHubClient(token="test_token", repo="owner/repo")
            pr = await client.get_pr(1)
            assert pr.number == 1
            assert pr.author == "test"

            print("github_client.py self-check passed.")

    asyncio.run(_self_check())
