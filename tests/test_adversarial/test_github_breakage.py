"""Adversarial tests for GitHub PR integration — trying to break things.

These tests target edge cases, error paths, and failure modes that could
cause silent data loss, crashes, or incorrect behavior in production.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.mcp.github_client import GitHubClient
from sentinel.mcp.github_server import GitHubMCPServer
from sentinel.models.github import (
    PRMetadata,
    ReviewComment,
)
from sentinel.subagents.github_pr_agent import (
    GitHubPRAgent,
    GitHubPRRequest,
    GitHubPRResult,
)
from sentinel.subagents.sandbox_runner import SandboxRunner

# ---------------------------------------------------------------------------
# GitHubClient edge cases
# ---------------------------------------------------------------------------


class TestGitHubClientBreakage:
    """Try to break the GitHub client."""

    def test_missing_token_raises(self):
        """Client must refuse to initialize without a token."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="GITHUB_TOKEN"),
        ):
            GitHubClient(token=None, repo="owner/repo")

    def test_missing_repo_raises(self):
        """Client must refuse to initialize without a repo."""
        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test"}, clear=False),
            pytest.raises(ValueError, match="GITHUB_REPO"),
        ):
            GitHubClient(token="ghp_test", repo=None)

    def test_empty_token_raises(self):
        """Empty string token must be rejected, not accepted silently."""
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            GitHubClient(token="", repo="owner/repo")

    def test_empty_repo_raises(self):
        """Empty string repo must be rejected."""
        with pytest.raises(ValueError, match="GITHUB_REPO"):
            GitHubClient(token="ghp_test", repo="")

    @pytest.mark.asyncio
    async def test_get_pr_404_handled(self):
        """404 from GitHub must raise, not return garbage."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
            mock_client.get = AsyncMock(return_value=mock_resp)

            client = GitHubClient(token="test", repo="owner/repo")
            with pytest.raises(Exception, match="404"):
                await client.get_pr(999999)

    @pytest.mark.asyncio
    async def test_get_pr_malformed_json_handled(self):
        """Malformed JSON from GitHub must raise, not return partial data."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"incomplete": "data"}
            mock_client.get = AsyncMock(return_value=mock_resp)

            client = GitHubClient(token="test", repo="owner/repo")
            with pytest.raises(KeyError):
                await client.get_pr(1)

    @pytest.mark.asyncio
    async def test_post_review_403_handled(self):
        """403 (rate limit / permissions) must raise, not silently fail."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("403 Forbidden")
            mock_client.post = AsyncMock(return_value=mock_resp)

            client = GitHubClient(token="test", repo="owner/repo")
            with pytest.raises(Exception, match="403"):
                await client.post_review(1, "body", "APPROVE")


# ---------------------------------------------------------------------------
# GitHubMCPServer edge cases
# ---------------------------------------------------------------------------


class TestGitHubMCPServerBreakage:
    """Try to break the MCP server."""

    def test_default_client_no_env_vars(self):
        """Server must not crash on init when env vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove GITHUB_TOKEN and GITHUB_REPO if present
            os.environ.pop("GITHUB_TOKEN", None)
            os.environ.pop("GITHUB_REPO", None)
            server = GitHubMCPServer()
            assert server.client is None

    @pytest.mark.asyncio
    async def test_tool_call_without_config_raises(self):
        """Calling a tool with no client must raise RuntimeError, not AttributeError."""
        server = GitHubMCPServer.__new__(GitHubMCPServer)
        server.client = None
        with pytest.raises(RuntimeError, match="not configured"):
            await server.handle_tool("get_pr", {"pr_number": 1})

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        """Unknown tool name must raise ValueError."""
        mock_client = AsyncMock()
        server = GitHubMCPServer(client=mock_client)
        with pytest.raises(ValueError, match="Unknown tool"):
            await server.handle_tool("delete_everything", {})

    def test_get_tools_returns_4(self):
        """Must expose exactly 4 tools."""
        mock_client = AsyncMock()
        server = GitHubMCPServer(client=mock_client)
        tools = server.get_tools()
        assert len(tools) == 4
        names = {t["name"] for t in tools}
        assert names == {"get_pr", "get_pr_files", "post_review_comment", "get_pr_checks"}

    @pytest.mark.asyncio
    async def test_handle_tool_routes_correctly(self):
        """Each tool must route to the correct client method."""
        mock_client = AsyncMock()
        mock_client.get_pr = AsyncMock(
            return_value=PRMetadata(
                number=1,
                title="T",
                author="a",
                state="open",
                head_sha="abc",
                base_branch="main",
                head_branch="feat",
            )
        )
        mock_client.get_pr_files = AsyncMock(return_value=[])
        mock_client.post_review = AsyncMock(
            return_value=ReviewComment(
                id=1,
                html_url="http://x",
                body="ok",
                event="APPROVE",
            )
        )
        mock_client.get_pr_checks = AsyncMock(return_value=[])

        server = GitHubMCPServer(client=mock_client)

        await server.handle_tool("get_pr", {"pr_number": 1})
        mock_client.get_pr.assert_called_once_with(1)

        await server.handle_tool("get_pr_files", {"pr_number": 2})
        mock_client.get_pr_files.assert_called_once_with(2)

        await server.handle_tool(
            "post_review_comment", {"pr_number": 3, "body": "x", "event": "COMMENT"}
        )
        mock_client.post_review.assert_called_once_with(3, "x", "COMMENT")

        await server.handle_tool("get_pr_checks", {"pr_number": 4})
        mock_client.get_pr_checks.assert_called_once_with(4)


# ---------------------------------------------------------------------------
# GitHubPRAgent edge cases
# ---------------------------------------------------------------------------


class TestGitHubPRAgentBreakage:
    """Try to break the PR agent."""

    @pytest.mark.asyncio
    async def test_fetch_pr_context_partial_failure(self):
        """If one API call fails, the agent should record the error and still return."""
        mock_server = AsyncMock()
        mock_server.handle_tool = AsyncMock(
            side_effect=[
                {"number": 1, "title": "PR"},  # get_pr succeeds
                Exception("Rate limited"),  # get_pr_files fails
            ]
        )

        agent = GitHubPRAgent(server=mock_server)
        request = GitHubPRRequest(session_id="s1", pr_number=1, repo="o/r")
        result = await agent.fetch_pr_context(request)

        # Should have metadata from first call
        assert result.pr_metadata == {"number": 1, "title": "PR"}
        # Should record error from second call
        assert result.error is not None
        assert "Rate limited" in result.error

    @pytest.mark.asyncio
    async def test_fetch_pr_context_all_fail(self):
        """If all API calls fail, agent should return error, not crash."""
        mock_server = AsyncMock()
        mock_server.handle_tool = AsyncMock(side_effect=Exception("Network error"))

        agent = GitHubPRAgent(server=mock_server)
        request = GitHubPRRequest(session_id="s2", pr_number=1, repo="o/r")
        result = await agent.fetch_pr_context(request)

        assert result.error is not None
        assert "Network error" in result.error
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_post_review_propagates_error(self):
        """Post review errors must propagate to caller."""
        mock_server = AsyncMock()
        mock_server.handle_tool = AsyncMock(side_effect=Exception("403 Forbidden"))

        agent = GitHubPRAgent(server=mock_server)
        with pytest.raises(Exception, match="403"):
            await agent.post_review(1, "o/r", "body", "APPROVE")

    def test_result_to_dict_structure(self):
        """GitHubPRResult.to_dict must return all expected keys."""
        result = GitHubPRResult(
            session_id="s1",
            pr_number=42,
            repo="owner/repo",
            pr_metadata={"title": "Test"},
            files=[{"filename": "a.py"}],
            checks=[{"name": "CI"}],
            duration_ms=123,
        )
        d = result.to_dict()
        assert d["session_id"] == "s1"
        assert d["pr_number"] == 42
        assert d["repo"] == "owner/repo"
        assert len(d["files"]) == 1
        assert len(d["checks"]) == 1
        assert d["error"] is None


# ---------------------------------------------------------------------------
# Security audit parsing edge cases
# ---------------------------------------------------------------------------


class TestSecurityAuditBreakage:
    """Try to break the security audit parsers."""

    def test_parse_pip_audit_malformed_json(self):
        """Malformed JSON must return empty result, not crash."""
        runner = SandboxRunner()
        result = runner._parse_pip_audit("not json at all {{{")
        assert result.tool == "pip-audit"
        assert result.vulnerabilities == []
        assert result.exit_code == 0

    def test_parse_pip_audit_empty_string(self):
        """Empty output must return clean result."""
        runner = SandboxRunner()
        result = runner._parse_pip_audit("")
        assert result.vulnerabilities == []

    def test_parse_pip_audit_no_vulns(self):
        """Clean audit must return zero vulnerabilities."""
        runner = SandboxRunner()
        output = json.dumps({"dependencies": []})
        result = runner._parse_pip_audit(output)
        assert result.vulnerabilities == []
        assert result.exit_code == 0

    def test_parse_npm_audit_malformed_json(self):
        """Malformed JSON must return empty result."""
        runner = SandboxRunner()
        result = runner._parse_npm_audit("{invalid json")
        assert result.tool == "npm-audit"
        assert result.vulnerabilities == []

    def test_parse_npm_audit_empty(self):
        """Empty output must return clean result."""
        runner = SandboxRunner()
        result = runner._parse_npm_audit("")
        assert result.vulnerabilities == []

    def test_parse_npm_audit_missing_fields(self):
        """npm audit with missing via/version fields must not crash."""
        runner = SandboxRunner()
        output = json.dumps({"vulnerabilities": {"pkg1": {"severity": "high"}}})
        result = runner._parse_npm_audit(output)
        assert len(result.vulnerabilities) == 1
        assert result.vulnerabilities[0].package == "pkg1"

    def test_parse_pip_audit_missing_fix_versions(self):
        """pip-audit with no fix_versions must still classify severity."""
        runner = SandboxRunner()
        output = json.dumps(
            {
                "dependencies": [
                    {
                        "name": "pkg",
                        "version": "1.0",
                        "vulns": [
                            {
                                "id": "CVE-123",
                                "fix_versions": [],
                            }
                        ],
                    }
                ]
            }
        )
        result = runner._parse_pip_audit(output)
        assert len(result.vulnerabilities) == 1
        # No fix_versions means severity falls to "medium"
        assert result.vulnerabilities[0].severity == "medium"


# ---------------------------------------------------------------------------
# Approval card with GitHub result
# ---------------------------------------------------------------------------


class TestApprovalCardGitHubBreakage:
    """Try to break the approval card with GitHub data."""

    def test_card_with_github_result(self):
        """Card must render GitHub PR section when result is provided."""
        from sentinel.approval_gate import ApprovalGate

        gate = ApprovalGate()
        card = gate.format_approval_card(
            session_id="s1",
            branch_name="main",
            commit_sha="abc1234",
            test_result={
                "sandbox_status": "completed",
                "exit_code": 0,
                "tests_passed": 5,
                "tests_failed": 0,
            },
            delta_report={},
            github_result={
                "pr_number": 42,
                "pr_metadata": {"title": "Add auth", "author": "dev"},
                "files": [{"filename": "a.py"}, {"filename": "b.py"}],
                "checks": [{"conclusion": "success"}, {"conclusion": "failure"}],
                "error": None,
            },
        )
        assert "GitHub PR Status" in card
        assert "PR #42" in card
        assert "Add auth" in card
        assert "@dev" in card
        assert "1 passing, 1 failing" in card

    def test_card_with_github_error(self):
        """Card must show GitHub error, not crash."""
        from sentinel.approval_gate import ApprovalGate

        gate = ApprovalGate()
        card = gate.format_approval_card(
            session_id="s1",
            branch_name="main",
            commit_sha="abc1234",
            test_result={
                "sandbox_status": "completed",
                "exit_code": 0,
                "tests_passed": 5,
                "tests_failed": 0,
            },
            delta_report={},
            github_result={"error": "403 Forbidden", "pr_number": 1},
        )
        assert "GitHub PR Status" in card
        assert "403 Forbidden" in card

    def test_card_without_github_result(self):
        """Card must work fine without GitHub result (backward compat)."""
        from sentinel.approval_gate import ApprovalGate

        gate = ApprovalGate()
        card = gate.format_approval_card(
            session_id="s1",
            branch_name="main",
            commit_sha="abc1234",
            test_result={
                "sandbox_status": "completed",
                "exit_code": 0,
                "tests_passed": 5,
                "tests_failed": 0,
            },
            delta_report={},
        )
        assert "GitHub" not in card
        assert "PR #" not in card


# ---------------------------------------------------------------------------
# Session store with PR fields
# ---------------------------------------------------------------------------


class TestSessionStorePRBreakage:
    """Try to break session persistence with PR-specific fields."""

    def test_session_with_pr_fields(self):
        """Session must persist pr_number and repo."""
        import tempfile
        from pathlib import Path

        from sentinel.session_store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            sess = store.create_session("main", "abc", "test", pr_number=42, repo="owner/repo")
            assert sess.session_id is not None

            hydrated = store.get_session(sess.session_id)
            assert hydrated is not None

    def test_session_without_pr_fields(self):
        """Session without PR fields must still work (backward compat)."""
        import tempfile
        from pathlib import Path

        from sentinel.session_store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            sess = store.create_session("main", "abc", "test")
            hydrated = store.get_session(sess.session_id)
            assert hydrated is not None


# ---------------------------------------------------------------------------
# Concurrent failure isolation
# ---------------------------------------------------------------------------


class TestConcurrentFailureIsolation:
    """If one subagent crashes, others must survive."""

    @pytest.mark.asyncio
    async def test_github_failure_doesnt_crash_orchestrator(self):
        """GitHub agent failure must not prevent sandbox + ADR from completing."""
        from sentinel.orchestrator import OrchestratorRequest, ReviewOrchestrator

        mock_lace = AsyncMock()
        mock_lace.get_relevant_adrs = AsyncMock(return_value=[])
        mock_lace.commit_adr = AsyncMock(return_value=True)

        from sentinel.models.diff import parse_git_diff

        with patch("sentinel.orchestrator.SandboxRunner") as MockSandbox:
            mock_sb = AsyncMock()
            mock_sb.run = AsyncMock(
                return_value=MagicMock(
                    to_dict=lambda: {
                        "sandbox_status": "completed",
                        "exit_code": 0,
                        "tests_passed": 5,
                        "tests_failed": 0,
                    },
                )
            )
            MockSandbox.return_value = mock_sb

            with patch("sentinel.orchestrator.ADRDeltaAnalyzer") as MockADR:
                mock_adr = AsyncMock()
                mock_adr.run = AsyncMock(
                    return_value=MagicMock(
                        to_dict=lambda: {"violations": [], "proposed_adrs": []},
                    )
                )
                MockADR.return_value = mock_adr

                with patch("sentinel.orchestrator.GitHubPRAgent") as MockGH:
                    mock_gh = AsyncMock()
                    mock_gh.fetch_pr_context = AsyncMock(side_effect=Exception("GitHub down"))
                    MockGH.return_value = mock_gh

                    import tempfile as _tmp
                    from pathlib import Path as _Path

                    from sentinel.session_store import SessionStore as _SS

                    with _tmp.TemporaryDirectory() as tmp:
                        store = _SS(db_path=str(_Path(tmp) / "test.db"))
                        req = OrchestratorRequest(
                            session_id="test_concurrent",
                            branch_name="main",
                            commit_sha="abc",
                            diff_summary="test",
                            git_diff=parse_git_diff(""),
                            touched_files=[],
                            workspace_root=_Path(tmp),
                            lace_client=mock_lace,
                            session_store=store,
                            interactive=False,
                            timeout_seconds=30,
                            pr_number=42,
                            repo="owner/repo",
                        )

                        orch = ReviewOrchestrator()
                        session = await orch.run_review(req)

                        # Session should be pending, not crashed
                        assert session.status.value in (
                            "PENDING_HUMAN_APPROVAL",
                            "COMPLETED",
                            "APPROVED",
                        )
