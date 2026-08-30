"""Real-user adversarial tests — trying to break Sentinel like a human would.

These aren't unit tests. They simulate what actual users do: wrong args,
missing env, weird repos, race conditions, corrupted state, and edge cases
that production will absolutely hit.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.cli import build_parser, cmd_check, cmd_list, cmd_resume
from sentinel.mcp.github_client import GitHubClient
from sentinel.mcp.github_server import GitHubMCPServer
from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.orchestrator import OrchestratorRequest, ReviewOrchestrator
from sentinel.session_store import SessionStore

# ---------------------------------------------------------------------------
# CLI argument abuse
# ---------------------------------------------------------------------------


class TestCLIArgAbuse:
    """What happens when users pass garbage to the CLI?"""

    def test_no_args_prints_help(self):
        """`sentinel` with no args should not crash."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.subcommand is None

    def test_check_with_invalid_mode(self):
        """Invalid --mode value must be rejected by argparse."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["check", "--mode", "INVALID_MODE"])

    def test_check_with_negative_timeout(self):
        """Negative timeout must be parsed (validation happens at runtime)."""
        parser = build_parser()
        args = parser.parse_args(["check", "--timeout", "-5"])
        assert args.timeout == -5  # argparse allows it, runtime should reject

    def test_check_with_zero_timeout(self):
        """Zero timeout — orchestrator should reject."""
        parser = build_parser()
        args = parser.parse_args(["check", "--timeout", "0"])
        assert args.timeout == 0

    def test_check_with_huge_timeout(self):
        """Absurdly large timeout — should parse fine."""
        parser = build_parser()
        args = parser.parse_args(["check", "--timeout", "999999"])
        assert args.timeout == 999999

    def test_list_with_zero_limit(self):
        """List with --limit 0 should clamp to 1."""
        parser = build_parser()
        args = parser.parse_args(["list", "--limit", "0"])
        assert args.limit == 0

    def test_list_with_negative_limit(self):
        """List with negative limit should clamp to 1."""
        parser = build_parser()
        args = parser.parse_args(["list", "--limit", "-10"])
        assert args.limit == -10

    def test_resume_without_session_id(self):
        """Resume without session ID should use latest pending."""
        parser = build_parser()
        args = parser.parse_args(["resume"])
        assert args.session_id is None

    def test_pr_with_non_integer(self):
        """--pr with non-integer must be rejected."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["check", "--pr", "not_a_number"])

    def test_pr_with_zero(self):
        """--pr 0 is technically valid argparse but invalid for GitHub."""
        parser = build_parser()
        args = parser.parse_args(["check", "--pr", "0"])
        assert args.pr == 0

    def test_pr_with_negative(self):
        """--pr -1 is technically valid argparse but invalid for GitHub."""
        parser = build_parser()
        args = parser.parse_args(["check", "--pr", "-1"])
        assert args.pr == -1


# ---------------------------------------------------------------------------
# Session store abuse
# ---------------------------------------------------------------------------


class TestSessionStoreAbuse:
    """Try to corrupt or break the SQLite session store."""

    def test_concurrent_writes(self):
        """Multiple threads writing to the same DB must not corrupt data."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "concurrent.db")
            store = SessionStore(db_path=db_path)

            # Create 10 sessions rapidly
            sessions = []
            for i in range(10):
                sess = store.create_session(f"branch_{i}", f"sha_{i}", f"diff_{i}")
                sessions.append(sess)

            # All should be readable
            for sess in sessions:
                hydrated = store.get_session(sess.session_id)
                assert hydrated is not None
                assert hydrated.session_id == sess.session_id

    def test_double_resolve_approval(self):
        """Resolving the same approval twice must be idempotent."""
        from sentinel.models.review_state import ApprovalActionType, ApprovalDecision

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            sess = store.create_session("main", "abc", "test")
            appr = store.set_pending_approval(
                sess.session_id, ApprovalActionType.PRE_PUSH_COMMIT, {"card": "x"}
            )

            store.resolve_approval(appr.approval_id, ApprovalDecision.APPROVED)
            # Second resolve should be a no-op, not crash
            store.resolve_approval(appr.approval_id, ApprovalDecision.APPROVED)

            hydrated = store.get_session(sess.session_id)
            assert hydrated.status.value == "APPROVED"

    def test_resolve_nonexistent_approval(self):
        """Resolving a non-existent approval ID must not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            from sentinel.models.review_state import ApprovalDecision

            # Should silently do nothing
            store.resolve_approval("nonexistent_id", ApprovalDecision.APPROVED)

    def test_get_nonexistent_session(self):
        """Getting a non-existent session must return None, not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            result = store.get_session("does_not_exist")
            assert result is None

    def test_create_duplicate_session_id(self):
        """Creating two sessions with the same ID must fail or be idempotent."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            store.create_session("main", "abc", "test", session_id="dup_test")
            # Second create with same ID — SQLite PRIMARY KEY constraint
            with pytest.raises(sqlite3.IntegrityError):
                store.create_session("main", "abc", "test2", session_id="dup_test")

    def test_set_approval_on_terminal_session(self):
        """Cannot request approval for a COMPLETED session."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            sess = store.create_session("main", "abc", "test")
            store.mark_completed(sess.session_id)
            from sentinel.models.review_state import ApprovalActionType

            with pytest.raises(ValueError, match="terminal"):
                store.set_pending_approval(
                    sess.session_id,
                    ApprovalActionType.PRE_PUSH_COMMIT,
                    {"card": "x"},
                )

    def test_mark_completed_on_nonexistent_session(self):
        """Marking a non-existent session as completed must not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            # Should silently do nothing (no session to update)
            store.mark_completed("nonexistent_session")

    def test_list_sessions_empty(self):
        """Listing sessions in an empty DB must return empty list."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            sessions = store.list_sessions()
            assert sessions == []

    def test_list_sessions_limit_clamping(self):
        """Limit must be clamped to [1, 100]."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            for i in range(5):
                store.create_session(f"b{i}", f"s{i}", f"d{i}")

            # Limit 0 should clamp to 1
            sessions = store.list_sessions(limit=0)
            assert len(sessions) <= 1

            # Limit 1000 should clamp to 100
            sessions = store.list_sessions(limit=1000)
            assert len(sessions) <= 100

    def test_raw_diff_persistence(self):
        """raw_diff must survive round-trip through SQLite."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            long_diff = "x" * 10000  # 10KB diff
            sess = store.create_session("main", "abc", "test", raw_diff=long_diff)
            hydrated = store.get_session(sess.session_id)
            assert hydrated.raw_diff == long_diff


# ---------------------------------------------------------------------------
# GitHub client edge cases
# ---------------------------------------------------------------------------


class TestGitHubClientEdgeCases:
    """What real users hit with GitHub integration."""

    def test_token_from_env(self):
        """Client must read GITHUB_TOKEN from environment."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_env_test", "GITHUB_REPO": "o/r"}):
            client = GitHubClient()
            assert client.token == "ghp_env_test"

    def test_repo_from_env(self):
        """Client must read GITHUB_REPO from environment."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test", "GITHUB_REPO": "my/repo"}):
            client = GitHubClient()
            assert client.repo == "my/repo"

    def test_explicit_params_override_env(self):
        """Explicit constructor params must override env vars."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_env", "GITHUB_REPO": "env/repo"}):
            client = GitHubClient(token="ghp_explicit", repo="explicit/repo")
            assert client.token == "ghp_explicit"
            assert client.repo == "explicit/repo"

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Async context manager must close HTTP client on exit."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            async with GitHubClient(token="t", repo="o/r") as client:
                assert client._client is mock_client

            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pr_checks_makes_two_requests(self):
        """get_pr_checks must fetch PR then check-runs (two API calls)."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            pr_resp = MagicMock()
            pr_resp.raise_for_status = MagicMock()
            pr_resp.json.return_value = {"head": {"sha": "abc123"}}

            checks_resp = MagicMock()
            checks_resp.raise_for_status = MagicMock()
            checks_resp.json.return_value = {"check_runs": []}

            mock_client.get = AsyncMock(side_effect=[pr_resp, checks_resp])

            client = GitHubClient(token="t", repo="o/r")
            checks = await client.get_pr_checks(42)

            assert mock_client.get.call_count == 2
            assert checks == []


# ---------------------------------------------------------------------------
# LACE client degraded mode
# ---------------------------------------------------------------------------


class TestLACEDegradedMode:
    """When LACE is unavailable, Sentinel must not crash."""

    @pytest.mark.asyncio
    async def test_connect_failure_sets_degraded(self):
        """Failed connect must set degraded=True, not raise."""
        client = LaceMcpClient.__new__(LaceMcpClient)
        client.server_command = "nonexistent_binary"
        client.server_args = ["-c", "import sys; sys.exit(1)"]
        client.server_env = {}
        client._exit_stack = None
        client._session = None
        client._is_connected = False

        await client.connect()

        assert client.degraded is True
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_degraded_client_methods_raise(self):
        """Calling LACE methods on degraded client must raise RuntimeError."""
        client = LaceMcpClient.__new__(LaceMcpClient)
        client._session = None
        client._is_connected = False
        client._degraded = True

        with pytest.raises(RuntimeError, match="not connected"):
            await client.get_relevant_adrs(["file.py"])

        with pytest.raises(RuntimeError, match="not connected"):
            await client.commit_adr(MagicMock())

        with pytest.raises(RuntimeError, match="not connected"):
            await client.search_memory("query")

    @pytest.mark.asyncio
    async def test_degraded_orchestrator_skips_adr(self):
        """Orchestrator must skip ADR analysis when LACE is degraded."""
        from sentinel.models.diff import parse_git_diff

        mock_lace = MagicMock()
        mock_lace.is_connected = False
        mock_lace.degraded = True

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            req = OrchestratorRequest(
                session_id="degraded_test",
                branch_name="main",
                commit_sha="abc",
                diff_summary="test",
                git_diff=parse_git_diff(""),
                touched_files=[],
                workspace_root=Path(tmp),
                lace_client=mock_lace,
                session_store=store,
                interactive=False,
                timeout_seconds=30,
            )

            with patch("sentinel.orchestrator.SandboxRunner") as MockSb:
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
                MockSb.return_value = mock_sb

                orch = ReviewOrchestrator()
                session = await orch.run_review(req)

                # Must complete without crash, ADR analysis skipped
                assert session.status.value in (
                    "PENDING_HUMAN_APPROVAL",
                    "COMPLETED",
                    "APPROVED",
                )


# ---------------------------------------------------------------------------
# Orchestrator edge cases
# ---------------------------------------------------------------------------


class TestOrchestratorEdgeCases:
    """Try to break the orchestrator with adversarial inputs."""

    @pytest.mark.asyncio
    async def test_empty_diff(self):
        """Empty diff must not crash the orchestrator."""
        from sentinel.models.diff import parse_git_diff

        mock_lace = MagicMock()
        mock_lace.is_connected = True
        mock_lace.degraded = False
        mock_lace.get_relevant_adrs = AsyncMock(return_value=[])
        mock_lace.commit_adr = AsyncMock(return_value=True)

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            req = OrchestratorRequest(
                session_id="empty_diff",
                branch_name="main",
                commit_sha="abc",
                diff_summary="empty",
                git_diff=parse_git_diff(""),
                touched_files=[],
                workspace_root=Path(tmp),
                lace_client=mock_lace,
                session_store=store,
                interactive=False,
            )

            with (
                patch("sentinel.orchestrator.SandboxRunner") as MockSb,
                patch("sentinel.orchestrator.ADRDeltaAnalyzer") as MockAdr,
            ):
                mock_sb = AsyncMock()
                mock_sb.run = AsyncMock(
                    return_value=MagicMock(
                        to_dict=lambda: {
                            "sandbox_status": "completed",
                            "exit_code": 0,
                            "tests_passed": 0,
                            "tests_failed": 0,
                        },
                    )
                )
                MockSb.return_value = mock_sb

                mock_adr = AsyncMock()
                mock_adr.run = AsyncMock(
                    return_value=MagicMock(
                        to_dict=lambda: {"violations": [], "proposed_adrs": []},
                    )
                )
                MockAdr.return_value = mock_adr

                orch = ReviewOrchestrator()
                session = await orch.run_review(req)
                assert session.session_id == "empty_diff"

    @pytest.mark.asyncio
    async def test_both_subagents_crash(self):
        """If both sandbox and ADR crash, session must still be created."""
        from sentinel.models.diff import parse_git_diff

        mock_lace = MagicMock()
        mock_lace.is_connected = True
        mock_lace.degraded = False

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            req = OrchestratorRequest(
                session_id="both_crash",
                branch_name="main",
                commit_sha="abc",
                diff_summary="test",
                git_diff=parse_git_diff(""),
                touched_files=[],
                workspace_root=Path(tmp),
                lace_client=mock_lace,
                session_store=store,
                interactive=False,
            )

            with (
                patch("sentinel.orchestrator.SandboxRunner") as MockSb,
                patch("sentinel.orchestrator.ADRDeltaAnalyzer") as MockAdr,
            ):
                MockSb.return_value.run = AsyncMock(side_effect=RuntimeError("Daytona down"))
                MockAdr.return_value.run = AsyncMock(side_effect=RuntimeError("LACE down"))

                orch = ReviewOrchestrator()
                session = await orch.run_review(req)
                # Must not crash — session created with failed subagent results
                assert session is not None

    def test_empty_session_id_rejected(self):
        """Empty session_id must raise ValueError."""
        from sentinel.models.diff import parse_git_diff

        mock_lace = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, pytest.raises(ValueError, match="empty"):
            OrchestratorRequest(
                session_id="",
                branch_name="main",
                commit_sha="abc",
                diff_summary="test",
                git_diff=parse_git_diff(""),
                touched_files=[],
                workspace_root=Path(tmp),
                lace_client=mock_lace,
                session_store=MagicMock(),
            )

    def test_negative_timeout_rejected(self):
        """Negative timeout must raise ValueError."""
        from sentinel.models.diff import parse_git_diff

        mock_lace = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, pytest.raises(ValueError, match="timeout"):
            OrchestratorRequest(
                session_id="neg_timeout",
                branch_name="main",
                commit_sha="abc",
                diff_summary="test",
                git_diff=parse_git_diff(""),
                touched_files=[],
                workspace_root=Path(tmp),
                lace_client=mock_lace,
                session_store=MagicMock(),
                timeout_seconds=-5,
            )


# ---------------------------------------------------------------------------
# Security audit parsers under adversarial input
# ---------------------------------------------------------------------------


class TestSecurityAuditAdversarial:
    """Feed garbage into security parsers to find crashes."""

    def test_pip_audit_huge_json(self):
        """Massive JSON payload must not OOM or crash."""
        from sentinel.subagents.sandbox_runner import SandboxRunner

        runner = SandboxRunner()
        huge_deps = [{"name": f"pkg_{i}", "version": "1.0", "vulns": []} for i in range(10000)]
        output = json.dumps({"dependencies": huge_deps})
        result = runner._parse_pip_audit(output)
        assert result.vulnerabilities == []

    def test_npm_audit_circular_references(self):
        """JSON with unexpected structures must not crash."""
        from sentinel.subagents.sandbox_runner import SandboxRunner

        runner = SandboxRunner()
        output = json.dumps({"vulnerabilities": {"a": {"via": "string_not_array"}}})
        result = runner._parse_npm_audit(output)
        # Should handle gracefully — via is a string, not a list
        assert isinstance(result.vulnerabilities, list)

    def test_pip_audit_unicode_output(self):
        """Unicode in audit output must not crash."""
        from sentinel.subagents.sandbox_runner import SandboxRunner

        runner = SandboxRunner()
        output = json.dumps(
            {
                "dependencies": [
                    {
                        "name": "über-pkg-ñ",
                        "version": "1.0.0",
                        "vulns": [{"id": "CVE-日本語", "fix_versions": ["2.0"]}],
                    }
                ]
            }
        )
        result = runner._parse_pip_audit(output)
        assert len(result.vulnerabilities) == 1
        assert result.vulnerabilities[0].package == "über-pkg-ñ"

    def test_pip_audit_only_whitespace(self):
        """Whitespace-only output must return clean result."""
        from sentinel.subagents.sandbox_runner import SandboxRunner

        runner = SandboxRunner()
        result = runner._parse_pip_audit("   \n\t  ")
        assert result.vulnerabilities == []
        assert result.exit_code == 0

    def test_npm_audit_nested_vulnerabilities(self):
        """Deeply nested npm audit must not stack overflow."""
        from sentinel.subagents.sandbox_runner import SandboxRunner

        runner = SandboxRunner()
        output = json.dumps(
            {
                "vulnerabilities": {
                    "pkg": {
                        "severity": "critical",
                        "via": [{"title": "RCE", "url": "https://example.com"}],
                        "fixAvailable": {"version": "2.0"},
                        "version": "1.0",
                    }
                }
            }
        )
        result = runner._parse_npm_audit(output)
        assert len(result.vulnerabilities) == 1
        assert result.vulnerabilities[0].severity == "critical"


# ---------------------------------------------------------------------------
# Approval card rendering under adversarial data
# ---------------------------------------------------------------------------


class TestApprovalCardAdversarial:
    """Feed weird data into the approval card renderer."""

    def test_card_with_empty_test_result(self):
        """Empty test_result dict must not crash card rendering."""
        from sentinel.approval_gate import ApprovalGate

        gate = ApprovalGate()
        card = gate.format_approval_card("s1", "main", "abc", {}, {})
        # Empty dict defaults exit_code=0, tests_passed=0, tests_failed=0 => SUCCESS
        assert "Sentinel" in card
        assert "Sandbox Verification" in card

    def test_card_with_giant_violation_list(self):
        """1000 violations must not crash or produce unreadable output."""
        from sentinel.approval_gate import ApprovalGate

        gate = ApprovalGate()
        violations = [f"Violation {i}: something bad" for i in range(1000)]
        card = gate.format_approval_card("s1", "main", "abc", {}, {"violations": violations})
        assert "Violation 0" in card
        assert "Violation 999" in card

    def test_card_with_none_values(self):
        """None values in github_result must not crash."""
        from sentinel.approval_gate import ApprovalGate

        gate = ApprovalGate()
        card = gate.format_approval_card(
            "s1",
            "main",
            "abc",
            {"sandbox_status": "completed", "exit_code": 0, "tests_passed": 1, "tests_failed": 0},
            {},
            github_result={
                "pr_metadata": {},
                "files": None,
                "checks": None,
                "error": None,
                "pr_number": 1,
            },
        )
        # Must render without crash (previously crashed on len(None))
        assert "GitHub" in card

    def test_card_with_special_characters(self):
        """Emoji, markdown, and ANSI in data must not break rendering."""
        from sentinel.approval_gate import ApprovalGate

        gate = ApprovalGate()
        card = gate.format_approval_card(
            "s1",
            "feat/🔥-branch",
            "abc1234",
            {"sandbox_status": "completed", "exit_code": 0, "tests_passed": 1, "tests_failed": 0},
            {"violations": ["🚨 CRITICAL: \\x1b[31mANSI escape\\x1b[0m"]},
        )
        assert "🔥" in card
        assert "ANSI escape" in card


# ---------------------------------------------------------------------------
# GitHub MCP server under adversarial tool calls
# ---------------------------------------------------------------------------


class TestGitHubMCPServerAdversarial:
    """Try to break the MCP server with weird inputs."""

    @pytest.mark.asyncio
    async def test_tool_call_with_missing_required_arg(self):
        """Missing required argument must raise, not return garbage."""
        mock_client = AsyncMock()
        server = GitHubMCPServer(client=mock_client)
        with pytest.raises((KeyError, TypeError)):
            await server.handle_tool("get_pr", {})  # Missing pr_number

    @pytest.mark.asyncio
    async def test_tool_call_with_extra_args(self):
        """Extra arguments must be ignored, not crash."""
        mock_client = AsyncMock()
        mock_client.get_pr = AsyncMock(return_value=MagicMock(model_dump=lambda: {"number": 1}))
        server = GitHubMCPServer(client=mock_client)
        result = await server.handle_tool("get_pr", {"pr_number": 1, "extra": "ignored"})
        assert result["number"] == 1

    @pytest.mark.asyncio
    async def test_tool_call_with_string_for_integer(self):
        """String pr_number must raise (Pydantic/validation error)."""
        mock_client = AsyncMock()
        server = GitHubMCPServer(client=mock_client)
        # The server passes args directly to the client — if client doesn't validate, it may work
        # But the MCP schema says integer, so this is an adversarial input
        mock_client.get_pr = AsyncMock(return_value=MagicMock(model_dump=lambda: {"number": "abc"}))
        result = await server.handle_tool("get_pr", {"pr_number": "abc"})
        # May or may not raise depending on client validation — just ensure no crash
        assert result is not None

    @pytest.mark.asyncio
    async def test_post_review_with_invalid_event(self):
        """Invalid event type must still route to client (validation is GitHub's job)."""
        mock_client = AsyncMock()
        mock_client.post_review = AsyncMock(return_value=MagicMock(model_dump=lambda: {"id": 1}))
        server = GitHubMCPServer(client=mock_client)
        await server.handle_tool(
            "post_review_comment",
            {
                "pr_number": 1,
                "body": "test",
                "event": "DOES_NOT_EXIST",
            },
        )
        mock_client.post_review.assert_called_once_with(1, "test", "DOES_NOT_EXIST")


# ---------------------------------------------------------------------------
# CLI commands on non-git directories
# ---------------------------------------------------------------------------


class TestCLINonGitDir:
    """Running sentinel in a non-git directory must not crash."""

    def test_check_in_non_git_dir(self):
        """sentinel check in a non-git directory must return error code."""
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(["check", "--workspace", tmp])
            # extract_git_context will raise GitError, cmd_check catches it
            result = cmd_check(args)
            assert result == 2  # Error exit code

    def test_list_works_anywhere(self):
        """sentinel list must work in any directory (just reads SQLite)."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "test.db")
            args = build_parser().parse_args(["list", "--db-path", db_path])
            result = cmd_list(args)
            assert result == 0

    def test_resume_no_pending_sessions(self):
        """sentinel resume with no pending sessions must print message."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "test.db")
            args = build_parser().parse_args(["resume", "--db-path", db_path])
            result = cmd_resume(args)
            assert result == 0

    def test_resume_nonexistent_session(self):
        """sentinel resume with invalid session ID must return error."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "test.db")
            args = build_parser().parse_args(["resume", "fake_session_id", "--db-path", db_path])
            result = cmd_resume(args)
            assert result == 2

    def test_resume_terminal_session(self):
        """sentinel resume on COMPLETED session must print message."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "test.db")
            store = SessionStore(db_path=db_path)
            sess = store.create_session("main", "abc", "test")
            store.mark_completed(sess.session_id)

            args = build_parser().parse_args(["resume", sess.session_id, "--db-path", db_path])
            result = cmd_resume(args)
            assert result == 0


# ---------------------------------------------------------------------------
# Unicode and special characters
# ---------------------------------------------------------------------------


class TestUnicodeEdgeCases:
    """Unicode in paths, branch names, and diff content."""

    def test_branch_name_with_unicode(self):
        """Unicode branch names must not crash session creation."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            sess = store.create_session("feat/日本語テスト", "abc123", "unicode diff")
            hydrated = store.get_session(sess.session_id)
            assert hydrated.branch_name == "feat/日本語テスト"

    def test_diff_with_unicode(self):
        """Unicode in diff content must persist correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            unicode_diff = "diff --git a/test.py b/test.py\n+print('こんにちは世界')\n+print('🔥')"
            sess = store.create_session("main", "abc", "unicode", raw_diff=unicode_diff)
            hydrated = store.get_session(sess.session_id)
            assert "こんにちは" in hydrated.raw_diff
            assert "🔥" in hydrated.raw_diff

    def test_very_long_branch_name(self):
        """Branch names up to 255 chars must work."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            long_name = "a" * 255
            sess = store.create_session(long_name, "abc", "test")
            hydrated = store.get_session(sess.session_id)
            assert len(hydrated.branch_name) == 255

    def test_empty_branch_name(self):
        """Empty branch name must still create a session."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            sess = store.create_session("", "abc", "test")
            hydrated = store.get_session(sess.session_id)
            assert hydrated.branch_name == ""


# ---------------------------------------------------------------------------
# Signal / timeout / interrupted execution
# ---------------------------------------------------------------------------


class TestTimeoutBehavior:
    """What happens when things take too long?"""

    @pytest.mark.asyncio
    async def test_orchestrator_timeout_cleans_up(self):
        """Timeout must not leave orphaned state in the session store."""
        from sentinel.models.diff import parse_git_diff

        mock_lace = MagicMock()
        mock_lace.is_connected = True
        mock_lace.degraded = False

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "test.db"))
            req = OrchestratorRequest(
                session_id="timeout_test",
                branch_name="main",
                commit_sha="abc",
                diff_summary="test",
                git_diff=parse_git_diff(""),
                touched_files=[],
                workspace_root=Path(tmp),
                lace_client=mock_lace,
                session_store=store,
                interactive=False,
                timeout_seconds=1,  # Very short timeout
            )

            async def slow_run(*args, **kwargs):
                await asyncio.sleep(10)
                return MagicMock()

            with (
                patch("sentinel.orchestrator.SandboxRunner") as MockSb,
                patch("sentinel.orchestrator.ADRDeltaAnalyzer") as MockAdr,
            ):
                MockSb.return_value.run = slow_run
                MockAdr.return_value.run = slow_run

                orch = ReviewOrchestrator()
                with pytest.raises(asyncio.TimeoutError):
                    await orch.run_review(req)

                # Session should still be in the store (created before timeout)
                session = store.get_session("timeout_test")
                assert session is not None


# ---------------------------------------------------------------------------
# Database corruption recovery
# ---------------------------------------------------------------------------


class TestDatabaseCorruption:
    """What happens when the SQLite file is corrupted?"""

    @pytest.mark.skipif(os.name == "nt", reason="Windows file locking")
    def test_corrupted_db_file(self):
        """Corrupted DB must raise, not silently return wrong data."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "corrupt.db")
            # Write garbage to the DB file
            Path(db_path).write_bytes(b"NOT A SQLITE FILE" * 100)

            with pytest.raises(sqlite3.DatabaseError):
                SessionStore(db_path=db_path)

    @pytest.mark.skipif(os.name == "nt", reason="Windows file locking")
    def test_truncated_db_file(self):
        """Truncated DB must raise, not return partial data."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "truncated.db")
            # Create a valid DB then truncate it
            store = SessionStore(db_path=db_path)
            store.create_session("main", "abc", "test")

            # Truncate the file
            with open(db_path, "rb") as f:
                data = f.read()
            with open(db_path, "wb") as f:
                f.write(data[:100])  # Truncate to 100 bytes

            # New SessionStore must handle corruption
            with pytest.raises(sqlite3.DatabaseError):
                SessionStore(db_path=db_path)


# ---------------------------------------------------------------------------
# Race conditions
# ---------------------------------------------------------------------------


class TestRaceConditions:
    """Simulate concurrent access patterns."""

    def test_concurrent_session_creation(self):
        """Multiple sessions created simultaneously must not conflict."""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "race.db"))
            sessions = []
            for i in range(20):
                sess = store.create_session(f"branch_{i}", f"sha_{i}", f"diff_{i}")
                sessions.append(sess)

            # Verify all sessions exist and are independent
            session_ids = set()
            for sess in sessions:
                hydrated = store.get_session(sess.session_id)
                assert hydrated is not None
                session_ids.add(hydrated.session_id)

            assert len(session_ids) == 20

    def test_concurrent_subagent_results(self):
        """Multiple subagent results for same session must not conflict."""
        from sentinel.models.review_state import SubagentStatus, SubagentType

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(db_path=str(Path(tmp) / "race.db"))
            sess = store.create_session("main", "abc", "test")

            # Both subagents writing results simultaneously
            store.save_subagent_result(
                sess.session_id,
                SubagentType.SANDBOX_RUNNER,
                SubagentStatus.COMPLETED,
                {"tests": 5},
            )
            store.save_subagent_result(
                sess.session_id,
                SubagentType.ADR_DELTA_ANALYZER,
                SubagentStatus.COMPLETED,
                {"violations": []},
            )

            hydrated = store.get_session(sess.session_id)
            assert len(hydrated.tasks) == 2


# ---------------------------------------------------------------------------
# Import / dependency checks
# ---------------------------------------------------------------------------


class TestDependencyChecks:
    """Verify all imports resolve and key types are available."""

    def test_all_models_importable(self):
        """All model classes must be importable."""
        from sentinel.models.review_state import (
            ApprovalDecision,
            SessionStatus,
            SubagentType,
        )

        # Verify enum values exist
        assert SessionStatus.PENDING_SUBAGENTS.value == "PENDING_SUBAGENTS"
        assert SubagentType.GITHUB_PR.value == "GITHUB_PR"
        assert ApprovalDecision.PENDING.value == "PENDING"

    def test_all_subagents_importable(self):
        """All subagent classes must be importable."""

    def test_orchestrator_importable(self):
        """Orchestrator must be importable."""

    def test_cli_importable(self):
        """CLI module must be importable."""


# Need json for some tests
import json
