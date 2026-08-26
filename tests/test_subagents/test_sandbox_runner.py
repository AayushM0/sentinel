"""Tests for SandboxRunner — all Daytona SDK calls are mocked (no API key needed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.subagents.sandbox_runner import SandboxRequest, SandboxRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exec_response(exit_code: int, result: str) -> MagicMock:
    r = MagicMock()
    r.exit_code = exit_code
    r.result = result
    return r


def _make_mock_client(sandbox: MagicMock) -> AsyncMock:
    """Return a mock AsyncDaytona client that yields sandbox on create()."""
    client = AsyncMock()
    client.create = AsyncMock(return_value=sandbox)
    client.delete = AsyncMock()
    return client


def _make_mock_sandbox(exec_responses: list) -> MagicMock:
    """Return a mock AsyncSandbox whose process.exec returns responses in order."""
    sandbox = MagicMock()
    sandbox.process = MagicMock()
    responses = list(exec_responses)

    async def fake_exec(cmd: str, *args, **kwargs):
        if cmd.startswith("mkdir"):
            return _make_exec_response(0, "")
        if responses:
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, type) and issubclass(item, Exception):
                raise item()
            return item
        return _make_exec_response(0, "")

    sandbox.process.exec = AsyncMock(side_effect=fake_exec)
    sandbox.fs = MagicMock()
    sandbox.fs.upload_file = AsyncMock()
    return sandbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_request(tmp_path) -> SandboxRequest:
    # Write minimal project files so upload_files doesn't skip them
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n")
    f = tmp_path / "src" / "foo.py"
    f.parent.mkdir(parents=True)
    f.write_text("x = 1\n")
    return SandboxRequest(
        session_id="sess_test",
        project_root=str(tmp_path),
        changed_files=[str(f)],
        test_command="uv run pytest -v",
        timeout_seconds=30,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_runner_success(basic_request):
    """Clean test run → sandbox_status='completed', exit_code=0."""
    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),  # uv sync
            _make_exec_response(0, "5 passed in 0.42s"),  # pytest
        ]
    )
    client = _make_mock_client(sandbox)

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(basic_request)

    assert result.sandbox_status == "completed"
    assert result.exit_code == 0
    assert result.tests_passed == 5
    assert result.tests_failed == 0
    client.delete.assert_called_once_with(sandbox)


@pytest.mark.asyncio
async def test_sandbox_runner_failure(basic_request):
    """Failing tests → sandbox_status='failed', tests_failed populated."""
    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),  # uv sync
            _make_exec_response(1, "3 passed, 2 failed in 0.88s"),  # pytest
        ]
    )
    client = _make_mock_client(sandbox)

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(basic_request)

    assert result.sandbox_status == "failed"
    assert result.exit_code == 1
    assert result.tests_passed == 3
    assert result.tests_failed == 2


@pytest.mark.asyncio
async def test_sandbox_runner_compile_error(basic_request):
    """Compile error before tests → exit_code=1, counts=(0,0), status='failed'."""
    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),  # uv sync
            _make_exec_response(1, "ERROR collecting tests/\nImportError: bad import"),
        ]
    )
    client = _make_mock_client(sandbox)

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(basic_request)

    assert result.sandbox_status == "failed"
    assert result.tests_passed == 0
    assert result.tests_failed == 0


@pytest.mark.asyncio
async def test_sandbox_runner_timeout(basic_request):
    """process.exec timeout → sandbox_status='timeout', exit_code=-1."""
    from daytona import DaytonaProcessExecutionTimeoutError

    sandbox = _make_mock_sandbox([])
    client = _make_mock_client(sandbox)
    # uv sync raises timeout
    sandbox.process.exec = AsyncMock(side_effect=DaytonaProcessExecutionTimeoutError("timed out"))

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(basic_request)

    assert result.sandbox_status == "timeout"
    assert result.exit_code == -1


@pytest.mark.asyncio
async def test_sandbox_runner_always_deletes(basic_request):
    """Even when test command raises mid-run, client.delete() is still called."""
    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),  # uv sync
            RuntimeError("unexpected crash"),  # pytest raises
        ]
    )
    client = _make_mock_client(sandbox)

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(basic_request)

    client.delete.assert_called_once_with(sandbox)
    assert result.sandbox_status == "error"


@pytest.mark.asyncio
async def test_sandbox_runner_create_fails(basic_request):
    """client.create() raises → delete is NOT called (sandbox was never created)."""
    client = AsyncMock()
    client.create = AsyncMock(side_effect=RuntimeError("auth error"))
    client.delete = AsyncMock()

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(basic_request)

    client.delete.assert_not_called()
    assert result.sandbox_status == "error"


@pytest.mark.asyncio
async def test_sandbox_runner_saves_to_store(basic_request, tmp_path):
    """When session_store provided, save_subagent_result() called with correct payload."""
    from sentinel.session_store import SessionStore

    db_path = tmp_path / "test.db"
    store = SessionStore(db_path=str(db_path))
    store.create_session(branch_name="feat/test", commit_sha="abc123")

    # Override session_id to match the created session
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    session_id = conn.execute("SELECT session_id FROM review_sessions").fetchone()[0]
    conn.close()
    basic_request.session_id = session_id

    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),
            _make_exec_response(0, "5 passed in 0.4s"),
        ]
    )
    client = _make_mock_client(sandbox)

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner(session_store=store).run(basic_request)

    assert result.sandbox_status == "completed"

    # Verify it was persisted
    session = store.get_session(session_id)
    assert session is not None
    assert len(session.tasks) == 1
    assert session.tasks[0].result_payload["sandbox_status"] == "completed"
    assert session.tasks[0].result_payload["tests_passed"] == 5


@pytest.mark.asyncio
async def test_sandbox_runner_linter_errors_non_fatal(basic_request):
    """Linter failure surfaces errors on card but tests still run."""
    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),  # uv sync
            _make_exec_response(1, "src/foo.py:1:1: E501 line too long"),  # linter fails
            _make_exec_response(0, "5 passed in 0.4s"),  # tests pass
        ]
    )
    client = _make_mock_client(sandbox)

    request_with_linter = SandboxRequest(
        session_id=basic_request.session_id,
        project_root=basic_request.project_root,
        changed_files=basic_request.changed_files,
        test_command=basic_request.test_command,
        linter_command="uv run ruff check src",
        timeout_seconds=30,
    )

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(request_with_linter)

    assert result.sandbox_status == "completed"  # linter non-fatal
    assert len(result.linter_errors) == 1
    assert "E501" in result.linter_errors[0]


@pytest.mark.asyncio
async def test_sandbox_runner_deep_nested_and_multiple_files(tmp_path):
    """Deep nested files (e.g. src/a/b/c/foo.py) are uploaded with POSIX paths and directories."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='nested'\n")
    nested_file = tmp_path / "src" / "a" / "b" / "c" / "deep.py"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("def deep(): return 42\n")

    req = SandboxRequest(
        session_id="sess_nested",
        project_root=str(tmp_path),
        changed_files=[str(nested_file)],
        test_command="pytest",
        timeout_seconds=30,
    )

    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),
            _make_exec_response(0, "1 passed in 0.1s"),
        ]
    )
    client = _make_mock_client(sandbox)

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(req)

    assert result.sandbox_status == "completed"
    assert result.tests_passed == 1
    # Verify upload was called with forward slashes in destination
    uploaded_dests = [call.args[1] for call in sandbox.fs.upload_file.call_args_list]
    assert any("/workspace/src/a/b/c/deep.py" in d for d in uploaded_dests)


@pytest.mark.asyncio
async def test_sandbox_runner_end_to_end_approval_gate_integration(tmp_path, monkeypatch):
    """End-to-end test verifying SandboxRunner output seamlessly feeds into ApprovalGate."""
    from sentinel.approval_gate import ApprovalDecision, ApprovalGate
    from sentinel.session_store import SessionStore

    db_path = tmp_path / "e2e.db"
    store = SessionStore(db_path=str(db_path))
    session = store.create_session(branch_name="feat/e2e", commit_sha="e2e123")

    req = SandboxRequest(
        session_id=session.session_id,
        project_root=str(tmp_path),
        changed_files=[],
        test_command="pytest",
        timeout_seconds=30,
    )

    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),
            _make_exec_response(0, "10 passed in 1.0s"),
        ]
    )
    client = _make_mock_client(sandbox)

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        runner_result = await SandboxRunner(session_store=store).run(req)

    assert runner_result.sandbox_status == "completed"
    assert runner_result.exit_code == 0

    # Feed result into ApprovalGate
    gate = ApprovalGate(session_store=store)
    card = gate.format_approval_card(
        session_id=session.session_id,
        branch_name="feat/e2e",
        commit_sha="e2e123",
        test_result=runner_result.to_dict(),
        delta_report={},
    )
    assert "**Status:** `SUCCESS`" in card
    assert "10/10" in card

    # User approves
    monkeypatch.setattr("builtins.input", lambda _: "approve")
    decision = gate.request_approval(
        session=session,
        test_result=runner_result.to_dict(),
        delta_report={},
        interactive=True,
    )
    assert decision == ApprovalDecision.APPROVED

    # Confirm session status is now APPROVED
    updated_session = store.get_session(session.session_id)
    assert updated_session is not None
    assert updated_session.status.value == "APPROVED"


@pytest.mark.asyncio
async def test_sandbox_runner_dependency_failure(basic_request):
    """Nonzero exit code in pip install / uv sync returns error status and halts test run."""
    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(1, "error: package not found"),  # uv sync fails
        ]
    )
    client = _make_mock_client(sandbox)

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(basic_request)

    assert result.sandbox_status == "error"
    assert result.exit_code == 1
    assert "Dependency installation failed" in result.logs
    client.delete.assert_called_once_with(sandbox)


@pytest.mark.asyncio
async def test_sandbox_runner_linter_timeout_or_crash_non_fatal(basic_request):
    """Linter raising exception or timing out is non-fatal; tests still run."""
    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),  # uv sync
            RuntimeError("Linter daemon timed out"),  # linter command crashes
            _make_exec_response(0, "7 passed in 0.5s"),  # pytest runs successfully
        ]
    )
    client = _make_mock_client(sandbox)

    req = SandboxRequest(
        session_id=basic_request.session_id,
        project_root=basic_request.project_root,
        changed_files=basic_request.changed_files,
        test_command=basic_request.test_command,
        linter_command="ruff check src",
        timeout_seconds=30,
    )

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(req)

    assert result.sandbox_status == "completed"
    assert result.exit_code == 0
    assert result.tests_passed == 7
    assert len(result.linter_errors) == 1
    assert "Linter execution error" in result.linter_errors[0]


@pytest.mark.asyncio
async def test_sandbox_runner_delete_retry_and_leak_logging(basic_request):
    """If client.delete() fails, it retries and surfaces leak info in logs."""
    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),
            _make_exec_response(0, "2 passed in 0.1s"),
        ]
    )
    sandbox.id = "sbx_leak_999"
    client = AsyncMock()
    client.create = AsyncMock(return_value=sandbox)
    client.delete = AsyncMock(side_effect=RuntimeError("API Gateway 502"))

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(basic_request)

    assert client.delete.call_count == 3
    assert "[CLEANUP ERROR]" in result.logs
    assert "LEAKED_SANDBOX_sbx_leak_999" in result.logs


@pytest.mark.asyncio
async def test_sandbox_runner_escaped_symlink_upload_rejected(tmp_path):
    """Paths escaping project_root via symlinks or traversal are omitted from upload."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='p'\n")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret_data")

    runner = SandboxRunner()
    paths = runner._collect_upload_paths(root, [str(outside_file)])
    assert outside_file not in paths


@pytest.mark.asyncio
async def test_sandbox_runner_uploads_all_workspace_files(tmp_path):
    """Unchanged files in project root (README, fixtures) are collected and uploaded."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='all_ws'\n")
    (tmp_path / "README.md").write_text("# Readme\n")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "conftest.py").write_text("# fixtures\n")
    (test_dir / "test_a.py").write_text("def test_ok(): pass\n")

    changed = tmp_path / "src" / "new.py"
    changed.parent.mkdir()
    changed.write_text("y = 2\n")

    req = SandboxRequest(
        session_id="sess_all_ws",
        project_root=str(tmp_path),
        changed_files=[str(changed)],
        test_command="pytest",
    )

    sandbox = _make_mock_sandbox(
        [
            _make_exec_response(0, ""),
            _make_exec_response(0, "1 passed in 0.1s"),
        ]
    )
    client = _make_mock_client(sandbox)

    with patch("sentinel.subagents.sandbox_runner.AsyncDaytona") as MockDaytona:
        MockDaytona.return_value.__aenter__ = AsyncMock(return_value=client)
        MockDaytona.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await SandboxRunner().run(req)

    assert result.sandbox_status == "completed"
    uploaded_dests = [call.args[1] for call in sandbox.fs.upload_file.call_args_list]
    assert "/workspace/pyproject.toml" in uploaded_dests
    assert "/workspace/README.md" in uploaded_dests
    assert "/workspace/tests/conftest.py" in uploaded_dests
    assert "/workspace/tests/test_a.py" in uploaded_dests
    assert "/workspace/src/new.py" in uploaded_dests


# ---------------------------------------------------------------------------
# _parse_pytest_output unit tests (pure, no mocks needed)
# ---------------------------------------------------------------------------


def test_parse_pytest_output_standard():
    runner = SandboxRunner()
    p, f = runner._parse_pytest_output("3 passed, 1 failed in 0.42s")
    assert p == 3 and f == 1


def test_parse_pytest_output_passed_only():
    runner = SandboxRunner()
    p, f = runner._parse_pytest_output("32 passed in 1.38s")
    assert p == 32 and f == 0


def test_parse_pytest_output_empty():
    runner = SandboxRunner()
    p, f = runner._parse_pytest_output("")
    assert p == 0 and f == 0


def test_parse_pytest_output_no_tests():
    runner = SandboxRunner()
    p, f = runner._parse_pytest_output("no tests ran")
    assert p == 0 and f == 0


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "src/sentinel/subagents/sandbox_runner.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)
    print("test_sandbox_runner.py standalone checks passed.")
