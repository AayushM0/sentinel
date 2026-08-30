"""Sentinel Subagent A: Daytona Sandbox Runner.

Runs the project test suite inside an isolated Daytona sandbox and returns
a typed SandboxResult. The sandbox is always deleted on exit.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from daytona import (
        AsyncDaytona,
        CreateSandboxFromImageParams,
        DaytonaProcessExecutionTimeoutError,
        Image,
    )
except ImportError:  # pragma: no cover
    AsyncDaytona = None  # type: ignore[misc,assignment]
    CreateSandboxFromImageParams = None  # type: ignore[misc,assignment]
    DaytonaProcessExecutionTimeoutError = type(
        "DaytonaProcessExecutionTimeoutError", (Exception,), {}
    )  # type: ignore[misc,assignment]
    Image = None  # type: ignore[misc,assignment]

from sentinel.models.github import SecurityAuditResult, SecurityVulnerability
from sentinel.models.review_state import SubagentStatus, SubagentType
from sentinel.session_store import SessionStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data contracts (match TRD §3.3 exactly)
# ---------------------------------------------------------------------------


@dataclass
class SandboxRequest:
    """Input to SandboxRunner.run()."""

    session_id: str
    project_root: str  # absolute local path
    changed_files: list[str]  # absolute local paths
    test_command: str  # e.g. "uv run pytest -v"
    linter_command: str | None = None  # e.g. "uv run ruff check src" — optional, non-fatal
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")


@dataclass
class SandboxResult:
    """Output of SandboxRunner.run(). Matches TRD §3.3 Sandbox Execution Response."""

    sandbox_status: str  # "completed" | "failed" | "timeout" | "error"
    exit_code: int  # 0 = success, nonzero = failure, -1 = timeout/error
    tests_passed: int
    tests_failed: int
    linter_errors: list[str]
    duration_ms: int
    logs: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_WORKSPACE = "/workspace"
_DEPS_TIMEOUT = 120  # seconds for uv sync — separate from test timeout
_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    "dist",
    "build",
    ".eggs",
    ".lace",
    ".idea",
    ".vscode",
}


class SandboxRunner:
    """Daytona Sandbox Runner subagent.

    Usage::

        runner = SandboxRunner(session_store=store)
        result = await runner.run(request)
    """

    def __init__(self, session_store: SessionStore | None = None) -> None:
        self.session_store = session_store

    async def run(self, request: SandboxRequest) -> SandboxResult:
        """Create a Daytona sandbox, run tests, delete sandbox, return result."""
        start = time.monotonic()
        sandbox = None
        cleanup_error: str | None = None

        try:
            async with AsyncDaytona() as client:
                params = CreateSandboxFromImageParams(
                    image=Image.debian_slim("3.13"),
                    language="python",
                )
                sandbox = await client.create(params)

                try:
                    result = await self._run_in_sandbox(client, sandbox, request, start)
                finally:
                    if sandbox is not None:
                        cleanup_error = await self._delete_sandbox_with_retry(client, sandbox)

        except DaytonaProcessExecutionTimeoutError:
            result = SandboxResult(
                sandbox_status="timeout",
                exit_code=-1,
                tests_passed=0,
                tests_failed=0,
                linter_errors=[],
                duration_ms=int((time.monotonic() - start) * 1000),
                logs="Daytona process execution timed out.",
            )
        except Exception as exc:  # noqa: BLE001
            result = SandboxResult(
                sandbox_status="error",
                exit_code=-1,
                tests_passed=0,
                tests_failed=0,
                linter_errors=[],
                duration_ms=int((time.monotonic() - start) * 1000),
                logs=str(exc),
            )

        if cleanup_error:
            result.logs = f"{result.logs}\n\n[CLEANUP ERROR] {cleanup_error}".strip()

        self._persist(request, result)
        return result

    async def _delete_sandbox_with_retry(
        self,
        client: AsyncDaytona,
        sandbox,
        max_retries: int = 3,
    ) -> str | None:
        """Attempt sandbox deletion with bounded retries; returns error message if leaked."""
        sandbox_id = getattr(sandbox, "id", None) or repr(sandbox)
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                await client.delete(sandbox)
                return None
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Sandbox deletion attempt %d/%d failed for %s: %s",
                    attempt,
                    max_retries,
                    sandbox_id,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (2 ** (attempt - 1)))

        leak_msg = f"LEAKED_SANDBOX_{sandbox_id}: Failed to delete after {max_retries} attempts ({last_exc})"
        logger.critical(leak_msg)
        return leak_msg

    async def _run_in_sandbox(
        self,
        client: AsyncDaytona,
        sandbox,
        request: SandboxRequest,
        start: float,
    ) -> SandboxResult:
        """Upload files, install deps, run linter + tests, return result."""
        # 1. Upload workspace files + changed files
        await self._upload_files(sandbox, request)

        # 2. Install dependencies (fixed timeout, separate from test timeout)
        dep_res = await sandbox.process.exec(
            f"cd {_WORKSPACE} && pip install uv -q && uv sync --dev",
            timeout=_DEPS_TIMEOUT,
        )
        if dep_res.exit_code != 0:
            return SandboxResult(
                sandbox_status="error",
                exit_code=dep_res.exit_code,
                tests_passed=0,
                tests_failed=0,
                linter_errors=[],
                duration_ms=int((time.monotonic() - start) * 1000),
                logs=f"Dependency installation failed (exit code {dep_res.exit_code}):\n{dep_res.result}",
            )

        # 3. Run linter (optional, non-fatal — errors or timeouts isolated)
        linter_errors: list[str] = []
        if request.linter_command:
            try:
                lr = await sandbox.process.exec(
                    f"cd {_WORKSPACE} && {request.linter_command}",
                    timeout=request.timeout_seconds,
                )
                if lr.exit_code != 0:
                    linter_errors = [line for line in lr.result.splitlines() if line.strip()]
            except Exception as exc:  # noqa: BLE001
                linter_errors = [f"Linter execution error (non-fatal): {exc}"]

        # 4. Run test command (authoritative for sandbox_status)
        tr = await sandbox.process.exec(
            f"cd {_WORKSPACE} && {request.test_command}",
            timeout=request.timeout_seconds,
        )
        passed, failed = self._parse_pytest_output(tr.result)
        status = "completed" if tr.exit_code == 0 else "failed"

        # 5. Security audit (non-fatal, informational)
        security_findings: list[str] = []
        try:
            # Check for Python project and run pip-audit
            check_py = await sandbox.process.exec(
                f"cd {_WORKSPACE} && test -f pyproject.toml -o -f requirements.txt",
                timeout=5,
            )
            if check_py.exit_code == 0:
                # ponytail: pip-audit exit code 1 = vulnerabilities found (expected),
                #           exit code 2+ = tool/install failure. Parse output regardless.
                audit_res = await sandbox.process.exec(
                    f"cd {_WORKSPACE} && pip install pip-audit -q && pip-audit --format json",
                    timeout=60,
                )
                if audit_res.exit_code <= 1 and audit_res.result.strip():
                    parsed = self._parse_pip_audit(audit_res.result)
                    if parsed.vulnerabilities:
                        security_findings.append(f"pip-audit: {parsed.summary}")
                elif audit_res.exit_code > 1:
                    security_findings.append(
                        f"pip-audit: tool error (exit code {audit_res.exit_code})"
                    )

            # Check for JS project and run npm audit
            check_js = await sandbox.process.exec(
                f"cd {_WORKSPACE} && test -f package.json",
                timeout=5,
            )
            if check_js.exit_code == 0:
                audit_res = await sandbox.process.exec(
                    f"cd {_WORKSPACE} && npm audit --json",
                    timeout=60,
                )
                # npm audit exit code 1 = vulnerabilities found, 2+ = tool error
                if audit_res.exit_code <= 1 and audit_res.result.strip():
                    parsed = self._parse_npm_audit(audit_res.result)
                    if parsed.vulnerabilities:
                        security_findings.append(f"npm-audit: {parsed.summary}")
                elif audit_res.exit_code > 1:
                    security_findings.append(
                        f"npm-audit: tool error (exit code {audit_res.exit_code})"
                    )
        except Exception as exc:  # noqa: BLE001
            security_findings.append(f"Security audit error (non-fatal): {exc}")

        logs = tr.result
        if security_findings:
            logs += "\n\n--- Security Audit ---\n" + "\n".join(security_findings)

        return SandboxResult(
            sandbox_status=status,
            exit_code=tr.exit_code,
            tests_passed=passed,
            tests_failed=failed,
            linter_errors=linter_errors,
            duration_ms=int((time.monotonic() - start) * 1000),
            logs=logs,
        )

    def _collect_upload_paths(self, root: Path, changed_files: list[str]) -> list[Path]:
        """Collect all relevant workspace files and changed files, canonicalized."""
        root_resolved = root.resolve()
        candidates: set[Path] = set()

        if root_resolved.is_dir():
            for item in root_resolved.rglob("*"):
                if not item.is_file():
                    continue
                # Skip ignored directories
                rel_parts = set(item.relative_to(root_resolved).parts)
                if rel_parts.intersection(_IGNORED_DIRS):
                    continue
                try:
                    if not item.resolve().is_relative_to(root_resolved):
                        continue
                except OSError:
                    continue
                candidates.add(item)

        for cf in changed_files:
            p = Path(cf)
            try:
                p_resolved = p.resolve()
            except OSError:
                continue
            if (
                p_resolved.exists()
                and p_resolved.is_file()
                and p_resolved.is_relative_to(root_resolved)
            ):
                candidates.add(p_resolved)

        return sorted(candidates)

    async def _upload_files(self, sandbox, request: SandboxRequest) -> None:
        """Upload complete project workspace + changed files into /workspace."""
        root = Path(request.project_root)
        root_resolved = root.resolve()

        paths = self._collect_upload_paths(root, request.changed_files)

        for p in paths:
            try:
                p_resolved = p.resolve()
            except OSError:
                continue

            if not p_resolved.exists() or not p_resolved.is_file():
                continue

            # Strict containment check against resolved root
            try:
                rel = p_resolved.relative_to(root_resolved)
            except ValueError:
                # Outside project root or escaping symlink — skip safely
                continue

            # Double-check relative path components do not contain parent traversal
            if ".." in rel.parts:
                continue

            dest = f"{_WORKSPACE}/{rel.as_posix()}"
            parent = Path(dest).parent.as_posix()
            if parent != _WORKSPACE:
                await sandbox.process.exec(f"mkdir -p {shlex.quote(parent)}")
            await sandbox.fs.upload_file(p_resolved.read_bytes(), dest)

    def _parse_pytest_output(self, logs: str) -> tuple[int, int]:
        """Extract (passed, failed) counts from pytest summary line.

        Reverse-scans for the first line containing 'passed' or 'failed'
        as a token preceded by a digit. Falls back to (0, 0) safely —
        exit_code is always the authoritative success signal.

        # ponytail: naive reverse token scan; breaks with custom pytest reporters
        #           or --no-header. Upgrade: run pytest with --tb=no -q for stable output.
        """
        passed = failed = 0
        for line in reversed(logs.splitlines()):
            tokens = line.lower().split()
            for i, tok in enumerate(tokens):
                clean = tok.rstrip(",.;")  # handle "passed," "failed,"
                if clean == "passed" and i > 0 and tokens[i - 1].isdigit():
                    passed = int(tokens[i - 1])
                if clean == "failed" and i > 0 and tokens[i - 1].isdigit():
                    failed = int(tokens[i - 1])
            if passed or failed:
                break
        return passed, failed

    def _parse_pip_audit(self, output: str) -> SecurityAuditResult:
        """Parse pip-audit JSON output into SecurityAuditResult."""
        import json

        vulnerabilities = []
        try:
            data = json.loads(output)
            for dep in data.get("dependencies", []):
                for vuln in dep.get("vulns", []):
                    vulnerabilities.append(
                        SecurityVulnerability(
                            package=dep["name"],
                            installed_version=dep["version"],
                            fixed_version=(vuln.get("fix_versions") or [None])[0],
                            severity="high" if vuln.get("fix_versions") else "medium",
                            advisory=vuln.get("id", ""),
                            url=(vuln.get("aliases") or [None])[0],
                        )
                    )
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        summary = (
            f"Found {len(vulnerabilities)} vulnerabilities"
            if vulnerabilities
            else "No vulnerabilities found"
        )
        return SecurityAuditResult(
            tool="pip-audit",
            vulnerabilities=vulnerabilities,
            summary=summary,
            exit_code=1 if vulnerabilities else 0,
        )

    def _parse_npm_audit(self, output: str) -> SecurityAuditResult:
        """Parse npm audit JSON output into SecurityAuditResult."""
        import json

        vulnerabilities = []
        try:
            data = json.loads(output)
            for pkg, info in data.get("vulnerabilities", {}).items():
                via = info.get("via", [{}])
                url = via[0].get("url") if via and isinstance(via[0], dict) else None
                vulnerabilities.append(
                    SecurityVulnerability(
                        package=pkg,
                        installed_version=info.get("version", "unknown"),
                        fixed_version=info.get("fixAvailable", {}).get("version")
                        if isinstance(info.get("fixAvailable"), dict)
                        else None,
                        severity=info.get("severity", "unknown"),
                        advisory=via[0].get("title", "")
                        if via and isinstance(via[0], dict)
                        else "",
                        url=url,
                    )
                )
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        summary = (
            f"Found {len(vulnerabilities)} vulnerabilities"
            if vulnerabilities
            else "No vulnerabilities found"
        )
        return SecurityAuditResult(
            tool="npm-audit",
            vulnerabilities=vulnerabilities,
            summary=summary,
            exit_code=1 if vulnerabilities else 0,
        )

    def _persist(self, request: SandboxRequest, result: SandboxResult) -> None:
        """Save result to SessionStore if one was provided."""
        if self.session_store is None:
            return
        status = SubagentStatus.COMPLETED if result.exit_code == 0 else SubagentStatus.FAILED
        self.session_store.save_subagent_result(
            session_id=request.session_id,
            subagent_type=SubagentType.SANDBOX_RUNNER,
            status=status,
            result_payload=result.to_dict(),
            task_id=f"{request.session_id}_sandbox",
        )


# ---------------------------------------------------------------------------
# Rule 2903681 standalone self-check (no Daytona, no pytest, no network)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    runner = SandboxRunner()

    # 1. Output parser checks
    p, f = runner._parse_pytest_output("5 passed, 2 failed in 1.23s")
    assert p == 5 and f == 2, f"Expected (5, 2), got ({p}, {f})"

    p2, f2 = runner._parse_pytest_output("")
    assert p2 == 0 and f2 == 0, "Empty output should yield (0, 0)"

    p3, f3 = runner._parse_pytest_output("ERROR collecting tests/test_foo.py\nImportError: ...")
    assert p3 == 0 and f3 == 0, "Compile error output should yield (0, 0)"

    p4, f4 = runner._parse_pytest_output("32 passed in 1.38s")
    assert p4 == 32 and f4 == 0, f"Expected (32, 0), got ({p4}, {f4})"

    # 2. SandboxResult contract
    sr = SandboxResult(
        sandbox_status="completed",
        exit_code=0,
        tests_passed=5,
        tests_failed=0,
        linter_errors=[],
        duration_ms=1200,
        logs="5 passed in 1.2s",
    )
    d = sr.to_dict()
    assert d["sandbox_status"] == "completed"
    assert d["exit_code"] == 0
    assert sr.sandbox_status in ("completed", "success")

    # 3. SandboxRequest rejects non-positive timeout
    try:
        SandboxRequest(
            session_id="s1",
            project_root="/p",
            changed_files=[],
            test_command="pytest",
            timeout_seconds=0,
        )
        raise AssertionError("Should have raised ValueError for timeout_seconds=0")
    except ValueError:
        pass

    # 4. Async lifecycle self-checks with in-memory fakes
    class FakeExecResult:
        def __init__(self, exit_code: int, result: str):
            self.exit_code = exit_code
            self.result = result

    class FakeFS:
        def __init__(self):
            self.uploaded: list[tuple[bytes, str]] = []

        async def upload_file(self, data: bytes, dest: str) -> None:
            self.uploaded.append((data, dest))

    class FakeProcess:
        def __init__(self, exec_handler):
            self.exec_handler = exec_handler

        async def exec(self, cmd: str, timeout: int | None = None):
            return await self.exec_handler(cmd, timeout)

    class FakeSandbox:
        def __init__(self, exec_handler):
            self.id = "sbx_test_123"
            self.process = FakeProcess(exec_handler)
            self.fs = FakeFS()

    class FakeClient:
        def __init__(self, sandbox, delete_exc=None):
            self.sandbox = sandbox
            self.delete_calls: list = []
            self.delete_exc = delete_exc

        async def create(self, params):
            return self.sandbox

        async def delete(self, sandbox):
            self.delete_calls.append(sandbox)
            if self.delete_exc:
                raise self.delete_exc

    async def run_lifecycle_checks():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmproot = Path(tmpdir)
            (tmproot / "pyproject.toml").write_text("[project]\nname='test'\n")
            (tmproot / "README.md").write_text("# Test\n")
            src_dir = tmproot / "src"
            src_dir.mkdir()
            (src_dir / "mod.py").write_text("x = 1\n")

            # Check A: Happy path lifecycle
            async def happy_exec(cmd, timeout):
                if "pip install" in cmd:
                    return FakeExecResult(0, "Installed")
                if "pytest" in cmd:
                    return FakeExecResult(0, "3 passed in 0.5s")
                return FakeExecResult(0, "")

            sb_happy = FakeSandbox(happy_exec)
            client_happy = FakeClient(sb_happy)

            req = SandboxRequest(
                session_id="s_happy",
                project_root=str(tmproot),
                changed_files=[str(src_dir / "mod.py")],
                test_command="pytest",
            )

            runner_inst = SandboxRunner()
            res_happy = await runner_inst._run_in_sandbox(
                client_happy, sb_happy, req, time.monotonic()
            )
            assert res_happy.sandbox_status == "completed", f"Got {res_happy.sandbox_status}"
            assert res_happy.exit_code == 0
            assert res_happy.tests_passed == 3
            # Check all workspace files uploaded
            uploaded_dests = [dest for _, dest in sb_happy.fs.uploaded]
            assert "/workspace/pyproject.toml" in uploaded_dests
            assert "/workspace/README.md" in uploaded_dests
            assert "/workspace/src/mod.py" in uploaded_dests

            # Check B: Dependency installation failure
            async def dep_fail_exec(cmd, timeout):
                if "pip install" in cmd:
                    return FakeExecResult(1, "Could not resolve dependencies")
                return FakeExecResult(0, "3 passed")

            sb_dep = FakeSandbox(dep_fail_exec)
            res_dep = await runner_inst._run_in_sandbox(client_happy, sb_dep, req, time.monotonic())
            assert res_dep.sandbox_status == "error", (
                f"Expected error, got {res_dep.sandbox_status}"
            )
            assert res_dep.exit_code == 1
            assert "Dependency installation failed" in res_dep.logs

            # Check C: Linter timeout/error isolation (non-fatal)
            async def linter_fail_exec(cmd, timeout):
                if "pip install" in cmd:
                    return FakeExecResult(0, "Installed")
                if "ruff" in cmd:
                    raise RuntimeError("Linter process crashed")
                if "pytest" in cmd:
                    return FakeExecResult(0, "4 passed in 0.2s")
                return FakeExecResult(0, "")

            sb_lint = FakeSandbox(linter_fail_exec)
            req_lint = SandboxRequest(
                session_id="s_lint",
                project_root=str(tmproot),
                changed_files=[],
                test_command="pytest",
                linter_command="ruff check",
            )
            res_lint = await runner_inst._run_in_sandbox(
                client_happy, sb_lint, req_lint, time.monotonic()
            )
            assert res_lint.sandbox_status == "completed", (
                "Linter crash must not abort test execution"
            )
            assert res_lint.tests_passed == 4
            assert len(res_lint.linter_errors) == 1
            assert "Linter execution error" in res_lint.linter_errors[0]

            # Check D: Deletion retry and leak ID reporting
            client_fail_delete = FakeClient(sb_happy, delete_exc=RuntimeError("Network disconnect"))
            leak_msg = await runner_inst._delete_sandbox_with_retry(
                client_fail_delete, sb_happy, max_retries=2
            )
            assert leak_msg is not None
            assert "LEAKED_SANDBOX_sbx_test_123" in leak_msg
            assert len(client_fail_delete.delete_calls) == 2

            # Check E: Symlink escaping project root is rejected
            symlink_dir = tmproot / "sym_dir"
            symlink_dir.mkdir()
            escaped_target = Path(tmpdir).parent / "outside_secret.txt"
            escaped_target.write_text("secret")
            symlink_file = symlink_dir / "link.txt"
            try:
                symlink_file.symlink_to(escaped_target)
                upload_paths = runner_inst._collect_upload_paths(tmproot, [str(symlink_file)])
                # Resolved symlink must not be in upload paths
                assert not any(p.resolve() == escaped_target.resolve() for p in upload_paths)
            except (OSError, NotImplementedError):
                pass  # Symlinks might require elevation on some Windows environments

    asyncio.run(run_lifecycle_checks())
    print("SandboxRunner standalone self-check (Rule 2903681) passed.")
