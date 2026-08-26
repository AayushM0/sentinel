"""Sentinel Subagent A: Daytona Sandbox Runner.

Runs the project test suite inside an isolated Daytona sandbox and returns
a typed SandboxResult. The sandbox is always deleted on exit.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    DaytonaProcessExecutionTimeoutError,
    Image,
)

from sentinel.models.review_state import SubagentStatus, SubagentType
from sentinel.session_store import SessionStore

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
                        await client.delete(sandbox)

        except DaytonaProcessExecutionTimeoutError:
            result = SandboxResult(
                sandbox_status="timeout",
                exit_code=-1,
                tests_passed=0,
                tests_failed=0,
                linter_errors=[],
                duration_ms=int((time.monotonic() - start) * 1000),
                logs="",
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

        self._persist(request, result)
        return result

    async def _run_in_sandbox(
        self,
        client: AsyncDaytona,
        sandbox,
        request: SandboxRequest,
        start: float,
    ) -> SandboxResult:
        """Upload files, install deps, run linter + tests, return result."""
        # 1. Upload changed files + project metadata
        await self._upload_files(sandbox, request)

        # 2. Install dependencies (fixed timeout, separate from test timeout)
        await sandbox.process.exec(
            f"cd {_WORKSPACE} && pip install uv -q && uv sync --dev",
            timeout=_DEPS_TIMEOUT,
        )

        # 3. Run linter (optional, non-fatal)
        linter_errors: list[str] = []
        if request.linter_command:
            lr = await sandbox.process.exec(
                f"cd {_WORKSPACE} && {request.linter_command}",
                timeout=request.timeout_seconds,
            )
            if lr.exit_code != 0:
                linter_errors = [line for line in lr.result.splitlines() if line.strip()]

        # 4. Run test command (authoritative for sandbox_status)
        tr = await sandbox.process.exec(
            f"cd {_WORKSPACE} && {request.test_command}",
            timeout=request.timeout_seconds,
        )
        passed, failed = self._parse_pytest_output(tr.result)
        status = "completed" if tr.exit_code == 0 else "failed"

        return SandboxResult(
            sandbox_status=status,
            exit_code=tr.exit_code,
            tests_passed=passed,
            tests_failed=failed,
            linter_errors=linter_errors,
            duration_ms=int((time.monotonic() - start) * 1000),
            logs=tr.result,
        )

    async def _upload_files(self, sandbox, request: SandboxRequest) -> None:
        """Upload changed files + pyproject.toml + uv.lock into /workspace."""
        root = Path(request.project_root)

        # Collect: changed files + project metadata
        extras = [str(root / f) for f in ("pyproject.toml", "uv.lock") if (root / f).exists()]
        paths = list(dict.fromkeys([*request.changed_files, *extras]))  # dedup, order preserved

        for local_path in paths:
            p = Path(local_path)
            if not p.exists():
                continue  # deleted file in diff — skip silently
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue  # outside project root — skip with no crash

            dest = f"{_WORKSPACE}/{rel.as_posix()}"
            # Ensure parent directory exists in sandbox
            parent = str(Path(dest).parent)
            if parent != _WORKSPACE:
                await sandbox.process.exec(f"mkdir -p {parent}")
            await sandbox.fs.upload_file(p.read_bytes(), dest)

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
    runner = SandboxRunner()

    # Standard summary line
    p, f = runner._parse_pytest_output("5 passed, 2 failed in 1.23s")
    assert p == 5, f"Expected 5 passed, got {p}"
    assert f == 2, f"Expected 2 failed, got {f}"

    # Empty output — exit_code carries the truth
    p2, f2 = runner._parse_pytest_output("")
    assert p2 == 0 and f2 == 0, "Empty output should yield (0, 0)"

    # Compile error — no summary line
    p3, f3 = runner._parse_pytest_output("ERROR collecting tests/test_foo.py\nImportError: ...")
    assert p3 == 0 and f3 == 0, "Compile error output should yield (0, 0)"

    # Passed only
    p4, f4 = runner._parse_pytest_output("32 passed in 1.38s")
    assert p4 == 32 and f4 == 0, f"Expected (32, 0), got ({p4}, {f4})"

    # SandboxResult.to_dict roundtrip
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

    # Invariant: "completed" matches approval_gate.py check
    assert sr.sandbox_status in ("completed", "success"), (
        "sandbox_status must match approval_gate.py gate check"
    )

    # SandboxRequest rejects non-positive timeout
    try:
        SandboxRequest(
            session_id="s1",
            project_root="/p",
            changed_files=[],
            test_command="pytest",
            timeout_seconds=0,
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    print("SandboxRunner standalone self-check passed.")
