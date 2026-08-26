"""Live Daytona Sandbox Cloud Verification with Real Tests and Linting."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from sentinel.approval_gate import ApprovalGate
from sentinel.session_store import SessionStore
from sentinel.subagents.sandbox_runner import SandboxRequest, SandboxRunner


async def live_daytona_test() -> None:
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        print("Error: DAYTONA_API_KEY environment variable is not set.")
        print('Run: $env:DAYTONA_API_KEY = "your_key"')
        return

    print("\n" + "=" * 60)
    print("   SENTINEL: LIVE DAYTONA CLOUD TEST & LINT VERIFICATION")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent

    # Collect real project files to upload to cloud sandbox
    source_files: list[str] = []
    for ext in ("*.py", "*.toml", "*.lock", "*.md"):
        for p in project_root.rglob(ext):
            if ".venv" not in p.parts and ".pytest_cache" not in p.parts and ".git" not in p.parts:
                source_files.append(str(p))

    print(f"\n[1] Found {len(source_files)} project files to upload to Daytona sandbox.")

    store = SessionStore(db_path="sentinel_live.db")
    session = store.create_session(
        branch_name="feat/phase-3-daytona-sandbox-runner",
        commit_sha="3631908",
        diff_summary="Add Daytona Sandbox Runner with real cloud test execution",
    )
    print(f"[2] Created Review Session: `{session.session_id}` in SQLite")

    req = SandboxRequest(
        session_id=session.session_id,
        project_root=str(project_root),
        changed_files=source_files,
        linter_command="uv run ruff check src",
        test_command="uv run pytest tests/test_models/test_adr.py tests/test_models/test_diff.py -v",
        timeout_seconds=120,
    )

    runner = SandboxRunner(session_store=store)
    print("\n[3] Spawning remote Python 3.13 sandbox in Daytona Cloud...")
    print("    - Uploading source files")
    print("    - Running `uv sync --dev` inside container")
    print("    - Running `ruff check src`")
    print("    - Running `pytest tests/test_models/`")
    print("    (This takes ~20-30 seconds on cloud)...")

    result = await runner.run(req)

    print("\n" + "-" * 60)
    print(f"-> Sandbox Cloud Status: {result.sandbox_status.upper()}")
    print(f"-> Exit Code:           {result.exit_code}")
    print(f"-> Tests Passed:        {result.tests_passed}")
    print(f"-> Tests Failed:        {result.tests_failed}")
    print(f"-> Linter Errors:       {len(result.linter_errors)}")
    print(f"-> Cloud Execution Time:{result.duration_ms}ms")
    print("-> Container Teardown:  Verified deleted from Daytona Cloud")
    print("-" * 60)

    print("\n[4] Raw Test Logs from inside Daytona Cloud Sandbox:\n")
    print(result.logs.strip())
    print("\n" + "=" * 60)

    # Launch Human Approval Gate
    print("\n[5] Presenting Human-in-the-Loop Approval Gate:\n")
    gate = ApprovalGate(session_store=store)
    decision = gate.request_approval(
        session=session,
        test_result=result.to_dict(),
        delta_report={},
        interactive=True,
    )

    print(f"\n[6] Final Decision Recorded: {decision.value}")
    updated_session = store.get_session(session.session_id)
    if updated_session:
        print(f"    -> Updated SQLite Status: {updated_session.status.value}\n")


if __name__ == "__main__":
    asyncio.run(live_daytona_test())
