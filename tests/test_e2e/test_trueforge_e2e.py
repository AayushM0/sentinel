"""Deterministic End-to-End Cross-Session Lifecycle Tests for TrueForge & Sentinel."""

from __future__ import annotations

from pathlib import Path
from typing import Self
from unittest.mock import AsyncMock

import pytest

from sentinel.approval_gate import ApprovalDecision
from sentinel.models.adr import ADR
from sentinel.models.diff import parse_git_diff
from sentinel.models.review_state import (
    ApprovalActionType,
    SessionStatus,
    SubagentStatus,
    SubagentType,
)
from sentinel.orchestrator import OrchestratorRequest, ReviewOrchestrator
from sentinel.session_store import SessionStore
from sentinel.subagents.sandbox_runner import SandboxResult
from sentinel.trueforge_adapter import TrueForgeAdapter


class InMemoryLaceVault:
    """In-memory simulated LACE vault for cross-session deterministic E2E testing."""

    def __init__(self) -> None:
        self.is_connected = True
        self.vault: list[ADR] = []

    async def get_relevant_adrs(self, touched_files: list[str], query: str = "") -> list[ADR]:
        return [adr for adr in self.vault if adr.status in ("accepted", "proposed")]

    async def commit_adr(self, adr: ADR) -> bool:
        adr.status = "accepted"
        self.vault.append(adr)
        return True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


@pytest.mark.asyncio
async def test_trueforge_e2e_cross_session_lifecycle(tmp_path: Path) -> None:
    """Session 1: Architectural pivot -> proposed ADR -> approval -> commit.
    Session 2: Cross-session memory recall -> violation detected on subsequent diff.
    """
    db_path = str(tmp_path / "sentinel_e2e.db")
    store = SessionStore(db_path=db_path)
    lace_vault = InMemoryLaceVault()
    adapter = TrueForgeAdapter()

    # -------------------------------------------------------------------------
    # 1. Session 1: Architectural Pivot with duckdb
    # -------------------------------------------------------------------------
    sess1 = store.create_session(
        branch_name="feat/analytics-duckdb",
        commit_sha="a1b2c3d4e5f60000000000000000000000000001",
        diff_summary="Add DuckDB analytics engine",
        raw_diff="""diff --git a/src/analytics.py b/src/analytics.py
new file mode 100644
--- /dev/null
+++ b/src/analytics.py
@@ -0,0 +1,5 @@
+import duckdb
+
+def run_query(q: str):
+    return duckdb.sql(q)
""",
    )

    diff1 = parse_git_diff(sess1.raw_diff)

    # Subagent A Mock (Daytona Sandbox Passes)
    mock_sandbox_runner = AsyncMock()
    mock_sandbox_runner.run.return_value = SandboxResult(
        sandbox_status="completed",
        exit_code=0,
        tests_passed=12,
        tests_failed=0,
        linter_errors=[],
        duration_ms=250,
        logs="12 passed in 0.25s",
    )

    orchestrator1 = ReviewOrchestrator(sandbox_runner=mock_sandbox_runner)
    req1 = OrchestratorRequest(
        session_id=sess1.session_id,
        branch_name=sess1.branch_name,
        commit_sha=sess1.commit_sha,
        diff_summary=sess1.diff_summary,
        git_diff=diff1,
        touched_files=["src/analytics.py"],
        workspace_root=tmp_path,
        lace_client=lace_vault,  # type: ignore[arg-type]
        session_store=store,
        interactive=False,
    )

    result_sess1 = await orchestrator1.run_review(req1)

    # Invariants verification for Session 1 review
    assert result_sess1.status == SessionStatus.PENDING_HUMAN_APPROVAL
    assert result_sess1.pending_approval is not None
    assert result_sess1.pending_approval.action_type == ApprovalActionType.PRE_PUSH_COMMIT

    # Verify proposed ADR created by Subagent B
    adr_task1 = next(
        t for t in result_sess1.tasks if t.subagent_type == SubagentType.ADR_DELTA_ANALYZER
    )
    assert adr_task1.status == SubagentStatus.COMPLETED
    assert len(adr_task1.result_payload["proposed_adrs"]) == 1
    proposed_adr = adr_task1.result_payload["proposed_adrs"][0]
    assert "duckdb" in proposed_adr["title"].lower() or "duckdb" in proposed_adr["body"].lower()

    # Human Approves the Review Gate
    resolution = await adapter.resolve_approval(
        approval_id=result_sess1.pending_approval.approval_id,
        decision=ApprovalDecision.APPROVED,
        db_path=db_path,
        lace_client=lace_vault,  # type: ignore[arg-type]
    )

    assert resolution["status"] == "resolved"
    assert resolution["decision"] == "APPROVED"

    # Confirm session status completed and ADR committed into LACE vault
    completed_sess1 = store.get_session(sess1.session_id)
    assert completed_sess1 is not None
    assert completed_sess1.status == SessionStatus.COMPLETED
    assert len(lace_vault.vault) == 1
    committed_adr = lace_vault.vault[0]
    assert committed_adr.status == "accepted"

    # -------------------------------------------------------------------------
    # 2. Session 2: Cross-Session Recall & Enforcement
    # -------------------------------------------------------------------------
    # Configure code_pattern on the committed ADR to enforce pattern checks
    committed_adr.code_pattern = "duckdb.sql"
    committed_adr.constraints = ["Do not use raw duckdb.sql without connection wrapper"]

    sess2 = store.create_session(
        branch_name="feat/violating-change",
        commit_sha="a1b2c3d4e5f60000000000000000000000000002",
        diff_summary="Violate duckdb architectural pattern",
        raw_diff="""diff --git a/src/raw_exec.py b/src/raw_exec.py
new file mode 100644
--- /dev/null
+++ b/src/raw_exec.py
@@ -0,0 +1,3 @@
+import duckdb
+def execute_raw():
+    duckdb.sql("SELECT 1")
""",
    )

    diff2 = parse_git_diff(sess2.raw_diff)

    orchestrator2 = ReviewOrchestrator(sandbox_runner=mock_sandbox_runner)
    req2 = OrchestratorRequest(
        session_id=sess2.session_id,
        branch_name=sess2.branch_name,
        commit_sha=sess2.commit_sha,
        diff_summary=sess2.diff_summary,
        git_diff=diff2,
        touched_files=["src/raw_exec.py"],
        workspace_root=tmp_path,
        lace_client=lace_vault,  # type: ignore[arg-type]
        session_store=store,
        interactive=False,
    )

    result_sess2 = await orchestrator2.run_review(req2)

    # Verify Session 2 recalled the ADR committed in Session 1 and detected violation
    adr_task2 = next(
        t for t in result_sess2.tasks if t.subagent_type == SubagentType.ADR_DELTA_ANALYZER
    )
    assert adr_task2.status == SubagentStatus.COMPLETED
    assert len(adr_task2.result_payload["violations"]) >= 1
    assert any("matches prohibited pattern" in v for v in adr_task2.result_payload["violations"])


@pytest.mark.asyncio
async def test_trueforge_e2e_rejection_lifecycle(tmp_path: Path) -> None:
    """When a human rejects the approval gate, no ADRs are committed and session is REJECTED."""
    db_path = str(tmp_path / "sentinel_reject.db")
    store = SessionStore(db_path=db_path)
    lace_vault = InMemoryLaceVault()
    adapter = TrueForgeAdapter()

    sess = store.create_session("feat/rejected-feature", "sha999", "Reject test")
    appr = store.set_pending_approval(
        sess.session_id,
        ApprovalActionType.PRE_PUSH_COMMIT,
        {
            "proposed_adrs": [
                {
                    "id": "ADR-999",
                    "title": "Use Redis Cache",
                    "status": "proposed",
                    "context": "Testing rejection",
                }
            ]
        },
    )

    res = await adapter.resolve_approval(
        approval_id=appr.approval_id,
        decision=ApprovalDecision.REJECTED,
        db_path=db_path,
        lace_client=lace_vault,  # type: ignore[arg-type]
    )

    assert res["status"] == "resolved"
    assert res["decision"] == "REJECTED"
    assert len(lace_vault.vault) == 0

    hydrated = store.get_session(sess.session_id)
    assert hydrated is not None
    assert hydrated.status == SessionStatus.REJECTED
