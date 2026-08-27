"""Concurrent Multi-Subagent Orchestrator for Sentinel Code Review Sessions.

Orchestrates Subagent A (Daytona SandboxRunner) and Subagent B (LACE ADRDeltaAnalyzer)
concurrently via asyncio, manages session state lifecycle, and connects to the Human Approval Gate.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure src is on sys.path for direct script execution
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from sentinel.approval_gate import ApprovalGate
from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.adr import ADR
from sentinel.models.diff import GitDiff
from sentinel.models.review_state import (
    ApprovalDecision,
    ReviewSession,
    SessionStatus,
    SubagentStatus,
    SubagentType,
)
from sentinel.session_store import SessionStore
from sentinel.subagents.adr_delta_analyzer import (
    ADRDeltaAnalyzer,
    DeltaReport,
    DeltaRequest,
)
from sentinel.subagents.sandbox_runner import (
    SandboxRequest,
    SandboxResult,
    SandboxRunner,
)

logger = logging.getLogger("sentinel.orchestrator")


@dataclass
class OrchestratorRequest:
    """Input payload for coordinating a full review session."""

    session_id: str
    branch_name: str
    commit_sha: str
    diff_summary: str
    git_diff: GitDiff
    touched_files: list[str]
    workspace_root: Path
    lace_client: LaceMcpClient
    session_store: SessionStore
    daytona_client: Any = None
    interactive: bool = True
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


class ReviewOrchestrator:
    """Coordinates parallel subagent execution and human approval gates."""

    def __init__(
        self,
        sandbox_runner: SandboxRunner | None = None,
        adr_analyzer: ADRDeltaAnalyzer | None = None,
        approval_gate: ApprovalGate | None = None,
    ) -> None:
        self.sandbox_runner = sandbox_runner or SandboxRunner()
        self.adr_analyzer = adr_analyzer or ADRDeltaAnalyzer()
        self.approval_gate = approval_gate or ApprovalGate()

    async def run_review(self, request: OrchestratorRequest) -> ReviewSession:
        """Run subagents concurrently, prompt approval gate, and commit approved ADRs."""
        # 1. Inspect existing session state in store
        existing = request.session_store.get_session(request.session_id)
        if existing is not None:
            # Short-circuit if session is already in a terminal state
            if existing.status in (SessionStatus.COMPLETED, SessionStatus.REJECTED):
                return existing

            # If pending approval and subagent results exist, resume without rerunning subagents
            if existing.status == SessionStatus.PENDING_HUMAN_APPROVAL and existing.tasks:
                sb_task = next(
                    (t for t in existing.tasks if t.subagent_type == SubagentType.SANDBOX_RUNNER),
                    None,
                )
                adr_task = next(
                    (
                        t
                        for t in existing.tasks
                        if t.subagent_type == SubagentType.ADR_DELTA_ANALYZER
                    ),
                    None,
                )
                if sb_task and adr_task:
                    test_result = sb_task.result_payload or {}
                    delta_report = adr_task.result_payload or {}
                    self.approval_gate.session_store = request.session_store
                    decision = self.approval_gate.request_approval(
                        session=existing,
                        test_result=test_result,
                        delta_report=delta_report,
                        interactive=request.interactive,
                    )
                    return await self._handle_approval_decision(
                        request, existing, decision, delta_report
                    )
        else:
            request.session_store.create_session(
                session_id=request.session_id,
                branch_name=request.branch_name,
                commit_sha=request.commit_sha,
                diff_summary=request.diff_summary,
            )

        # 2. Build subagent request payloads matching exact AST models
        abs_workspace = str(request.workspace_root.resolve())
        abs_changed_files = [
            str((request.workspace_root / f).resolve()) for f in request.touched_files
        ]

        sandbox_req = SandboxRequest(
            session_id=request.session_id,
            project_root=abs_workspace,
            changed_files=abs_changed_files,
            test_command="uv run pytest -v",
            linter_command="uv run ruff check src",
            timeout_seconds=request.timeout_seconds,
        )

        delta_req = DeltaRequest(
            session_id=request.session_id,
            git_diff=request.git_diff,
            touched_files=request.touched_files,
            lace_client=request.lace_client,
            timeout_seconds=request.timeout_seconds,
        )

        # 3. Execute Subagents concurrently with fault isolation
        sandbox_coro = self._run_sandbox_safely(sandbox_req, request.session_store)
        adr_coro = self._run_adr_safely(delta_req, request.session_store)

        sb_res, adr_res = await asyncio.gather(sandbox_coro, adr_coro)

        # 4. Prepare structured dictionary payloads for ApprovalGate
        test_result = sb_res.to_dict() if hasattr(sb_res, "to_dict") else sb_res
        delta_report = adr_res.to_dict() if hasattr(adr_res, "to_dict") else adr_res

        # 5. Fetch updated session model
        session = request.session_store.get_session(request.session_id)
        if session is None:
            raise RuntimeError(f"Session {request.session_id} not found in store")

        # 6. Intercept with Human Approval Gate
        self.approval_gate.session_store = request.session_store
        decision = self.approval_gate.request_approval(
            session=session,
            test_result=test_result,
            delta_report=delta_report,
            interactive=request.interactive,
        )

        # 7. Apply human decision & post-approval side effects
        return await self._handle_approval_decision(request, session, decision, adr_res)

    async def _handle_approval_decision(
        self,
        request: OrchestratorRequest,
        session: ReviewSession,
        decision: ApprovalDecision,
        adr_res: Any,
    ) -> ReviewSession:
        """Process approval outcome, commit proposed ADRs, and transition session lifecycle."""
        if decision == ApprovalDecision.APPROVED:
            session.status = SessionStatus.APPROVED
            all_committed = True

            # Rehydrate proposed ADRs if present
            proposed_list = getattr(adr_res, "proposed_adrs", None)
            if not proposed_list and isinstance(adr_res, dict):
                raw_proposed = adr_res.get("proposed_adrs", [])
                proposed_list = [ADR(**p) if isinstance(p, dict) else p for p in raw_proposed]

            if proposed_list:
                for proposed in proposed_list:
                    try:
                        success = await request.lace_client.commit_adr(adr=proposed)
                        if not success:
                            all_committed = False
                            logger.error("LACE MCP rejected commit of ADR %s", proposed.id)
                        else:
                            logger.info("Committed proposed ADR %s to LACE vault", proposed.id)
                    except Exception as exc:  # noqa: BLE001
                        all_committed = False
                        logger.error("Failed to commit ADR %s to LACE: %s", proposed.id, exc)

            if all_committed:
                request.session_store.mark_completed(request.session_id)
                session.status = SessionStatus.COMPLETED
            else:
                logger.warning(
                    "Session %s left in APPROVED state due to ADR commit failure",
                    request.session_id,
                )

        elif decision == ApprovalDecision.REJECTED:
            session.status = SessionStatus.REJECTED
        else:
            session.status = SessionStatus.PENDING_HUMAN_APPROVAL

        return session

    async def _run_sandbox_safely(self, req: SandboxRequest, store: SessionStore) -> SandboxResult:
        """Execute SandboxRunner with top-level error capture; single persistence owner."""
        self.sandbox_runner.session_store = store
        try:
            return await self.sandbox_runner.run(req)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"SandboxRunner failed: {exc}"
            logger.error("%s\n%s", err_msg, traceback.format_exc())
            store.save_subagent_result(
                session_id=req.session_id,
                subagent_type=SubagentType.SANDBOX_RUNNER,
                status=SubagentStatus.FAILED,
                result_payload={"error": str(exc), "traceback": traceback.format_exc()},
                task_id=f"{req.session_id}_sandbox",
            )
            return SandboxResult(
                sandbox_status="crashed",
                exit_code=1,
                tests_passed=0,
                tests_failed=0,
                linter_errors=[],
                duration_ms=0,
                logs=str(exc),
            )

    async def _run_adr_safely(self, req: DeltaRequest, store: SessionStore) -> DeltaReport:
        """Execute ADRDeltaAnalyzer with top-level error capture; single persistence owner."""
        try:
            return await self.adr_analyzer.run(req, session_store=store)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"ADRDeltaAnalyzer failed: {exc}"
            logger.error("%s\n%s", err_msg, traceback.format_exc())
            store.save_subagent_result(
                session_id=req.session_id,
                subagent_type=SubagentType.ADR_DELTA_ANALYZER,
                status=SubagentStatus.FAILED,
                result_payload={"error": str(exc), "traceback": traceback.format_exc()},
                task_id=f"{req.session_id}_adr",
            )
            return DeltaReport(
                session_id=req.session_id,
                violations=[],
                summary=f"ADRDeltaAnalyzer failed: {exc}",
            )


if __name__ == "__main__":
    import tempfile
    from unittest.mock import AsyncMock

    from sentinel.models.diff import parse_git_diff

    # Standalone zero-dependency self-checks (Rule 2903681)
    async def _self_test() -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = SessionStore(db_path=tmp_path / "self_test.db")
            mock_client = AsyncMock(spec=LaceMcpClient)
            mock_client.get_relevant_adrs = AsyncMock(return_value=[])
            mock_client.commit_adr = AsyncMock(return_value=True)

            # 1. Non-interactive pending approval flow
            req = OrchestratorRequest(
                session_id="self_orch_sess",
                branch_name="main",
                commit_sha="abcdef1",
                diff_summary="test summary",
                git_diff=parse_git_diff(""),
                touched_files=[],
                workspace_root=tmp_path,
                lace_client=mock_client,
                session_store=store,
                interactive=False,
            )

            orch = ReviewOrchestrator()
            session = await orch.run_review(req)
            assert session.session_id == "self_orch_sess"
            assert session.status == SessionStatus.PENDING_HUMAN_APPROVAL

            # 2. Validation constraints
            try:
                OrchestratorRequest(
                    session_id="",
                    branch_name="main",
                    commit_sha="abcdef1",
                    diff_summary="empty",
                    git_diff=parse_git_diff(""),
                    touched_files=[],
                    workspace_root=tmp_path,
                    lace_client=mock_client,
                    session_store=store,
                )
                raise AssertionError("Should have raised ValueError on empty session_id")
            except ValueError:
                pass

            print("ReviewOrchestrator standalone self-check passed successfully.")

    asyncio.run(_self_test())
