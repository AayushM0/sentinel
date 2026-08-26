from __future__ import annotations

from typing import Any

from sentinel.models.review_state import (
    ApprovalActionType,
    ApprovalDecision,
    ReviewSession,
)
from sentinel.session_store import SessionStore


class ApprovalGate:
    """TrueForge Human-in-the-Loop Approval Gate interceptor."""

    def __init__(self, session_store: SessionStore | None = None) -> None:
        self.session_store = session_store

    def format_approval_card(
        self,
        session_id: str,
        branch_name: str,
        commit_sha: str,
        test_result: dict[str, Any],
        delta_report: dict[str, Any],
    ) -> str:
        """Format a rich Markdown approval card for Chat UI and CLI terminal."""
        tests_passed = test_result.get("tests_passed", 0)
        tests_failed = test_result.get("tests_failed", 0)
        total_tests = tests_passed + tests_failed
        duration_ms = test_result.get("duration_ms", 0)

        violations = delta_report.get("violations", [])
        proposed_adrs = delta_report.get("proposed_adrs", [])
        modified_adrs = delta_report.get("modified_adrs", [])

        status_str = "SUCCESS" if tests_failed == 0 else "FAILURE"
        pass_str = (
            f"{tests_passed}/{total_tests} ({duration_ms}ms)"
            if total_tests > 0
            else f"{tests_passed} passed"
        )

        lines = [
            "### Sentinel Human-in-the-Loop Approval Card",
            f"**Session:** `{session_id}` | **Branch:** `{branch_name}` | **Commit:** `{commit_sha[:7]}`",
            "",
            "#### 1. Daytona Sandbox Verification",
            f"- **Status:** `{status_str}`",
            f"- **Tests Passed:** {pass_str}",
        ]

        if test_result.get("linter_errors"):
            lines.append("- **Linter Warnings/Errors:**")
            for err in test_result["linter_errors"]:
                lines.append(f"  - [WARN] `{err}`")

        lines.extend(
            [
                "",
                "#### 2. LACE Architectural Delta & ADR Analysis",
            ]
        )

        if violations:
            lines.append("- **Rule Violations Detected:**")
            for v in violations:
                lines.append(f"  - [VIOLATION] {v}")
        else:
            lines.append("- **Rule Violations:** Zero violations detected.")

        if modified_adrs:
            lines.append(f"- **Updated ADRs:** {', '.join(modified_adrs)}")

        if proposed_adrs:
            lines.append("- **New Pattern Proposals:**")
            for p in proposed_adrs:
                p_id = p.get("id", "ADR-NEW")
                p_title = p.get("title", "Untitled Pattern")
                lines.append(f"  - **{p_id}**: {p_title}")

        lines.extend(
            [
                "",
                "---",
                "**Action Required:**",
                "- `[Approve & Push]`: Proceed with Git push and commit ADRs to LACE vault.",
                "- `[Reject / Edit]`: Abort push and report remediation instructions.",
            ]
        )

        sep = chr(10)
        return sep.join(lines)

    def request_approval(
        self,
        session: ReviewSession,
        test_result: dict[str, Any],
        delta_report: dict[str, Any],
        action_type: ApprovalActionType = ApprovalActionType.PRE_PUSH_COMMIT,
        interactive: bool = True,
    ) -> ApprovalDecision:
        """Prompt user for approval or register pending approval state in SQLite."""
        card = self.format_approval_card(
            session_id=session.session_id,
            branch_name=session.branch_name,
            commit_sha=session.commit_sha,
            test_result=test_result,
            delta_report=delta_report,
        )

        pending_appr = None
        if self.session_store is not None:
            pending_appr = self.session_store.set_pending_approval(
                session_id=session.session_id,
                action_type=action_type,
                payload={
                    "card": card,
                    "test_result": test_result,
                    "delta_report": delta_report,
                },
            )

        if not interactive:
            return ApprovalDecision.PENDING

        sep = chr(10)
        print(sep + "=" * 60)
        print(card)
        print("=" * 60 + sep)

        try:
            choice = input("Select Action -> [a]pprove / [r]eject: ").strip().lower()
            decision = (
                ApprovalDecision.APPROVED if choice.startswith("a") else ApprovalDecision.REJECTED
            )
        except (KeyboardInterrupt, EOFError):
            decision = ApprovalDecision.REJECTED

        if self.session_store is not None and pending_appr is not None:
            self.session_store.resolve_approval(pending_appr.approval_id, decision)

        return decision

    def resolve_approval(
        self,
        approval_id: str,
        decision: ApprovalDecision,
    ) -> None:
        """Resolve a pending approval by ID."""
        if self.session_store is not None:
            self.session_store.resolve_approval(approval_id, decision)
