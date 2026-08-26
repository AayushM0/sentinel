import tempfile
from pathlib import Path

import pytest

from sentinel.approval_gate import ApprovalGate
from sentinel.models.review_state import (
    ApprovalDecision,
    SessionStatus,
)
from sentinel.session_store import SessionStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_gate.db"
        yield SessionStore(db_path=str(db_path))


def test_generate_approval_card():
    gate = ApprovalGate()

    test_result = {
        "sandbox_status": "SUCCESS",
        "exit_code": 0,
        "tests_passed": 3,
        "tests_failed": 0,
        "duration_ms": 1420,
    }

    delta_report = {
        "violations": [
            "Line 42 violates ADR-014: direct access to window.localStorage is forbidden."
        ],
        "proposed_adrs": [{"id": "ADR-015", "title": "Encrypted Storage Wrapper"}],
        "modified_adrs": ["ADR-014"],
    }

    card = gate.format_approval_card(
        session_id="sess_demo123",
        branch_name="feat/auth",
        commit_sha="789abc",
        test_result=test_result,
        delta_report=delta_report,
    )

    assert "Sentinel Human-in-the-Loop Approval Card" in card
    assert "sess_demo123" in card
    assert "feat/auth" in card
    assert "3/3 (1420ms)" in card
    assert "Line 42 violates ADR-014" in card
    assert "ADR-015" in card
    assert "Encrypted Storage Wrapper" in card
    assert "[Approve & Push]" in card


def test_approval_gate_create_and_resolve(temp_store: SessionStore):
    gate = ApprovalGate(session_store=temp_store)
    session = temp_store.create_session(
        branch_name="feat/auth",
        commit_sha="789abc",
        diff_summary="Auth update",
    )

    test_result = {"exit_code": 0, "tests_passed": 3}
    delta_report = {"violations": []}

    # Non-interactive / server mode should register pending approval and return PENDING
    decision = gate.request_approval(
        session=session,
        test_result=test_result,
        delta_report=delta_report,
        interactive=False,
    )
    assert decision == ApprovalDecision.PENDING

    # Check session store state
    active = temp_store.get_active_session()
    assert active is not None
    assert active.status == SessionStatus.PENDING_HUMAN_APPROVAL
    assert active.pending_approval is not None

    # Resolve approval
    gate.resolve_approval(active.pending_approval.approval_id, ApprovalDecision.APPROVED)

    resolved = temp_store.get_session(session.session_id)
    assert resolved.status == SessionStatus.APPROVED
    assert resolved.pending_approval.user_decision == ApprovalDecision.APPROVED


def test_approval_gate_crashed_runner_shows_failure():
    gate = ApprovalGate()
    crashed_result = {
        "sandbox_status": "crashed",
        "exit_code": 1,
        "tests_passed": 0,
        "tests_failed": 0,
    }
    card = gate.format_approval_card(
        session_id="sess_crashed",
        branch_name="feat/crash",
        commit_sha="c0ffee1",
        test_result=crashed_result,
        delta_report={},
    )
    assert "**Status:** `FAILURE`" in card


def test_approval_gate_strict_input_rejection(monkeypatch, temp_store: SessionStore):
    gate = ApprovalGate(session_store=temp_store)

    # Ambiguous input like "abort" or "apple" must REJECT
    monkeypatch.setattr("builtins.input", lambda _: "abort")
    session_reject = temp_store.create_session(branch_name="feat/reject", commit_sha="111222")
    decision = gate.request_approval(
        session=session_reject, test_result={"exit_code": 0}, delta_report={}, interactive=True
    )
    assert decision == ApprovalDecision.REJECTED

    # Exact "approve" must APPROVE — fresh session since rejected sessions are terminal
    monkeypatch.setattr("builtins.input", lambda _: "approve")
    session_approve = temp_store.create_session(branch_name="feat/approve", commit_sha="333444")
    decision2 = gate.request_approval(
        session=session_approve, test_result={"exit_code": 0}, delta_report={}, interactive=True
    )
    assert decision2 == ApprovalDecision.APPROVED


if __name__ == "__main__":
    test_generate_approval_card()
    test_approval_gate_crashed_runner_shows_failure()
    print("test_approval_gate.py standalone checks passed.")
