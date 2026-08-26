import tempfile
from pathlib import Path

import pytest

from sentinel.models.review_state import (
    ApprovalActionType,
    ApprovalDecision,
    SessionStatus,
    SubagentStatus,
    SubagentType,
)
from sentinel.session_store import SessionStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_session.db"
        store = SessionStore(db_path=str(db_path))
        yield store


def test_session_store_init_and_create(temp_store: SessionStore):
    session = temp_store.create_session(
        branch_name="feat/auth",
        commit_sha="abc1234",
        diff_summary="Touched session.ts",
    )
    assert session.session_id.startswith("sess_")
    assert session.status == SessionStatus.PENDING_SUBAGENTS
    assert session.branch_name == "feat/auth"

    # Re-fetch from DB
    fetched = temp_store.get_session(session.session_id)
    assert fetched is not None
    assert fetched.session_id == session.session_id
    assert fetched.commit_sha == "abc1234"
    assert fetched.status == SessionStatus.PENDING_SUBAGENTS


def test_session_store_save_subagent_results(temp_store: SessionStore):
    session = temp_store.create_session(
        branch_name="feat/storage",
        commit_sha="def5678",
        diff_summary="Storage update",
    )

    # Save Subagent A result
    task_a = temp_store.save_subagent_result(
        session_id=session.session_id,
        subagent_type=SubagentType.SANDBOX_RUNNER,
        status=SubagentStatus.COMPLETED,
        result_payload={"exit_code": 0, "tests_passed": 5, "tests_failed": 0},
    )
    assert task_a.status == SubagentStatus.COMPLETED

    # Save Subagent B result
    task_b = temp_store.save_subagent_result(
        session_id=session.session_id,
        subagent_type=SubagentType.ADR_DELTA_ANALYZER,
        status=SubagentStatus.COMPLETED,
        result_payload={"violations": ["ADR-014 line 42"], "novel_patterns": []},
    )
    assert task_b.status == SubagentStatus.COMPLETED

    # Verify session hydration includes tasks
    hydrated = temp_store.get_session(session.session_id)
    assert hydrated is not None
    assert len(hydrated.tasks) == 2
    types = [t.subagent_type for t in hydrated.tasks]
    assert SubagentType.SANDBOX_RUNNER in types
    assert SubagentType.ADR_DELTA_ANALYZER in types


def test_session_store_pending_approval_and_reconnect(temp_store: SessionStore):
    session = temp_store.create_session(
        branch_name="feat/storage",
        commit_sha="def5678",
        diff_summary="Storage update",
    )

    # Set pending approval
    approval = temp_store.set_pending_approval(
        session_id=session.session_id,
        action_type=ApprovalActionType.PRE_PUSH_COMMIT,
        payload={"diff_summary": "Storage update", "violations": 1},
    )
    assert approval.approval_id.startswith("appr_")
    assert approval.user_decision == ApprovalDecision.PENDING

    # Verify session status is updated to PENDING_HUMAN_APPROVAL
    s_after = temp_store.get_session(session.session_id)
    assert s_after is not None
    assert s_after.status == SessionStatus.PENDING_HUMAN_APPROVAL
    assert s_after.pending_approval is not None

    # Simulate client disconnect and reconnect via get_active_session()
    active_sess = temp_store.get_active_session()
    assert active_sess is not None
    assert active_sess.session_id == session.session_id
    assert active_sess.status == SessionStatus.PENDING_HUMAN_APPROVAL
    assert active_sess.pending_approval.approval_id == approval.approval_id

    # Resolve approval
    temp_store.resolve_approval(approval.approval_id, ApprovalDecision.APPROVED)

    # Verify approval state
    s_resolved = temp_store.get_session(session.session_id)
    assert s_resolved.pending_approval.user_decision == ApprovalDecision.APPROVED
    assert s_resolved.pending_approval.decided_at is not None

    # Mark completed
    temp_store.mark_completed(session.session_id)
    s_completed = temp_store.get_session(session.session_id)
    assert s_completed.status == SessionStatus.COMPLETED

    # Verify no active pending session exists
    assert temp_store.get_active_session() is None


def test_session_store_stale_approval_protection(temp_store: SessionStore):
    session = temp_store.create_session(
        branch_name="feat/billing",
        commit_sha="c0ffee1",
        diff_summary="Billing changes",
    )

    # 1. Create first approval
    appr1 = temp_store.set_pending_approval(
        session_id=session.session_id,
        action_type=ApprovalActionType.PRE_PUSH_COMMIT,
        payload={"round": 1},
    )

    # 2. User edits diff, triggers second approval
    appr2 = temp_store.set_pending_approval(
        session_id=session.session_id,
        action_type=ApprovalActionType.PRE_PUSH_COMMIT,
        payload={"round": 2},
    )

    # Resolving first (stale) approval must NOT override session status of active appr2
    temp_store.resolve_approval(appr1.approval_id, ApprovalDecision.REJECTED)

    s = temp_store.get_session(session.session_id)
    # Session status must remain PENDING_HUMAN_APPROVAL because appr2 is the latest pending one!
    assert s.status == SessionStatus.PENDING_HUMAN_APPROVAL

    # Resolving the latest (appr2) will update session status
    temp_store.resolve_approval(appr2.approval_id, ApprovalDecision.APPROVED)
    s2 = temp_store.get_session(session.session_id)
    assert s2.status == SessionStatus.APPROVED


def test_session_store_task_id_scoped_to_session(temp_store: SessionStore):
    sess1 = temp_store.create_session(branch_name="b1", commit_sha="sha1")
    sess2 = temp_store.create_session(branch_name="b2", commit_sha="sha2")

    # Use same task_id for both sessions
    temp_store.save_subagent_result(
        session_id=sess1.session_id,
        subagent_type=SubagentType.SANDBOX_RUNNER,
        status=SubagentStatus.COMPLETED,
        result_payload={"result": "sess1"},
        task_id="shared_task_id",
    )

    temp_store.save_subagent_result(
        session_id=sess2.session_id,
        subagent_type=SubagentType.SANDBOX_RUNNER,
        status=SubagentStatus.COMPLETED,
        result_payload={"result": "sess2"},
        task_id="shared_task_id",
    )

    # Verify sess1 tasks are not corrupted by sess2
    s1_hydrated = temp_store.get_session(sess1.session_id)
    s2_hydrated = temp_store.get_session(sess2.session_id)

    assert len(s1_hydrated.tasks) == 1
    assert s1_hydrated.tasks[0].result_payload["result"] == "sess1"

    assert len(s2_hydrated.tasks) == 1
    assert s2_hydrated.tasks[0].result_payload["result"] == "sess2"


def test_session_store_legacy_db_migration(tmp_path):
    # Simulate an existing SQLite database with legacy schema (task_id PRIMARY KEY only)
    db_file = tmp_path / "legacy.db"
    import sqlite3

    conn = sqlite3.connect(str(db_file))
    conn.executescript(
        """
        CREATE TABLE review_sessions (
            session_id TEXT PRIMARY KEY,
            branch_name TEXT NOT NULL,
            commit_sha TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            diff_summary TEXT NOT NULL
        );
        CREATE TABLE subagent_tasks (
            task_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES review_sessions(session_id),
            subagent_type TEXT NOT NULL,
            status TEXT NOT NULL,
            result_payload TEXT,
            completed_at TEXT
        );
        CREATE TABLE pending_approvals (
            approval_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES review_sessions(session_id),
            action_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            user_decision TEXT NOT NULL,
            decided_at TEXT
        );
        """
    )
    conn.close()

    # Now initialize SessionStore on this pre-existing database
    store = SessionStore(db_path=str(db_file))
    sess1 = store.create_session(branch_name="feat/legacy1", commit_sha="123456")
    sess2 = store.create_session(branch_name="feat/legacy2", commit_sha="654321")

    # Upserting identical task_id in two distinct sessions must succeed on migrated DB
    task1 = store.save_subagent_result(
        session_id=sess1.session_id,
        subagent_type=SubagentType.SANDBOX_RUNNER,
        status=SubagentStatus.COMPLETED,
        result_payload={"legacy_migration": "sess1"},
        task_id="shared_task",
    )
    task2 = store.save_subagent_result(
        session_id=sess2.session_id,
        subagent_type=SubagentType.SANDBOX_RUNNER,
        status=SubagentStatus.COMPLETED,
        result_payload={"legacy_migration": "sess2"},
        task_id="shared_task",
    )
    assert task1.status == SubagentStatus.COMPLETED
    assert task2.status == SubagentStatus.COMPLETED


def test_session_store_terminal_session_cannot_reopen(temp_store: SessionStore):
    session = temp_store.create_session(branch_name="feat/terminal", commit_sha="sha999")
    temp_store.mark_completed(session.session_id)

    # Attempting to set pending approval on completed session must raise ValueError
    with pytest.raises(ValueError, match="Cannot request approval for terminal session"):
        temp_store.set_pending_approval(
            session_id=session.session_id,
            action_type=ApprovalActionType.PRE_PUSH_COMMIT,
            payload={},
        )


def test_session_store_nonfinal_decision_raises(temp_store: SessionStore):
    session = temp_store.create_session(branch_name="feat/test", commit_sha="sha888")
    appr = temp_store.set_pending_approval(
        session_id=session.session_id,
        action_type=ApprovalActionType.PRE_PUSH_COMMIT,
        payload={},
    )
    with pytest.raises(ValueError, match="Cannot resolve approval with non-final decision"):
        temp_store.resolve_approval(appr.approval_id, ApprovalDecision.PENDING)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(db_path=str(Path(tmpdir) / "test.db"))
        s = store.create_session("main", "111")
        assert s.status == SessionStatus.PENDING_SUBAGENTS
    print("test_session_store.py standalone checks passed.")
