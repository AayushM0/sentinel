from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinel.models.review_state import (
    ApprovalActionType,
    ApprovalDecision,
    PendingApproval,
    ReviewSession,
    SessionStatus,
    SubagentStatus,
    SubagentTask,
    SubagentType,
)


class SessionStore:
    """SQLite-backed session persistence store with WAL mode and reconnect resilience."""

    def __init__(self, db_path: str = ".sentinel/session.db") -> None:
        self.db_path = db_path
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_sessions (
                    session_id TEXT PRIMARY KEY,
                    branch_name TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    diff_summary TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS subagent_tasks (
                    task_id TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES review_sessions(session_id),
                    subagent_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_payload TEXT,
                    completed_at TEXT,
                    PRIMARY KEY (session_id, task_id)
                );

                CREATE TABLE IF NOT EXISTS pending_approvals (
                    approval_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES review_sessions(session_id),
                    action_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    user_decision TEXT NOT NULL,
                    decided_at TEXT
                );
                """
            )
            # Schema migration check: ensure subagent_tasks has composite primary key
            table_info = conn.execute("PRAGMA table_info(subagent_tasks);").fetchall()
            pk_cols = [r["name"] for r in table_info if r["pk"] > 0]
            if pk_cols and "session_id" not in pk_cols:
                # Legacy table detected with task_id PRIMARY KEY only - execute table rebuild migration
                conn.executescript(
                    """
                    CREATE TABLE subagent_tasks_v2 (
                        task_id TEXT NOT NULL,
                        session_id TEXT NOT NULL REFERENCES review_sessions(session_id),
                        subagent_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result_payload TEXT,
                        completed_at TEXT,
                        PRIMARY KEY (session_id, task_id)
                    );
                    INSERT OR IGNORE INTO subagent_tasks_v2 (task_id, session_id, subagent_type, status, result_payload, completed_at)
                    SELECT task_id, session_id, subagent_type, status, result_payload, completed_at FROM subagent_tasks;
                    DROP TABLE subagent_tasks;
                    ALTER TABLE subagent_tasks_v2 RENAME TO subagent_tasks;
                    """
                )

    def create_session(
        self,
        branch_name: str,
        commit_sha: str,
        diff_summary: str = "",
        session_id: str | None = None,
    ) -> ReviewSession:
        """Create a new review session in initial PENDING_SUBAGENTS status."""
        sess_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        now = datetime.now(UTC).isoformat()
        status = SessionStatus.PENDING_SUBAGENTS

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO review_sessions (session_id, branch_name, commit_sha, created_at, updated_at, status, diff_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sess_id, branch_name, commit_sha, now, now, status.value, diff_summary),
            )

        return ReviewSession(
            session_id=sess_id,
            branch_name=branch_name,
            commit_sha=commit_sha,
            created_at=now,
            updated_at=now,
            status=status,
            diff_summary=diff_summary,
        )

    def save_subagent_result(
        self,
        session_id: str,
        subagent_type: SubagentType,
        status: SubagentStatus = SubagentStatus.COMPLETED,
        result_payload: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> SubagentTask:
        """Record or update a subagent execution result scoped to session."""
        t_id = task_id or f"task_{subagent_type.value.lower()}_{uuid.uuid4().hex[:6]}"
        now = datetime.now(UTC).isoformat()
        payload_json = json.dumps(result_payload or {})

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO subagent_tasks (task_id, session_id, subagent_type, status, result_payload, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, task_id) DO UPDATE SET
                    status=excluded.status,
                    result_payload=excluded.result_payload,
                    completed_at=excluded.completed_at
                """,
                (t_id, session_id, subagent_type.value, status.value, payload_json, now),
            )
            # Update session timestamp
            conn.execute(
                "UPDATE review_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

        return SubagentTask(
            task_id=t_id,
            session_id=session_id,
            subagent_type=subagent_type,
            status=status,
            result_payload=result_payload or {},
            completed_at=now,
        )

    def set_pending_approval(
        self,
        session_id: str,
        action_type: ApprovalActionType,
        payload: dict[str, Any],
        approval_id: str | None = None,
    ) -> PendingApproval:
        """Transition session to PENDING_HUMAN_APPROVAL and record pending approval entity."""
        appr_id = approval_id or f"appr_{uuid.uuid4().hex[:8]}"
        payload_json = json.dumps(payload)
        now = datetime.now(UTC).isoformat()

        with self._get_connection() as conn:
            sess_row = conn.execute(
                "SELECT status FROM review_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not sess_row:
                raise ValueError(f"Session {session_id} not found")

            curr_status = SessionStatus(sess_row["status"])
            if curr_status in (SessionStatus.COMPLETED, SessionStatus.REJECTED):
                raise ValueError(
                    f"Cannot request approval for terminal session in status {curr_status.value}"
                )

            conn.execute(
                """
                INSERT INTO pending_approvals (approval_id, session_id, action_type, payload, user_decision, decided_at)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (
                    appr_id,
                    session_id,
                    action_type.value,
                    payload_json,
                    ApprovalDecision.PENDING.value,
                ),
            )
            conn.execute(
                "UPDATE review_sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (SessionStatus.PENDING_HUMAN_APPROVAL.value, now, session_id),
            )

        return PendingApproval(
            approval_id=appr_id,
            session_id=session_id,
            action_type=action_type,
            payload=payload,
            user_decision=ApprovalDecision.PENDING,
            decided_at=None,
        )

    def get_session(self, session_id: str) -> ReviewSession | None:
        """Hydrate complete review session with its subagent tasks and pending approval."""
        with self._get_connection() as conn:
            sess_row = conn.execute(
                "SELECT * FROM review_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not sess_row:
                return None

            task_rows = conn.execute(
                "SELECT * FROM subagent_tasks WHERE session_id = ?", (session_id,)
            ).fetchall()
            tasks = [
                SubagentTask(
                    task_id=r["task_id"],
                    session_id=r["session_id"],
                    subagent_type=SubagentType(r["subagent_type"]),
                    status=SubagentStatus(r["status"]),
                    result_payload=json.loads(r["result_payload"]) if r["result_payload"] else {},
                    completed_at=r["completed_at"],
                )
                for r in task_rows
            ]

            appr_row = conn.execute(
                "SELECT * FROM pending_approvals WHERE session_id = ? ORDER BY rowid DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            pending_appr = None
            if appr_row:
                pending_appr = PendingApproval(
                    approval_id=appr_row["approval_id"],
                    session_id=appr_row["session_id"],
                    action_type=ApprovalActionType(appr_row["action_type"]),
                    payload=json.loads(appr_row["payload"]) if appr_row["payload"] else {},
                    user_decision=ApprovalDecision(appr_row["user_decision"]),
                    decided_at=appr_row["decided_at"],
                )

            return ReviewSession(
                session_id=sess_row["session_id"],
                branch_name=sess_row["branch_name"],
                commit_sha=sess_row["commit_sha"],
                created_at=sess_row["created_at"],
                updated_at=sess_row["updated_at"],
                status=SessionStatus(sess_row["status"]),
                diff_summary=sess_row["diff_summary"],
                tasks=tasks,
                pending_approval=pending_appr,
            )

    def get_active_session(self) -> ReviewSession | None:
        """Find the latest active review session requiring attention or resumption."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT session_id FROM review_sessions
                WHERE status IN (?, ?)
                ORDER BY updated_at DESC LIMIT 1
                """,
                (
                    SessionStatus.PENDING_HUMAN_APPROVAL.value,
                    SessionStatus.PENDING_SUBAGENTS.value,
                ),
            ).fetchone()
            if not row:
                return None
            return self.get_session(row["session_id"])

    def resolve_approval(self, approval_id: str, decision: ApprovalDecision) -> None:
        """Record the user's decision at the approval gate with stale resolution protection."""
        if decision not in (ApprovalDecision.APPROVED, ApprovalDecision.REJECTED):
            raise ValueError(f"Cannot resolve approval with non-final decision {decision.value}")

        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            appr = conn.execute(
                "SELECT session_id, user_decision FROM pending_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if not appr:
                return

            # Only allow resolving approvals that are currently PENDING
            if appr["user_decision"] != ApprovalDecision.PENDING.value:
                return

            conn.execute(
                "UPDATE pending_approvals SET user_decision = ?, decided_at = ? WHERE approval_id = ?",
                (decision.value, now, approval_id),
            )

            # Check if this is the newest approval for this session
            latest = conn.execute(
                "SELECT approval_id FROM pending_approvals WHERE session_id = ? ORDER BY rowid DESC LIMIT 1",
                (appr["session_id"],),
            ).fetchone()

            if latest and latest["approval_id"] == approval_id:
                next_status = (
                    SessionStatus.APPROVED
                    if decision == ApprovalDecision.APPROVED
                    else SessionStatus.REJECTED
                )
                conn.execute(
                    "UPDATE review_sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                    (next_status.value, now, appr["session_id"]),
                )

    def mark_completed(self, session_id: str) -> None:
        """Mark a review session as successfully completed."""
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE review_sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (SessionStatus.COMPLETED.value, now, session_id),
            )


if __name__ == "__main__":
    # Framework-free self-check (Rule 2903681)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = str(Path(tmpdir) / "test.db")
        store = SessionStore(db_path=test_db)
        sess = store.create_session("main", "abc1234", "Diff test")
        assert sess.status == SessionStatus.PENDING_SUBAGENTS

        task = store.save_subagent_result(
            sess.session_id, SubagentType.SANDBOX_RUNNER, SubagentStatus.COMPLETED, {"tests": 5}
        )
        assert task.status == SubagentStatus.COMPLETED

        appr = store.set_pending_approval(
            sess.session_id, ApprovalActionType.PRE_PUSH_COMMIT, {"card": "Test"}
        )
        assert appr.user_decision == ApprovalDecision.PENDING

        store.resolve_approval(appr.approval_id, ApprovalDecision.APPROVED)
        hydrated = store.get_session(sess.session_id)
        assert hydrated is not None and hydrated.status == SessionStatus.APPROVED

        store.mark_completed(sess.session_id)
        assert store.get_session(sess.session_id).status == SessionStatus.COMPLETED

    print("SessionStore standalone self-check passed successfully.")
