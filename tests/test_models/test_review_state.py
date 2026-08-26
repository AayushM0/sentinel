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


def test_review_state_enums():
    assert SessionStatus.PENDING_SUBAGENTS.value == "PENDING_SUBAGENTS"
    assert SessionStatus.PENDING_HUMAN_APPROVAL.value == "PENDING_HUMAN_APPROVAL"
    assert SessionStatus.APPROVED.value == "APPROVED"
    assert SessionStatus.REJECTED.value == "REJECTED"
    assert SessionStatus.COMPLETED.value == "COMPLETED"

    assert SubagentType.SANDBOX_RUNNER.value == "SANDBOX_RUNNER"
    assert SubagentType.ADR_DELTA_ANALYZER.value == "ADR_DELTA_ANALYZER"

    assert SubagentStatus.RUNNING.value == "RUNNING"
    assert SubagentStatus.COMPLETED.value == "COMPLETED"
    assert SubagentStatus.FAILED.value == "FAILED"

    assert ApprovalActionType.PRE_PUSH_COMMIT.value == "PRE_PUSH_COMMIT"
    assert ApprovalDecision.PENDING.value == "PENDING"
    assert ApprovalDecision.APPROVED.value == "APPROVED"


def test_review_session_model():
    session = ReviewSession(
        session_id="sess_123",
        branch_name="feat/auth",
        commit_sha="a1b2c3d",
        diff_summary="Modified session.ts",
    )
    assert session.session_id == "sess_123"
    assert session.status == SessionStatus.PENDING_SUBAGENTS
    assert session.branch_name == "feat/auth"
    assert session.created_at is not None


def test_subagent_task_model():
    task = SubagentTask(
        task_id="task_sandbox_1",
        session_id="sess_123",
        subagent_type=SubagentType.SANDBOX_RUNNER,
        status=SubagentStatus.COMPLETED,
        result_payload={"exit_code": 0, "passed": 3},
    )
    assert task.task_id == "task_sandbox_1"
    assert task.subagent_type == SubagentType.SANDBOX_RUNNER
    assert task.result_payload["passed"] == 3


def test_pending_approval_model():
    approval = PendingApproval(
        approval_id="appr_001",
        session_id="sess_123",
        action_type=ApprovalActionType.PRE_PUSH_COMMIT,
        payload={"diff": "some diff", "rules": ["ADR-014"]},
    )
    assert approval.approval_id == "appr_001"
    assert approval.user_decision == ApprovalDecision.PENDING
    assert approval.decided_at is None
