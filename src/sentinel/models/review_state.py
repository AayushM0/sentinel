from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Finite State Machine lifecycle status for review sessions."""

    PENDING_SUBAGENTS = "PENDING_SUBAGENTS"
    PENDING_HUMAN_APPROVAL = "PENDING_HUMAN_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"


class SubagentType(str, Enum):
    """Specialized parallel subagent types."""

    SANDBOX_RUNNER = "SANDBOX_RUNNER"
    ADR_DELTA_ANALYZER = "ADR_DELTA_ANALYZER"


class SubagentStatus(str, Enum):
    """Execution status for subagent background tasks."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ApprovalActionType(str, Enum):
    """Action intercepted by the TrueForge Human-in-the-Loop gate."""

    PRE_PUSH_COMMIT = "PRE_PUSH_COMMIT"
    LACE_ADR_UPDATE = "LACE_ADR_UPDATE"
    PR_SUBMISSION = "PR_SUBMISSION"


class ApprovalDecision(str, Enum):
    """Human decision recorded at the approval gate."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EDITED = "EDITED"


class ReviewSession(BaseModel):
    """Top-level review session entity."""

    session_id: str
    branch_name: str
    commit_sha: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: SessionStatus = SessionStatus.PENDING_SUBAGENTS
    diff_summary: str = ""
    tasks: list[SubagentTask] = Field(default_factory=list)
    pending_approval: PendingApproval | None = None


class SubagentTask(BaseModel):
    """Subagent execution task entity."""

    task_id: str
    session_id: str
    subagent_type: SubagentType
    status: SubagentStatus = SubagentStatus.RUNNING
    result_payload: dict[str, Any] = Field(default_factory=dict)
    completed_at: str | None = None


class PendingApproval(BaseModel):
    """Pending human approval entity."""

    approval_id: str
    session_id: str
    action_type: ApprovalActionType
    payload: dict[str, Any] = Field(default_factory=dict)
    user_decision: ApprovalDecision = ApprovalDecision.PENDING
    decided_at: str | None = None
