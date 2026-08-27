"""TrueForge Agent Harness Adapter: Integrates Sentinel with TrueForge runtime contracts and tools."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Literal, Self

# Ensure src is on sys.path and script_dir does not shadow installed packages
_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import yaml
from pydantic import BaseModel, Field

from sentinel.approval_gate import ApprovalDecision
from sentinel.git_utils import extract_git_context
from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.diff import parse_git_diff
from sentinel.models.review_state import ApprovalActionType, SessionStatus
from sentinel.orchestrator import OrchestratorRequest, ReviewOrchestrator
from sentinel.session_store import SessionStore
from sentinel.subagents.sandbox_runner import SandboxRequest, SandboxRunner

logger = logging.getLogger("sentinel.trueforge_adapter")


class AgentMetadata(BaseModel):
    name: str = "sentinel"
    description: str = "PR Guardian and Architectural ADR Memory Agent"
    harness: str = "trueforge"


class StorageConfig(BaseModel):
    backend: str = "sqlite"
    path: str = ".sentinel/session.db"
    journal_mode: str = "WAL"
    busy_timeout_ms: int = 5000


class ApprovalPolicy(BaseModel):
    action: str
    require_human_confirmation: bool = True
    render_target: str = "all"


class ApprovalGateConfig(BaseModel):
    enabled: bool = True
    policies: list[ApprovalPolicy] = Field(default_factory=list)


class GatedTool(BaseModel):
    name: str
    description: str = ""
    approval_required: bool = True


def _default_gated_tools() -> list[GatedTool]:
    return [
        GatedTool(
            name="remember",
            description="Write or update an Architectural Decision Record in LACE vault",
            approval_required=True,
        ),
        GatedTool(
            name="git_push",
            description="Push local commits to remote GitHub repository",
            approval_required=True,
        ),
    ]


class SubagentSandboxConfig(BaseModel):
    provider: str = "daytona"
    default_test_cmd: str = "pytest tests/ -v"
    default_lint_cmd: str = "ruff check ."
    timeout_seconds: int = 30


class SubagentAdrConfig(BaseModel):
    model: str = "claude-3-5-sonnet"
    similarity_threshold: float = 0.75


class SubagentsConfig(BaseModel):
    sandbox_runner: SubagentSandboxConfig = Field(default_factory=SubagentSandboxConfig)
    adr_analyzer: SubagentAdrConfig = Field(default_factory=SubagentAdrConfig)


class McpServerConfig(BaseModel):
    command: str = "python"
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class TrueForgeConfig(BaseModel):
    version: str = "1.0.0"
    agent: AgentMetadata = Field(default_factory=AgentMetadata)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    approval_gate: ApprovalGateConfig = Field(default_factory=ApprovalGateConfig)
    gated_tools: list[GatedTool] = Field(default_factory=_default_gated_tools)
    subagents: SubagentsConfig = Field(default_factory=SubagentsConfig)
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)


class TrueForgeAdapter:
    """Adapter bridging Sentinel capabilities with the TrueForge Agent Harness."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else self._find_default_config()
        self.config = self._load_config()

    @staticmethod
    def _find_default_config() -> Path:
        """Search for trueforge.config.yaml in standard locations."""
        candidates = [
            Path("config/trueforge.config.yaml"),
            Path("trueforge.config.yaml"),
            Path(".sentinel/trueforge.config.yaml"),
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()
        return Path("config/trueforge.config.yaml").resolve()

    def _load_config(self) -> TrueForgeConfig:
        """Load and parse TrueForge configuration file, failing closed on malformed configs."""
        if self.config_path.is_file():
            try:
                raw_text = self.config_path.read_text(encoding="utf-8")
                parsed = yaml.safe_load(raw_text)
                if isinstance(parsed, dict):
                    return TrueForgeConfig.model_validate(parsed)
                raise ValueError(f"Config at {self.config_path} must be a YAML dictionary")
            except Exception as exc:
                raise ValueError(
                    f"Invalid TrueForge configuration at {self.config_path}: {exc}"
                ) from exc
        return TrueForgeConfig()

    def _get_lace_client(self) -> LaceMcpClient:
        """Instantiate LaceMcpClient using configured MCP server parameters."""
        lace_cfg = self.config.mcp_servers.get("lace")
        if lace_cfg:
            return LaceMcpClient(
                server_command=lace_cfg.command,
                server_args=lace_cfg.args,
                server_env=lace_cfg.env,
            )
        return LaceMcpClient()

    def is_tool_gated(self, tool_name: str) -> bool:
        """Check if a tool requires approval before execution."""
        for gt in self.config.gated_tools:
            if gt.name == tool_name:
                return gt.approval_required
        return False

    def requires_approval(self, action_type: str | ApprovalActionType) -> bool:
        """Check if an approval action requires human confirmation per policy."""
        if not self.config.approval_gate.enabled:
            return False
        action_val = (
            action_type.value if isinstance(action_type, ApprovalActionType) else str(action_type)
        )
        for policy in self.config.approval_gate.policies:
            if policy.action == action_val:
                return policy.require_human_confirmation
        return True

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return standardized JSON schema tool definitions for TrueForge agent discovery."""
        return [
            {
                "name": "sentinel_check_diff",
                "description": "Run Daytona sandbox tests and LACE ADR delta analysis on git workspace changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_path": {
                            "type": "string",
                            "description": "Path to repository workspace root (default: current directory).",
                            "default": ".",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["unstaged", "staged", "branch", "working_tree"],
                            "description": "Diff extraction mode (default: unstaged).",
                            "default": "unstaged",
                        },
                        "base_branch": {
                            "type": "string",
                            "description": "Base branch for comparison in branch mode (default: main).",
                            "default": "main",
                        },
                        "interactive": {
                            "type": "boolean",
                            "description": "Whether to prompt interactively for human approval.",
                            "default": True,
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "sentinel_query_adrs",
                "description": "Query historical Architectural Decision Records from LACE memory vault.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "touched_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of modified file paths to retrieve ADR context for.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Semantic search query for relevant ADRs.",
                            "default": "",
                        },
                    },
                    "required": ["touched_files"],
                },
            },
            {
                "name": "sentinel_run_sandbox",
                "description": "Execute isolated pytest and ruff linter checks inside Daytona sandbox.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_path": {
                            "type": "string",
                            "description": "Path to workspace directory containing tests to run.",
                            "default": ".",
                        },
                        "test_cmd": {
                            "type": "string",
                            "description": "Custom test command (default: pytest tests/ -v).",
                        },
                        "lint_cmd": {
                            "type": "string",
                            "description": "Custom lint command (default: ruff check .).",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "sentinel_resolve_approval",
                "description": "Resolve a pending TrueForge human approval gate decision.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "approval_id": {
                            "type": "string",
                            "description": "Unique identifier of the pending approval entity.",
                        },
                        "decision": {
                            "type": "string",
                            "enum": ["APPROVED", "REJECTED"],
                            "description": "Decision outcome.",
                        },
                    },
                    "required": ["approval_id", "decision"],
                },
            },
        ]

    async def check_diff(
        self,
        workspace_path: str = ".",
        mode: Literal["unstaged", "staged", "branch", "working_tree"] = "unstaged",
        base_branch: str = "main",
        interactive: bool = True,
        lace_client: LaceMcpClient | None = None,
        store: SessionStore | None = None,
    ) -> dict[str, Any]:
        """Execute full architectural and sandbox check through Sentinel orchestrator."""
        valid_modes = ("unstaged", "staged", "branch", "working_tree")
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid diff extraction mode '{mode}'. Must be one of {valid_modes}."
            )

        ws_root = Path(workspace_path).resolve()
        git_ctx = extract_git_context(ws_root, mode=mode, base_branch=base_branch)

        session_store = store or SessionStore(db_path=self.config.storage.path)
        client = lace_client or self._get_lace_client()
        git_diff = parse_git_diff(git_ctx.raw_diff)

        diff_summary = f"Branch '{git_ctx.branch_name}' (mode: {mode}) touching {len(git_ctx.touched_files)} file(s)"
        sess = session_store.create_session(
            branch_name=git_ctx.branch_name,
            commit_sha=git_ctx.commit_sha,
            diff_summary=diff_summary,
            raw_diff=git_ctx.raw_diff,
        )

        req = OrchestratorRequest(
            session_id=sess.session_id,
            branch_name=git_ctx.branch_name,
            commit_sha=git_ctx.commit_sha,
            diff_summary=diff_summary,
            git_diff=git_diff,
            touched_files=git_ctx.touched_files,
            workspace_root=ws_root,
            lace_client=client,
            session_store=session_store,
            interactive=interactive,
            timeout_seconds=self.config.subagents.sandbox_runner.timeout_seconds,
        )

        orchestrator = ReviewOrchestrator()
        try:
            if not client.is_connected:
                async with client:
                    result_session = await orchestrator.run_review(req)
            else:
                result_session = await orchestrator.run_review(req)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LACE MCP connection failed during check_diff: %s. Proceeding with fault-isolated review.",
                exc,
            )
            result_session = await orchestrator.run_review(req)

        return {
            "session_id": result_session.session_id,
            "status": result_session.status.value,
            "branch_name": result_session.branch_name,
            "commit_sha": result_session.commit_sha,
            "diff_summary": result_session.diff_summary,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "subagent_type": t.subagent_type.value,
                    "status": t.status.value,
                    "result_payload": t.result_payload,
                }
                for t in result_session.tasks
            ],
            "pending_approval": (
                {
                    "approval_id": result_session.pending_approval.approval_id,
                    "action_type": result_session.pending_approval.action_type.value,
                    "user_decision": result_session.pending_approval.user_decision.value,
                }
                if result_session.pending_approval
                else None
            ),
        }

    async def query_adrs(
        self,
        touched_files: list[str],
        query: str = "",
        lace_client: LaceMcpClient | None = None,
    ) -> list[dict[str, Any]]:
        """Query LACE ADR context directly with active connection management."""
        client = lace_client or self._get_lace_client()
        if not client.is_connected:
            async with client:
                adrs = await client.get_relevant_adrs(touched_files=touched_files, query=query)
        else:
            adrs = await client.get_relevant_adrs(touched_files=touched_files, query=query)

        return [
            {
                "title": adr.title,
                "status": adr.status.value if hasattr(adr.status, "value") else str(adr.status),
                "code_pattern": adr.code_pattern,
                "constraints": adr.constraints,
                "category": adr.category,
                "scope": adr.scope,
            }
            for adr in adrs
        ]

    async def run_sandbox(
        self,
        workspace_path: str = ".",
        test_cmd: str | None = None,
        lint_cmd: str | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Execute tests in Daytona sandbox."""
        ws_root = Path(workspace_path).resolve()
        timeout = (
            self.config.subagents.sandbox_runner.timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        req = SandboxRequest(
            session_id="tf_sandbox_test",
            project_root=str(ws_root),
            changed_files=[],
            test_command=test_cmd or self.config.subagents.sandbox_runner.default_test_cmd,
            linter_command=lint_cmd or self.config.subagents.sandbox_runner.default_lint_cmd,
            timeout_seconds=timeout,
        )
        runner = SandboxRunner()
        res = await runner.run(req)
        return {
            "sandbox_status": res.sandbox_status,
            "exit_code": res.exit_code,
            "tests_passed": res.tests_passed,
            "tests_failed": res.tests_failed,
            "linter_errors": res.linter_errors,
            "duration_ms": res.duration_ms,
            "logs": res.logs,
        }

    async def resolve_approval(
        self,
        approval_id: str,
        decision: str | ApprovalDecision,
        db_path: str | None = None,
        lace_client: LaceMcpClient | None = None,
    ) -> dict[str, Any]:
        """Resolve pending approval in session store, commit approved ADRs, and complete workflow."""
        store = SessionStore(db_path=db_path or self.config.storage.path)
        with store._get_connection() as conn:
            appr_row = conn.execute(
                "SELECT session_id, user_decision, payload FROM pending_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()

        if not appr_row:
            raise ValueError(f"Pending approval '{approval_id}' not found in database.")
        if appr_row["user_decision"] != ApprovalDecision.PENDING.value:
            raise ValueError(
                f"Pending approval '{approval_id}' has already been decided as '{appr_row['user_decision']}'."
            )

        dec = ApprovalDecision(decision) if isinstance(decision, str) else decision
        store.resolve_approval(approval_id, dec)

        if dec == ApprovalDecision.APPROVED:
            payload = appr_row["payload"] or {}
            if isinstance(payload, str):
                import json

                try:
                    payload = json.loads(payload)
                except Exception:  # noqa: BLE001
                    payload = {}

            proposed_adrs = payload.get("proposed_adrs", [])
            if proposed_adrs:
                from sentinel.models.adr import ADR

                client = lace_client or self._get_lace_client()
                if not client.is_connected:
                    async with client:
                        for p in proposed_adrs:
                            adr_obj = ADR(**p) if isinstance(p, dict) else p
                            await client.commit_adr(adr_obj)
                else:
                    for p in proposed_adrs:
                        adr_obj = ADR(**p) if isinstance(p, dict) else p
                        await client.commit_adr(adr_obj)

            store.mark_completed(appr_row["session_id"])

        return {
            "approval_id": approval_id,
            "status": "resolved",
            "decision": dec.value,
            "session_id": appr_row["session_id"],
        }


if __name__ == "__main__":
    # Standalone zero-dependency self-checks (Rule 2903681)
    import asyncio
    import tempfile

    class _DummyLaceClient:
        def __init__(self) -> None:
            self.is_connected = True

        async def get_relevant_adrs(self, touched_files: list[str], query: str = "") -> list[Any]:
            return []

        async def commit_adr(self, adr: Any) -> bool:
            return True

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    def _self_test() -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Verify default config parsing
            adapter = TrueForgeAdapter()
            assert adapter.config.agent.name == "sentinel"
            assert adapter.is_tool_gated("remember") is True
            assert adapter.is_tool_gated("git_push") is True
            assert adapter.is_tool_gated("unknown_tool") is False

            # 2. Verify tool definitions
            defs = adapter.get_tool_definitions()
            assert len(defs) == 4
            tool_names = [d["name"] for d in defs]
            assert "sentinel_check_diff" in tool_names
            assert "sentinel_resolve_approval" in tool_names

            # 3. Verify approval resolution through adapter
            db_path = str(tmp_path / "test_tf.db")
            store = SessionStore(db_path=db_path)
            sess = store.create_session("main", "sha123", "summary")
            appr = store.set_pending_approval(
                sess.session_id, ApprovalActionType.PRE_PUSH_COMMIT, {"card": "test"}
            )

            async def _async_self_tests() -> None:
                dummy_client = _DummyLaceClient()  # type: ignore[assignment]
                res = await adapter.resolve_approval(
                    appr.approval_id,
                    ApprovalDecision.APPROVED,
                    db_path=db_path,
                    lace_client=dummy_client,  # type: ignore[arg-type]
                )
                assert res["status"] == "resolved"
                hydrated = store.get_session(sess.session_id)
                assert hydrated is not None and hydrated.status == SessionStatus.COMPLETED

                # 4. Verify invalid approval resolution raises ValueError
                try:
                    await adapter.resolve_approval(
                        "nonexistent_id", ApprovalDecision.APPROVED, db_path=db_path
                    )
                    raise AssertionError("Should have raised ValueError on missing approval ID")
                except ValueError:
                    pass

                # 5. Async tool checks with dummy client
                adrs = await adapter.query_adrs(
                    touched_files=["test.py"],
                    lace_client=dummy_client,  # type: ignore[arg-type]
                )
                assert isinstance(adrs, list)

                try:
                    await adapter.check_diff(
                        workspace_path=str(tmp_path),
                        mode="invalid_mode",  # type: ignore[arg-type]
                    )
                    raise AssertionError("Should have rejected invalid mode")
                except ValueError:
                    pass

            asyncio.run(_async_self_tests())

        print("TrueForgeAdapter standalone self-check passed successfully.")

    _self_test()
