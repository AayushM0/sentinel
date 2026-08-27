"""Sentinel CLI Suite: Automated Architectural Guardrail and Code Review Engine."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import logging
import sys
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

# Ensure src is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from sentinel.git_utils import GitError, extract_git_context
from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.diff import parse_git_diff
from sentinel.models.review_state import SessionStatus
from sentinel.orchestrator import OrchestratorRequest, ReviewOrchestrator
from sentinel.session_store import SessionStore

logger = logging.getLogger("sentinel.cli")


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine safely, supporting both sync CLI and active test loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


def _get_lace_client(workspace_root: Path) -> LaceMcpClient:
    """Instantiate and configure LaceMcpClient for workspace."""
    return LaceMcpClient()


def cmd_check(args: argparse.Namespace) -> int:
    """Execute pre-push architectural and sandbox check on current diff."""
    workspace_root = Path(args.workspace).resolve()

    # Determine extraction mode
    mode = "unstaged"
    if args.staged:
        mode = "staged"
    elif args.branch:
        mode = "branch"
    elif hasattr(args, "mode") and args.mode:
        mode = args.mode

    try:
        git_ctx = extract_git_context(
            repo_path=workspace_root,
            mode=mode,
            base_branch=args.base_branch,
        )
    except GitError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    store = SessionStore(db_path=args.db_path)
    diff_summary = f"Branch '{git_ctx.branch_name}' (mode: {mode}) touching {len(git_ctx.touched_files)} file(s)"
    store.create_session(
        session_id=session_id,
        branch_name=git_ctx.branch_name,
        commit_sha=git_ctx.commit_sha,
        diff_summary=diff_summary,
        raw_diff=git_ctx.raw_diff,
    )
    lace_client = _get_lace_client(workspace_root)
    git_diff = parse_git_diff(git_ctx.raw_diff)

    req = OrchestratorRequest(
        session_id=session_id,
        branch_name=git_ctx.branch_name,
        commit_sha=git_ctx.commit_sha,
        diff_summary=diff_summary,
        git_diff=git_diff,
        touched_files=git_ctx.touched_files,
        workspace_root=workspace_root,
        lace_client=lace_client,
        session_store=store,
        interactive=not args.non_interactive,
        timeout_seconds=args.timeout,
    )

    orchestrator = ReviewOrchestrator()
    try:
        session = _run_async(orchestrator.run_review(req))
    except Exception as exc:  # noqa: BLE001
        print(f"Error running review orchestrator: {exc}", file=sys.stderr)
        return 2

    if session.status in (SessionStatus.COMPLETED, SessionStatus.APPROVED):
        print(f"Sentinel: Review session '{session.session_id}' approved.")
        return 0
    elif session.status == SessionStatus.REJECTED:
        print(f"Sentinel: Review session '{session.session_id}' rejected.", file=sys.stderr)
        return 1
    elif session.status == SessionStatus.PENDING_HUMAN_APPROVAL:
        if args.non_interactive:
            print(f"Sentinel: Review session '{session.session_id}' pending approval.")
            return 3
        return 0

    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume a review session from SQLite storage."""
    workspace_root = Path(args.workspace).resolve()
    store = SessionStore(db_path=args.db_path)

    session_id = args.session_id
    if not session_id:
        latest = store.get_latest_pending_session()
        if not latest:
            print("Sentinel: No pending review sessions found to resume.")
            return 0
        session_id = latest.session_id

    session = store.get_session(session_id)
    if not session:
        print(f"Error: Review session '{session_id}' not found in database.", file=sys.stderr)
        return 2

    if session.status in (SessionStatus.COMPLETED, SessionStatus.APPROVED, SessionStatus.REJECTED):
        print(
            f"Sentinel: Review session '{session_id}' is already in terminal status '{session.status.value}'."
        )
        return 0 if session.status in (SessionStatus.COMPLETED, SessionStatus.APPROVED) else 1

    lace_client = _get_lace_client(workspace_root)
    git_diff = parse_git_diff(session.raw_diff)
    touched = [f.path for f in git_diff.files]

    req = OrchestratorRequest(
        session_id=session.session_id,
        branch_name=session.branch_name,
        commit_sha=session.commit_sha,
        diff_summary=session.diff_summary,
        git_diff=git_diff,
        touched_files=touched,
        workspace_root=workspace_root,
        lace_client=lace_client,
        session_store=store,
        interactive=True,
        timeout_seconds=args.timeout,
    )

    orchestrator = ReviewOrchestrator()
    try:
        res = _run_async(orchestrator.run_review(req))
    except Exception as exc:  # noqa: BLE001
        print(f"Error resuming review session: {exc}", file=sys.stderr)
        return 2

    if res.status in (SessionStatus.COMPLETED, SessionStatus.APPROVED):
        print(f"Sentinel: Review session '{res.session_id}' approved.")
        return 0
    elif res.status == SessionStatus.REJECTED:
        print(f"Sentinel: Review session '{res.session_id}' rejected.", file=sys.stderr)
        return 1

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List recent review sessions and their outcomes from SQLite."""
    bounded_limit = max(1, min(args.limit, 100))
    store = SessionStore(db_path=args.db_path)
    sessions = store.list_sessions(limit=bounded_limit)

    if not sessions:
        print("Sentinel: No review sessions found in database.")
        return 0

    print(f"{'SESSION ID':<20} {'BRANCH':<20} {'COMMIT':<12} {'STATUS':<24} {'UPDATED AT'}")
    print("-" * 90)
    for s in sessions:
        short_commit = s.commit_sha[:10] if len(s.commit_sha) >= 10 else s.commit_sha
        print(
            f"{s.session_id:<20} {s.branch_name:<20} {short_commit:<12} {s.status.value:<24} {s.updated_at}"
        )

    return 0


def cmd_install_hook(args: argparse.Namespace) -> int:
    """Install pre-push git hook into repository."""
    from sentinel.hooks import HookError, install_pre_push_hook

    workspace_root = Path(args.workspace).resolve()
    try:
        hook_path = install_pre_push_hook(workspace_root)
        print(f"Sentinel: pre-push hook installed successfully at {hook_path}")
        return 0
    except HookError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def cmd_uninstall_hook(args: argparse.Namespace) -> int:
    """Remove pre-push git hook from repository."""
    from sentinel.hooks import HookError, uninstall_pre_push_hook

    workspace_root = Path(args.workspace).resolve()
    try:
        removed = uninstall_pre_push_hook(workspace_root)
        if removed:
            print("Sentinel: pre-push hook removed successfully.")
        else:
            print("Sentinel: No pre-push hook found to remove.")
        return 0
    except HookError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for Sentinel."""
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Sentinel: Automated Architectural Guardrail & Subagent Verification System",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # check command
    check_parser = subparsers.add_parser(
        "check",
        help="Run Daytona sandbox tests and LACE ADR delta analysis on current diff",
    )
    check_parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="Path to workspace root directory (default: current directory)",
    )
    check_parser.add_argument(
        "--base-branch",
        "-b",
        default="main",
        help="Base branch for branch diff mode (default: main)",
    )
    check_parser.add_argument(
        "--mode",
        choices=["unstaged", "staged", "branch", "working_tree"],
        default="unstaged",
        help="Diff extraction mode (default: unstaged)",
    )
    check_parser.add_argument(
        "--staged",
        action="store_true",
        help="Shorthand for --mode staged",
    )
    check_parser.add_argument(
        "--branch",
        action="store_true",
        help="Shorthand for --mode branch",
    )
    check_parser.add_argument(
        "--non-interactive",
        "-n",
        action="store_true",
        help="Run without interactive terminal prompts",
    )
    check_parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=60,
        help="Timeout in seconds for subagent execution (default: 60)",
    )
    check_parser.add_argument(
        "--db-path",
        default=".sentinel/session.db",
        help="Path to SQLite session database (default: .sentinel/session.db)",
    )

    # resume command
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an existing or pending review session from SQLite storage",
    )
    resume_parser.add_argument(
        "session_id",
        nargs="?",
        default=None,
        help="Session ID to resume (default: most recent pending approval session)",
    )
    resume_parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="Path to workspace root directory (default: current directory)",
    )
    resume_parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=60,
        help="Timeout in seconds for subagent execution (default: 60)",
    )
    resume_parser.add_argument(
        "--db-path",
        default=".sentinel/session.db",
        help="Path to SQLite session database (default: .sentinel/session.db)",
    )

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List recent review sessions and statuses",
    )
    list_parser.add_argument(
        "--db-path",
        default=".sentinel/session.db",
        help="Path to SQLite session database (default: .sentinel/session.db)",
    )
    list_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=20,
        help="Maximum number of sessions to display (default: 20)",
    )

    # install-hook command
    install_parser = subparsers.add_parser(
        "install-hook",
        help="Install Sentinel pre-push git hook into current repository",
    )
    install_parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="Path to workspace root directory (default: current directory)",
    )

    # uninstall-hook command
    uninstall_parser = subparsers.add_parser(
        "uninstall-hook",
        help="Remove Sentinel pre-push git hook from current repository",
    )
    uninstall_parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="Path to workspace root directory (default: current directory)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand or args.subcommand == "check":
        # Default to check if no subcommand provided with flags
        if not hasattr(args, "workspace"):
            # Invoked as bare `sentinel` with no args
            args = parser.parse_args(["check", *(argv or [])])
        return cmd_check(args)
    elif args.subcommand == "resume":
        return cmd_resume(args)
    elif args.subcommand == "list":
        return cmd_list(args)
    elif args.subcommand == "install-hook":
        return cmd_install_hook(args)
    elif args.subcommand == "uninstall-hook":
        return cmd_uninstall_hook(args)

    return 0


if __name__ == "__main__":
    # Standalone zero-dependency self-checks (Rule 2903681)
    def _self_test() -> None:
        parser = build_parser()

        # 1. Parse check with flags
        args_check = parser.parse_args(
            ["check", "--staged", "--non-interactive", "--timeout", "30"]
        )
        assert args_check.subcommand == "check"
        assert args_check.staged is True
        assert args_check.non_interactive is True
        assert args_check.timeout == 30

        # 2. Parse resume
        args_resume = parser.parse_args(["resume", "sess_test_123", "--timeout", "45"])
        assert args_resume.subcommand == "resume"
        assert args_resume.session_id == "sess_test_123"
        assert args_resume.timeout == 45

        # 3. Parse list
        args_list = parser.parse_args(["list", "--limit", "15"])
        assert args_list.subcommand == "list"
        assert args_list.limit == 15

        # 4. Parse install-hook / uninstall-hook
        args_inst = parser.parse_args(["install-hook", "--workspace", "."])
        assert args_inst.subcommand == "install-hook"

        print("cli.py standalone self-check passed successfully.")

    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        sys.exit(main())
