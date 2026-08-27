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
    lace_client = _get_lace_client(workspace_root)
    git_diff = parse_git_diff(git_ctx.raw_diff)

    diff_summary = f"Branch '{git_ctx.branch_name}' (mode: {mode}) touching {len(git_ctx.touched_files)} file(s)"

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

    return 0


if __name__ == "__main__":
    # Standalone zero-dependency self-checks (Rule 2903681)
    def _self_test() -> None:
        parser = build_parser()
        args = parser.parse_args(["check", "--help"])
        assert args is not None
        print("cli.py standalone self-check passed successfully.")

    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        sys.exit(main())
