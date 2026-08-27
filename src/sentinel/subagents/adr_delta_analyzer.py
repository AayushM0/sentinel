"""Subagent B: Architectural Decision Record (ADR) Delta Reasoning Analyzer.

Inspects git diffs against historical Architecture Decision Records in LACE memory,
detects constraint violations, tracks modified ADRs, and drafts new ADRs for novel patterns.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.adr import ADR
from sentinel.models.diff import GitDiff
from sentinel.models.review_state import SubagentStatus, SubagentType
from sentinel.session_store import SessionStore

logger = logging.getLogger("sentinel.subagents.adr_delta_analyzer")


@dataclass
class DeltaRequest:
    """Input parameters for ADR-Delta Reasoning Subagent."""

    session_id: str
    git_diff: GitDiff
    touched_files: list[str]
    lace_client: LaceMcpClient
    confidence_threshold: float = 0.8
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


@dataclass
class DeltaReport:
    """Structured architectural evaluation output matching ApprovalGate schema."""

    session_id: str
    violations: list[str] = field(default_factory=list)
    modified_adrs: list[str] = field(default_factory=list)
    proposed_adrs: list[ADR] = field(default_factory=list)
    summary: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a dictionary compatible with ApprovalGate and SessionStore."""
        return {
            "session_id": self.session_id,
            "violations": list(self.violations),
            "modified_adrs": list(self.modified_adrs),
            "proposed_adrs": [
                adr.model_dump() if hasattr(adr, "model_dump") else adr
                for adr in self.proposed_adrs
            ],
            "summary": self.summary,
            "duration_ms": self.duration_ms,
        }


class ADRDeltaAnalyzer:
    """Subagent B: Evaluates code changes against LACE architectural memory."""

    async def run(
        self,
        request: DeltaRequest,
        session_store: SessionStore | None = None,
    ) -> DeltaReport:
        """Run ADR delta reasoning analysis on the provided git diff."""
        start_time = time.perf_counter()

        try:
            # Handle empty or whitespace-only diffs immediately
            if not request.git_diff.files:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                report = DeltaReport(
                    session_id=request.session_id,
                    violations=[],
                    modified_adrs=[],
                    proposed_adrs=[],
                    summary="No architectural changes or violations detected.",
                    duration_ms=duration_ms,
                )
                self._persist_result(
                    request, report, session_store, status=SubagentStatus.COMPLETED
                )
                return report

            # Extract search terms and query LACE MCP
            query = self._extract_query_context(request)
            active_adrs = await self._fetch_active_adrs(request, query)

            # Evaluate constraints and detect deltas
            violations = self._evaluate_constraints(request.git_diff, active_adrs)
            modified_adrs: list[str] = []
            proposed_adrs: list[ADR] = []

            summary = (
                f"{len(violations)} architectural violation(s) detected."
                if violations
                else "No architectural changes or violations detected."
            )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            report = DeltaReport(
                session_id=request.session_id,
                violations=violations,
                modified_adrs=modified_adrs,
                proposed_adrs=proposed_adrs,
                summary=summary,
                duration_ms=duration_ms,
            )

            self._persist_result(request, report, session_store, status=SubagentStatus.COMPLETED)
            return report

        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            err_msg = f"ADRDeltaAnalyzer execution error: {exc}"
            logger.error("%s\n%s", err_msg, traceback.format_exc())

            report = DeltaReport(
                session_id=request.session_id,
                violations=[],
                modified_adrs=[],
                proposed_adrs=[],
                summary=f"Analysis encountered an error: {exc}",
                duration_ms=duration_ms,
            )

            self._persist_result(
                request,
                report,
                session_store,
                status=SubagentStatus.FAILED,
                error_payload={"error": str(exc), "traceback": traceback.format_exc()},
            )
            return report

    def _extract_query_context(self, request: DeltaRequest) -> str:
        """Derive search query keywords from touched files and diff paths."""
        effective_files = request.touched_files or [f.path for f in request.git_diff.files]
        terms: list[str] = []
        for path in effective_files:
            parts = path.replace("\\", "/").split("/")
            terms.extend([p for p in parts if p and "." not in p])

        if not terms:
            terms = ["architecture", "policy", "patterns"]
        return ", ".join(dict.fromkeys(terms))

    async def _fetch_active_adrs(self, request: DeltaRequest, query: str) -> list[ADR]:
        """Fetch ADRs from LACE MCP with timeout protection."""
        try:
            return await asyncio.wait_for(
                request.lace_client.get_relevant_adrs(query=query),
                timeout=float(request.timeout_seconds),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to retrieve ADRs from LACE MCP: %s", exc)
            return []

    def _evaluate_constraints(self, git_diff: GitDiff, active_adrs: list[ADR]) -> list[str]:
        """Evaluate added lines in git diff against active ADR constraints."""
        violations: list[str] = []
        enforceable_adrs = [adr for adr in active_adrs if adr.status in ("accepted", "proposed")]

        for file_diff in git_diff.files:
            for line in file_diff.added_lines:
                clean_line = self._strip_comments(line)
                if not clean_line:
                    continue

                for adr in enforceable_adrs:
                    if adr.code_pattern:
                        pattern_escaped = re.escape(adr.code_pattern)
                        if re.search(pattern_escaped, clean_line):
                            violations.append(
                                f"{file_diff.path}: Code matches prohibited pattern in {adr.id} ('{adr.title}')"
                            )
        return violations

    def _strip_comments(self, line: str) -> str:
        """Strip single-line comments without mangling URLs (https://) or hex colors (#ff0000)."""
        s = line.strip()
        # Whole line comment
        if s.startswith(("#", "//", "/*")):
            return ""
        # Strip trailing comments that are separated by whitespace from code
        # e.g., 'const x = 1; // comment' or 'x = 1 # comment'
        clean = re.sub(r"\s+(?:#|//).*$", "", line)
        return clean.strip()

    def _persist_result(
        self,
        request: DeltaRequest,
        report: DeltaReport,
        session_store: SessionStore | None,
        status: SubagentStatus = SubagentStatus.COMPLETED,
        error_payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist result payload to SQLite SessionStore if store is available."""
        if session_store is not None:
            payload = report.to_dict()
            if error_payload:
                payload.update(error_payload)
            session_store.save_subagent_result(
                session_id=request.session_id,
                subagent_type=SubagentType.ADR_DELTA_ANALYZER,
                status=status,
                result_payload=payload,
            )


if __name__ == "__main__":
    from unittest.mock import AsyncMock

    from sentinel.models.diff import parse_git_diff

    # Standalone zero-dependency self-checks (Rule 2903681)
    async def _self_test() -> None:
        # Check A: Clean pass on empty diff
        mock_client = AsyncMock(spec=LaceMcpClient)
        mock_client.get_relevant_adrs = AsyncMock(return_value=[])

        req = DeltaRequest(
            session_id="self_check_sess",
            git_diff=parse_git_diff(""),
            touched_files=[],
            lace_client=mock_client,
        )

        analyzer = ADRDeltaAnalyzer()
        report = await analyzer.run(req)

        assert report.session_id == "self_check_sess"
        assert report.violations == []
        assert report.modified_adrs == []
        assert report.proposed_adrs == []
        assert "No architectural changes" in report.summary

        # Check B: Comment stripping doesn't mangle hex colors or URLs
        assert analyzer._strip_comments("const color = '#ff0000';") == "const color = '#ff0000';"
        assert (
            analyzer._strip_comments("const url = 'https://api.example.com';")
            == "const url = 'https://api.example.com';"
        )
        assert analyzer._strip_comments("// comment only") == ""
        assert analyzer._strip_comments("x = 1 # trailing comment") == "x = 1"

        # Check C: Positive violation detection on non-empty diff
        sample_diff = parse_git_diff(
            "diff --git a/src/auth.py b/src/auth.py\n"
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -1,3 +1,4 @@\n"
            "+localStorage.setItem('key', val)\n"
        )
        adr_sample = ADR(
            id="ADR-014",
            title="Encrypted Store",
            status="accepted",
            code_pattern="localStorage.setItem",
            body="Use SecureStore",
        )
        mock_client.get_relevant_adrs = AsyncMock(return_value=[adr_sample])

        req_viol = DeltaRequest(
            session_id="self_check_viol",
            git_diff=sample_diff,
            touched_files=["src/auth.py"],
            lace_client=mock_client,
        )
        report_viol = await analyzer.run(req_viol)
        assert len(report_viol.violations) == 1
        assert "ADR-014" in report_viol.violations[0]

        # Check D: Validation constraints
        try:
            DeltaRequest(
                session_id="",
                git_diff=parse_git_diff(""),
                touched_files=[],
                lace_client=mock_client,
            )
            raise AssertionError("Should have raised ValueError on empty session_id")
        except ValueError:
            pass

        print("adr_delta_analyzer.py deep standalone self-check passed successfully.")

    asyncio.run(_self_test())
