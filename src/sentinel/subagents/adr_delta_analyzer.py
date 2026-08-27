"""Subagent B: Architectural Decision Record (ADR) Delta Reasoning Analyzer.

Inspects git diffs against historical Architecture Decision Records in LACE memory,
detects constraint violations, tracks modified ADRs, and drafts new ADRs for novel patterns.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
import tomllib
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.adr import ADR
from sentinel.models.diff import GitDiff
from sentinel.models.review_state import SubagentStatus, SubagentType
from sentinel.session_store import SessionStore

logger = logging.getLogger("sentinel.subagents.adr_delta_analyzer")

# Authoritative standard library module inventory from Python runtime
_STDLIB_MODULES = sys.stdlib_module_names


def _get_project_declared_dependencies() -> set[str]:
    """Extract declared project dependency top-level import names."""
    known = {"sentinel", "daytona", "mcp", "pydantic", "yaml", "pyyaml", "pytest", "ruff"}
    try:
        pyproject_path = Path("pyproject.toml")
        if pyproject_path.exists():
            content = pyproject_path.read_text(encoding="utf-8")
            data = tomllib.loads(content)
            deps = data.get("project", {}).get("dependencies", [])
            dev_deps = data.get("dependency-groups", {}).get("dev", [])
            for dep in deps + dev_deps:
                name = re.split(r"[><=~!;]", dep)[0].strip().lower().replace("-", "_")
                known.add(name)
                if name == "pyyaml":
                    known.add("yaml")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not parse pyproject.toml dependencies: %s", exc)
    return known


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
    """Evaluates semantic code deltas against LACE ADR knowledge and drafts architectural decisions."""

    async def run(
        self,
        request: DeltaRequest,
        session_store: SessionStore | None = None,
    ) -> DeltaReport:
        """Execute delta analysis: fetch ADRs, scan diff, identify violations, and propose drafts."""
        start_time = time.perf_counter()

        try:
            # Short-circuit on empty diff
            if not request.git_diff.files or all(
                not f.added_lines and not f.deleted_lines for f in request.git_diff.files
            ):
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

            # Evaluate constraints, modified ADRs, and novel proposals
            violations = self._evaluate_constraints(request, active_adrs)
            modified_adrs = self._detect_modified_adrs(request, active_adrs)
            proposed_adrs = await self._detect_novel_adrs(request, active_adrs)

            # Build readable summary
            summary_parts: list[str] = []
            if violations:
                summary_parts.append(f"{len(violations)} architectural violation(s) detected.")
            if modified_adrs:
                summary_parts.append(
                    f"{len(modified_adrs)} existing ADR(s) modified/superseded ({', '.join(modified_adrs)})."
                )
            if proposed_adrs:
                summary_parts.append(
                    f"{len(proposed_adrs)} new ADR draft(s) proposed ({', '.join(a.id for a in proposed_adrs)})."
                )
            if not summary_parts:
                summary_parts.append("No architectural changes or violations detected.")

            summary = " ".join(summary_parts)
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
        """Derive search query keywords from touched files and imported modules."""
        effective_files = request.touched_files or [f.path for f in request.git_diff.files]
        terms: list[str] = []

        # 1. Path components
        for path in effective_files:
            parts = path.replace("\\", "/").split("/")
            terms.extend([p for p in parts if p and "." not in p])

        # 2. Extract module names from imports in added lines
        import_pattern = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
        for file_diff in request.git_diff.files:
            for line in file_diff.added_lines:
                match = import_pattern.match(line)
                if match:
                    module_full = match.group(1)
                    module_parts = module_full.split(".")
                    terms.extend([p for p in module_parts if p])

        if not terms:
            terms = ["architecture", "policy", "patterns"]
        return ", ".join(dict.fromkeys(terms))

    async def _fetch_active_adrs(self, request: DeltaRequest, query: str) -> list[ADR]:
        """Fetch ADRs from LACE MCP passing required touched_files with timeout protection."""
        effective_files = request.touched_files or [f.path for f in request.git_diff.files]
        return await asyncio.wait_for(
            request.lace_client.get_relevant_adrs(
                touched_files=effective_files,
                query=query,
            ),
            timeout=float(request.timeout_seconds),
        )

    def _evaluate_constraints(self, request: DeltaRequest, active_adrs: list[ADR]) -> list[str]:
        """Evaluate added lines in git diff against active ADR constraints respecting confidence."""
        violations: list[str] = []
        enforceable_adrs = [
            adr
            for adr in active_adrs
            if adr.status in ("accepted", "proposed")
            and (adr.confidence is None or adr.confidence >= request.confidence_threshold)
        ]

        for file_diff in request.git_diff.files:
            clean_lines = self._clean_code_lines(file_diff.added_lines)

            for line in clean_lines:
                for adr in enforceable_adrs:
                    if adr.code_pattern:
                        pattern = adr.code_pattern.strip()
                        if re.match(r"^[a-zA-Z0-9_]+$", pattern):
                            regex = rf"\b{re.escape(pattern)}\b"
                        else:
                            regex = re.escape(pattern)

                        if re.search(regex, line):
                            violations.append(
                                f"{file_diff.path}: Code matches prohibited pattern in {adr.id} ('{adr.title}')"
                            )
        return violations

    def _detect_modified_adrs(self, request: DeltaRequest, active_adrs: list[ADR]) -> list[str]:
        """Detect existing ADRs that were deleted or modified in diff respecting confidence."""
        modified: list[str] = []
        enforceable_adrs = [
            adr
            for adr in active_adrs
            if (adr.confidence is None or adr.confidence >= request.confidence_threshold)
        ]

        for file_diff in request.git_diff.files:
            clean_deleted = self._clean_code_lines(file_diff.deleted_lines)
            for line in clean_deleted:
                for adr in enforceable_adrs:
                    if adr.code_pattern and adr.id not in modified:
                        pattern = adr.code_pattern.strip()
                        if re.search(re.escape(pattern), line):
                            modified.append(adr.id)
        return modified

    async def _detect_novel_adrs(self, request: DeltaRequest, active_adrs: list[ADR]) -> list[ADR]:
        """Detect novel 3rd-party modules and draft MADR 3.0 records with collision-free IDs."""
        import_pattern = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", re.MULTILINE)
        novel_packages: list[str] = []
        project_deps = _get_project_declared_dependencies()

        for file_diff in request.git_diff.files:
            for line in file_diff.added_lines:
                match = import_pattern.match(line)
                if match:
                    pkg = match.group(1).lower()
                    if (
                        pkg not in _STDLIB_MODULES
                        and pkg not in project_deps
                        and pkg not in novel_packages
                        and not pkg.startswith("sentinel")
                    ):
                        novel_packages.append(pkg)

        # Filter out packages already covered by existing ADRs
        uncovered = [
            pkg
            for pkg in novel_packages
            if not any(
                (adr.code_pattern and pkg in adr.code_pattern.lower())
                or (pkg in adr.title.lower())
                or (any(pkg in tag.lower() for tag in adr.tags))
                for adr in active_adrs
            )
        ]

        if not uncovered:
            return []

        # Derive next available ADR ID across active ADRs and LACE search vault
        occupied_ids: set[int] = set()
        for adr in active_adrs:
            m = re.search(r"ADR-(\d+)", adr.id)
            if m:
                occupied_ids.add(int(m.group(1)))

        if hasattr(request.lace_client, "search_memory"):
            try:
                memories = await request.lace_client.search_memory(
                    query="", category="decision", max_results=100
                )
                for item in memories:
                    text = getattr(item, "raw", "") or getattr(item, "content", "") or ""
                    for m in re.finditer(r"ADR-(\d+)", str(text)):
                        occupied_ids.add(int(m.group(1)))
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "LACE memory search for occupied ADR IDs encountered an error: %s", exc
                )

        highest_id_num = max(occupied_ids, default=0)

        proposed: list[ADR] = []
        for pkg in uncovered:
            highest_id_num += 1
            new_id = f"ADR-{highest_id_num:03d}"
            body_text = (
                f"## Context and Problem Statement\n\n"
                f"The codebase introduces `{pkg}` for project capabilities.\n\n"
                f"## Decision Outcome\n\n"
                f"Chosen option: `{pkg}`, because it provides required architectural functionality.\n\n"
                f"### Positive Consequences\n\n"
                f"- Standardized integration for {pkg}\n\n"
                f"### Negative Consequences\n\n"
                f"- Additional dependency maintenance"
            )
            adr_draft = ADR(
                id=new_id,
                title=f"Adopt {pkg} for Architectural Infrastructure",
                status="draft",
                category="architecture",
                tags=["architecture", "auto-generated", "pivot", pkg],
                constraints=[f"Use {pkg} in accordance with Sentinel guidelines"],
                code_pattern=pkg,
                confidence=1.0,
                body=body_text,
            )
            proposed.append(adr_draft)
        return proposed

    def _clean_code_lines(self, lines: list[str]) -> list[str]:
        """Strip single-line and multi-line comments from added/deleted lines."""
        if not lines:
            return []

        full_text = "\n".join(lines)
        full_text = re.sub(r"/\*[\s\S]*?\*/", "", full_text)
        full_text = re.sub(r'"""[\s\S]*?"""', "", full_text)
        full_text = re.sub(r"'''[\s\S]*?'''", "", full_text)

        result_lines: list[str] = []
        for raw_line in full_text.split("\n"):
            clean = self._strip_comments(raw_line)
            if clean:
                result_lines.append(clean)
        return result_lines

    def _strip_comments(self, line: str) -> str:
        """Strip inline comments (# or //) while preserving quoted string literals and URLs."""
        in_quote: str | None = None
        escape = False
        i = 0
        n = len(line)

        while i < n:
            ch = line[i]
            if escape:
                escape = False
                i += 1
                continue
            if ch == "\\":
                escape = True
                i += 1
                continue
            if in_quote is not None:
                if ch == in_quote:
                    in_quote = None
                i += 1
                continue

            # Outside string literal quotes
            if ch in ("'", '"', "`"):
                in_quote = ch
                i += 1
                continue
            if ch == "#":
                return line[:i].strip()
            if ch == "/" and i + 1 < n and line[i + 1] == "/":
                # Preserve URLs (http:// or https://)
                if i >= 5 and line[i - 5 : i] == "http:":
                    i += 2
                    continue
                if i >= 6 and line[i - 6 : i] == "https:":
                    i += 2
                    continue
                return line[:i].strip()
            i += 1

        return line.strip()

    def _persist_result(
        self,
        request: DeltaRequest,
        report: DeltaReport,
        session_store: SessionStore | None,
        status: SubagentStatus = SubagentStatus.COMPLETED,
        error_payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist result payload to SQLite SessionStore using deterministic task_id."""
        if session_store is not None:
            payload = report.to_dict()
            if error_payload:
                payload.update(error_payload)
            session_store.save_subagent_result(
                session_id=request.session_id,
                subagent_type=SubagentType.ADR_DELTA_ANALYZER,
                status=status,
                result_payload=payload,
                task_id=f"{request.session_id}_adr",
            )


if __name__ == "__main__":
    import tempfile
    from unittest.mock import AsyncMock

    from sentinel.models.diff import parse_git_diff

    # Standalone zero-dependency self-checks (Rule 2903681)
    async def _self_test() -> None:
        with tempfile.TemporaryDirectory():
            mock_client = AsyncMock(spec=LaceMcpClient)
            mock_client.get_relevant_adrs = AsyncMock(return_value=[])

            analyzer = ADRDeltaAnalyzer()

            # 1. Empty diff
            req = DeltaRequest(
                session_id="self_check_sess",
                git_diff=parse_git_diff(""),
                touched_files=[],
                lace_client=mock_client,
            )
            report = await analyzer.run(req)
            assert report.session_id == "self_check_sess"
            assert report.violations == []
            assert report.modified_adrs == []
            assert report.proposed_adrs == []
            assert "No architectural changes" in report.summary

            # 2. Quote-aware inline comment stripping
            assert (
                analyzer._strip_comments("const color = '#ff0000';") == "const color = '#ff0000';"
            )
            assert (
                analyzer._strip_comments("const url = 'https://api.example.com';")
                == "const url = 'https://api.example.com';"
            )
            assert analyzer._strip_comments("// comment only") == ""
            assert analyzer._strip_comments("x = 1 # trailing comment") == "x = 1"
            assert analyzer._strip_comments("value=1# no whitespace comment") == "value=1"
            assert analyzer._strip_comments("foo()// no whitespace comment") == "foo()"

            # 3. Substring collision protection
            diff_coll = parse_git_diff(
                "diff --git a/a.ts b/a.ts\n--- a/a.ts\n+++ b/a.ts\n@@ -1,1 +1,2 @@\n+const localStore = 1;\n"
            )
            adr_coll = ADR(
                id="ADR-014",
                title="Encrypted Store",
                status="accepted",
                code_pattern="localStorage",
                body="",
            )
            mock_client.get_relevant_adrs = AsyncMock(return_value=[adr_coll])
            report_coll = await analyzer.run(
                DeltaRequest(
                    session_id="coll_sess",
                    git_diff=diff_coll,
                    touched_files=["a.ts"],
                    lace_client=mock_client,
                )
            )
            assert len(report_coll.violations) == 0

            # 4. Confidence threshold filtering
            low_conf_adr = ADR(
                id="ADR-099",
                title="Low Confidence Rule",
                status="accepted",
                code_pattern="lowConfFunc",
                confidence=0.5,
                body="",
            )
            mock_client.get_relevant_adrs = AsyncMock(return_value=[low_conf_adr])
            diff_low_conf = parse_git_diff(
                "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,2 @@\n+lowConfFunc()\n"
            )
            report_low_conf = await analyzer.run(
                DeltaRequest(
                    session_id="low_conf_sess",
                    git_diff=diff_low_conf,
                    touched_files=["a.py"],
                    lace_client=mock_client,
                    confidence_threshold=0.8,
                )
            )
            assert len(report_low_conf.violations) == 0

            print("adr_delta_analyzer.py deep standalone self-check passed successfully.")

    asyncio.run(_self_test())
