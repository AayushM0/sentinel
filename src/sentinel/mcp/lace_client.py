from __future__ import annotations

import json
import os
import types
from contextlib import AsyncExitStack
from typing import Any, Self

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from sentinel.models.adr import ADR


def _default_server_config() -> tuple[str, list[str], dict[str, str]]:
    import sys
    from pathlib import Path

    env_lace_path = os.environ.get("LACE_PATH")
    if env_lace_path:
        lace_dir = Path(env_lace_path)
    else:
        candidate = Path(r"D:\Projects\LACE")
        lace_dir = candidate if candidate.exists() else Path.cwd()

    env_lace_python = os.environ.get("LACE_PYTHON")
    if env_lace_python:
        cmd = env_lace_python
    else:
        win_venv = lace_dir / ".venv" / "Scripts" / "python.exe"
        posix_venv = lace_dir / ".venv" / "bin" / "python"
        if win_venv.exists():
            cmd = str(win_venv)
        elif posix_venv.exists():
            cmd = str(posix_venv)
        else:
            cmd = sys.executable or "python"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(lace_dir / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    args = [
        "-u",
        "-c",
        "import asyncio; from lace.mcp.server import run_server; asyncio.run(run_server())",
    ]
    return cmd, args, env


class LaceMcpClient:
    """Stdio MCP client communicating with the LACE Persistent Memory Engine."""

    def __init__(
        self,
        server_command: str | None = None,
        server_args: list[str] | None = None,
        server_env: dict[str, str] | None = None,
    ) -> None:
        def_cmd, def_args, def_env = _default_server_config()
        self.server_command = server_command or def_cmd
        self.server_args = server_args or def_args
        self.server_env = server_env or def_env

        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._is_connected: bool = False

    async def connect(self) -> None:
        """Establish stdio MCP connection to LACE server."""
        if self._is_connected and self._session is not None:
            return

        self._exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
            env=self.server_env,
        )

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        self._is_connected = True

    async def close(self) -> None:
        """Close stdio MCP session and terminate background processes."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None
        self._is_connected = False

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self.close()

    def _ensure_connected(self) -> ClientSession:
        if not self._is_connected or self._session is None:
            raise RuntimeError("LACE MCP Client is not connected. Call connect() first.")
        return self._session

    async def get_relevant_adrs(
        self,
        touched_files: list[str],
        query: str = "",
    ) -> list[ADR]:
        """Retrieve active ADRs matching touched files or context query."""
        session = self._ensure_connected()
        full_query = f"{query} {' '.join(touched_files)}".strip()

        result = await session.call_tool("get_relevant_context", {"query": full_query})

        adrs: list[ADR] = []
        if not result.content:
            return adrs

        for content_item in result.content:
            text = getattr(content_item, "text", "")
            if not text or not text.strip():
                continue

            lines = text.splitlines(keepends=True)
            dash_indices = [i for i, line in enumerate(lines) if line.strip() == "---"]

            # Identify pairs of (start_dash, end_dash) that contain valid MADR frontmatter
            doc_boundaries: list[tuple[int, int, str]] = []
            for i in range(len(dash_indices) - 1):
                start_fm = dash_indices[i]
                end_fm = dash_indices[i + 1]
                fm_text = "".join(lines[start_fm + 1 : end_fm])
                try:
                    data = yaml.safe_load(fm_text)
                    if isinstance(data, dict) and (
                        "title" in data or "id" in data or "status" in data
                    ):
                        doc_boundaries.append((start_fm, end_fm, fm_text))
                except (yaml.YAMLError, TypeError, ValueError):
                    continue

            if len(doc_boundaries) > 1:
                for i, (start_fm, end_fm, fm_text) in enumerate(doc_boundaries):
                    next_doc_start = (
                        doc_boundaries[i + 1][0] if i + 1 < len(doc_boundaries) else len(lines)
                    )
                    body_text = "".join(lines[end_fm + 1 : next_doc_start]).strip()
                    doc_str = f"---\n{fm_text}---\n\n{body_text}"
                    try:
                        adrs.append(ADR.from_markdown(doc_str))
                    except (ValueError, TypeError, yaml.YAMLError):
                        continue
            else:
                try:
                    adrs.append(ADR.from_markdown(text.strip()))
                except (ValueError, TypeError, yaml.YAMLError):
                    continue

        return adrs

    async def commit_adr(self, adr: ADR) -> bool:
        """Commit an approved Architectural Decision Record into the LACE vault."""
        session = self._ensure_connected()
        result = await session.call_tool(
            "remember",
            {
                "title": adr.title,
                "content": adr.to_markdown(),
                "category": "decision",
                "tags": adr.tags,
            },
        )
        if getattr(result, "is_error", False) is True:
            return False
        return bool(result.content)

    async def search_memory(
        self,
        query: str,
        category: str = "decision",
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform semantic search against LACE memory vault."""
        session = self._ensure_connected()
        result = await session.call_tool(
            "search_memory",
            {
                "query": query,
                "category": category,
                "max_results": max_results,
            },
        )
        if not result.content:
            return []

        try:
            return json.loads(result.content[0].text)
        except (json.JSONDecodeError, AttributeError):
            return [{"raw": c.text} for c in result.content]

    async def set_context(
        self,
        working_directory: str,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """Update LACE active workspace directory and project scope."""
        session = self._ensure_connected()
        args: dict[str, Any] = {"working_directory": working_directory}
        if project_name:
            args["project_name"] = project_name

        result = await session.call_tool("set_context", args)
        if not result.content:
            return {}
        try:
            return json.loads(result.content[0].text)
        except (json.JSONDecodeError, AttributeError):
            return {"raw": result.content[0].text}
