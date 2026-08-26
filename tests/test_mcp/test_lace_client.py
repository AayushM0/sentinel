from unittest.mock import AsyncMock, MagicMock

import pytest

from sentinel.mcp.lace_client import LaceMcpClient
from sentinel.models.adr import ADR


@pytest.mark.asyncio
async def test_lace_client_mock_methods():
    client = LaceMcpClient()
    mock_session = AsyncMock()

    # Mock get_relevant_context tool call
    mock_result_context = MagicMock()
    mock_result_context.content = [
        MagicMock(
            text='---\nid: "ADR-014"\ntitle: "Encrypted Storage"\nstatus: "accepted"\n---\nDo not use localStorage.'
        )
    ]

    # Mock call_tool
    async def fake_call_tool(name, args):
        if name == "get_relevant_context":
            return mock_result_context
        elif name == "remember":
            mock_res = MagicMock()
            mock_res.content = [MagicMock(text='{"status": "success", "id": "ADR-015"}')]
            return mock_res
        elif name == "search_memory":
            mock_res = MagicMock()
            mock_res.content = [MagicMock(text='[{"id": "ADR-014", "title": "Encrypted Storage"}]')]
            return mock_res
        elif name == "set_context":
            mock_res = MagicMock()
            mock_res.content = [MagicMock(text='{"status": "ok"}')]
            return mock_res
        raise ValueError(f"Unknown tool {name}")

    mock_session.call_tool = AsyncMock(side_effect=fake_call_tool)
    client._session = mock_session
    client._is_connected = True

    # Test get_relevant_adrs
    adrs = await client.get_relevant_adrs(
        touched_files=["src/auth/session.ts"], query="storage auth"
    )
    assert len(adrs) == 1
    assert adrs[0].id == "ADR-014"
    assert adrs[0].title == "Encrypted Storage"

    # Test commit_adr
    new_adr = ADR(id="ADR-015", title="Query Memoization", body="Use useQueryMemo.")
    ok = await client.commit_adr(new_adr)
    assert ok is True

    # Test search_memory — now returns list[LaceMemoryItem]
    memories = await client.search_memory("storage")
    assert len(memories) == 1
    assert memories[0].id == "ADR-014"

    # Test set_context — now returns LaceContextResponse
    res = await client.set_context(r"D:\agentHarness\sentinel")
    assert res.status == "ok"



@pytest.mark.asyncio
async def test_lace_client_not_connected_raises():
    client = LaceMcpClient()
    with pytest.raises(RuntimeError, match="LACE MCP Client is not connected"):
        await client.get_relevant_adrs(touched_files=["foo.ts"])


@pytest.mark.asyncio
async def test_lace_client_concatenated_adrs():
    client = LaceMcpClient()
    mock_session = AsyncMock()

    concatenated_text = """---
id: "ADR-001"
title: "First Decision"
status: "accepted"
---
Body for first decision.

---
id: "ADR-002"
title: "Second Decision"
status: "accepted"
---
Body for second decision.
"""
    mock_res = MagicMock()
    mock_res.content = [MagicMock(text=concatenated_text)]
    mock_session.call_tool = AsyncMock(return_value=mock_res)
    client._session = mock_session
    client._is_connected = True

    adrs = await client.get_relevant_adrs(touched_files=["src/app.py"])
    assert len(adrs) == 2
    assert adrs[0].id == "ADR-001"
    assert adrs[1].id == "ADR-002"


@pytest.mark.asyncio
async def test_lace_client_commit_error_handling():
    client = LaceMcpClient()
    mock_session = AsyncMock()

    mock_error_res = MagicMock()
    mock_error_res.is_error = True
    mock_error_res.content = [MagicMock(text="Tool execution error: Invalid schema")]
    mock_session.call_tool = AsyncMock(return_value=mock_error_res)
    client._session = mock_session
    client._is_connected = True

    new_adr = ADR(id="ADR-099", title="Error Test", body="Test error handling.")
    ok = await client.commit_adr(new_adr)
    assert ok is False


@pytest.mark.asyncio
async def test_lace_client_adrs_with_thematic_breaks():
    client = LaceMcpClient()
    mock_session = AsyncMock()

    text_with_breaks = """---
id: "ADR-001"
title: "First Decision"
status: "accepted"
---
# Section 1
Here is section 1 text.

---

# Section 2 (thematic break above)
Still ADR-001 body content.

---
id: "ADR-002"
title: "Second Decision"
status: "accepted"
---
ADR 002 Body.
"""
    mock_res = MagicMock()
    mock_res.content = [MagicMock(text=text_with_breaks)]
    mock_session.call_tool = AsyncMock(return_value=mock_res)
    client._session = mock_session
    client._is_connected = True

    adrs = await client.get_relevant_adrs(touched_files=["src/app.py"])
    assert len(adrs) == 2
    assert adrs[0].id == "ADR-001"
    assert "# Section 2" in adrs[0].body
    assert "---" in adrs[0].body
    assert adrs[1].id == "ADR-002"
    assert adrs[1].body == "ADR 002 Body."
