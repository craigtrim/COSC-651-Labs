"""SDK-backed MCP handshake, discovery, and calls with a notebook-safe stderr stream."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from datetime import timedelta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Issue #32: The client contract includes discovery results and captured server stderr.
async def inspect_server(case_id="D01"):
    """Return local MCP tool names, a case diagnostic, a policy, and server stderr.

    The session uses a 20-second read timeout and a temporary file for child stderr.
    """
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-B", "-m", "maintenance_course.mcp_server"],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    # Jupyter's stderr may not expose a Windows file handle. The child needs a real stream.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as error_log:
        async with stdio_client(parameters, errlog=error_log) as (read, write):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=20)
            ) as session:
                await session.initialize()
                listed = await session.list_tools()
                diagnostic = await session.call_tool(
                    "diagnose_case", {"case_id": case_id}
                )
                policy = await session.call_tool(
                    "get_document", {"doc_id": "dates-v2", "app_version": 2}
                )
                result = {
                    "tools": [t.name for t in listed.tools],
                    "diagnostic": json.loads(
                        "".join(getattr(x, "text", "") for x in diagnostic.content)
                    ),
                    "policy": json.loads(
                        "".join(getattr(x, "text", "") for x in policy.content)
                    ),
                }
        error_log.seek(0)
        result["server_log"] = error_log.read()
        return result


if __name__ == "__main__":
    print(json.dumps(asyncio.run(inspect_server()), indent=2))
