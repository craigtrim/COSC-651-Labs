# Issue #32: MCP tool descriptions expose the fixture and document boundaries.
"""A real MCP stdio server. All tools operate on the supplied synthetic documents."""

from mcp.server.fastmcp import FastMCP
from .data import case_by_id
from .tools import ToolRegistry

mcp = FastMCP("LedgerLine course tools")


@mcp.tool()
def get_document(doc_id: str, app_version: int = 2) -> dict:
    """Read one version-matched policy document by its exact document identifier."""
    return ToolRegistry(
        {"id": "mcp", "app_version": app_version, "fixture_id": "valid"}
    ).call("read_document", {"doc_id": doc_id})


@mcp.tool()
def search_documents(query: str, app_version: int = 2) -> dict:
    """Search the small local documentation collection, filtering by application version."""
    return ToolRegistry(
        {"id": "mcp", "app_version": app_version, "fixture_id": "valid"}
    ).call("search_docs", {"query": query})


@mcp.tool()
def diagnose_case(case_id: str) -> dict:
    """Diagnose the fixture associated with a supplied development case; arbitrary paths are not accepted."""
    case = case_by_id(case_id)
    return ToolRegistry(case).call("diagnose_csv", {"fixture_id": case["fixture_id"]})


if __name__ == "__main__":
    mcp.run(transport="stdio")
