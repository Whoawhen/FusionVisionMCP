"""MCP tool definitions, grouped by domain.

`server()` used to hold all eleven inline, which is a large part of why the package
entry point reached 1,900 lines. Each module here registers its own tools against the
server; `register_all` fixes the order they are declared in, which is the order a
client sees them listed.
"""

from mcp.server.mcpserver import MCPServer

from fusion_vision_mcp.tools import aesthetics, detection, text, vqa


def register_all(mcp: MCPServer) -> None:
    """Attach every tool to `mcp`."""
    text.register(mcp)
    detection.register(mcp)
    vqa.register(mcp)
    aesthetics.register(mcp)


__all__ = ["register_all"]
