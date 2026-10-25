"""`server_status`: what the server holds in memory, and why a call was slow or refused."""

from typing import Any

from mcp.server.mcpserver import Context, MCPServer

from fusion_vision_mcp.protocols import AppContext


def register(mcp: MCPServer) -> None:
    @mcp.tool()
    def server_status(ctx: Context[AppContext]) -> dict[str, Any]:
        """Report this server's memory state: which models are loaded and what each holds (GB), the
        machine's live free RAM and hard limit (commit on Windows, RAM + swap elsewhere), heavy calls
        running now, and the last loads, releases and refusals with their reasons. Loads no model and
        runs instantly. Call it when a vision call was unexpectedly slow (usually a model being reloaded
        after it was released to make room) or was refused for lack of memory, to see what holds the
        memory before retrying. The server releases idle models on its own when memory runs short, so a
        refusal means everything still loaded was in use at that moment."""
        app = ctx.request_context.lifespan_context
        if app.memory is None:
            return {"managed": False}
        return {"managed": True, **app.memory.status()}
