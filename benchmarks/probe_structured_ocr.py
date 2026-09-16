import asyncio
import json

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

FIXTURE = "C:\\AI\\MCP\\FusionVisionMCP\\tests\\defect_test6.jpg"


async def test():
    # Force stdio using fast memory mode to make it quick
    server_params = StdioServerParameters(command="uv", args=["run", "fusion-vision-mcp", "--memory-mode", "0"])
    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        print("Server initialized.")

        res = await session.call_tool(
            "query_image",
            arguments={"src": FIXTURE, "question": "What is wrong with this image?", "structured_analysis": True},
        )

        # The result content is a list of TextContent objects. We can just print their text attribute
        # but wait, the content might be a complex dict returned as JSON string
        if hasattr(res.content[0], "text"):
            try:
                data = json.loads(res.content[0].text)
                print(json.dumps(data, indent=2))
            except Exception:
                print("Raw text:", res.content[0].text)
        else:
            print(res.content)


asyncio.run(test())
