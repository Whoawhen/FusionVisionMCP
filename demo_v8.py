import asyncio
import json
from pathlib import Path
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def run():
    base_dir = Path("C:/AI/MCP/FusionVisionMCP").resolve()
    
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run", 
            "fusion-vision-mcp", 
            "--reasoner-provider", "ollama",
            "--granite-docling-model", "ibm-granite/granite-vision-3.1-2b-preview"
        ],
    )
    
    print("Booting FusionVisionMCP v0.8.0 via stdio...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Server initialized successfully.\n")
            
            # 1. Structural Analysis (Sprint 10 & 11)
            img = str(base_dir / "tests" / "defect_test6.jpg")
            print(f"=== 1. AI Defect Detection ({img}) ===")
            print("Query: 'Are there any structural defects?' with structured_analysis=True")
            try:
                res = await session.call_tool("query_image", {
                    "src": img,
                    "question": "Are there any structural defects with the limbs?",
                    "structured_analysis": True
                })
                print(res.content[0].text)
            except Exception as e:
                print("Error:", e)

            # 2. OCR Layout (Sprint 6-8)
            img = str(base_dir / "tests" / "layout_two_column_ruled.png")
            print(f"\n=== 2. Granite-Docling OCR ({img}) ===")
            try:
                res = await session.call_tool("ocr", {
                    "src": img,
                })
                print(res.content[0].text)
            except Exception as e:
                print("Error:", e)
                
            # 3. Aesthetics & IQA (Sprint 12 & 13)
            img = str(base_dir / "tests" / "sample.jpg")
            print(f"\n=== 3. Aesthetic Scoring ({img}) ===")
            try:
                res = await session.call_tool("score_aesthetics", {
                    "src": img,
                    "style_context": True
                })
                print(res.content[0].text)
            except Exception as e:
                print("Error:", e)

if __name__ == "__main__":
    asyncio.run(run())
