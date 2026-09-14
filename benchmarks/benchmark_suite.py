import asyncio
import json
import time
from pathlib import Path
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

FIXTURES = {
    "flower": "tests/sample.jpg",
    "defect": "tests/defect_test6.jpg",
    "layout": "tests/layout_two_column_ruled.png"
}

async def run_battery():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "fusion-vision-mcp"]
    )
    
    results = {}
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Server initialized.")

            # Helper to parse JSON or fallback
            def parse_res(res_text):
                try:
                    return json.loads(res_text)
                except json.JSONDecodeError:
                    return {"text": res_text}

            # Test 1: Captioning (flower)
            t0 = time.time()
            res = await session.call_tool("caption", arguments={"src": FIXTURES["flower"]})
            t1 = time.time()
            results["caption_flower"] = {"time": t1-t0, "output": parse_res(res.content[0].text)}
            out_cap = results['caption_flower']['output'].get('caption', results['caption_flower']['output'].get('text', ''))
            print(f"[Caption] Flower -> {out_cap[:50]}...")

            # Test 2: OCR (defect)
            t0 = time.time()
            res = await session.call_tool("ocr", arguments={"src": FIXTURES["defect"]})
            t1 = time.time()
            results["ocr_defect"] = {"time": t1-t0, "output": parse_res(res.content[0].text)}
            print(f"[OCR] Defect -> (Output logged)")

            # Test 3: Object Counting (flower)
            t0 = time.time()
            res = await session.call_tool("count_objects", arguments={"src": FIXTURES["flower"], "object_name": "petal"})
            t1 = time.time()
            results["count_flower"] = {"time": t1-t0, "output": parse_res(res.content[0].text)}
            print(f"[Count] Flower petals -> {results['count_flower']['output'].get('count')} (Ambiguous: {results['count_flower']['output'].get('semantic_ambiguity')})")

            # Test 4: Structured Analysis (defect)
            t0 = time.time()
            res = await session.call_tool("query_image", arguments={"src": FIXTURES["defect"], "question": "What's wrong?", "structured_analysis": True})
            t1 = time.time()
            results["query_defect"] = {"time": t1-t0, "output": parse_res(res.content[0].text)}
            print(f"[Structured Analysis] Defect -> anomalies: {len(results['query_defect']['output'].get('evidence', {}).get('structured_analysis', {}).get('anomalies', []))}")

    # Write out full benchmark results
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_suite_latest.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark complete. Full output saved to {out_path}")

if __name__ == "__main__":
    asyncio.run(run_battery())
