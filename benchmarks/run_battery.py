import os
import json
import asyncio
from pathlib import Path
from fusion_vision_mcp import app_lifespan, MCPServer
from fusion_vision_mcp.cli import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_FLORENCE2_MODEL,
    DEFAULT_GROUNDING_DINO_MODEL,
    DEFAULT_MOONDREAM_MODEL,
    DEFAULT_MOONDREAM_REVISION,
    DEFAULT_SAM2_MODEL,
)
from mcp.server.mcpserver import Context
import mcp.types as types

async def test_battery():
    test_dir = Path("tests")
    images = [f for f in test_dir.iterdir() if f.suffix in ('.png', '.jpg', '.jpeg', '.pdf')]
    
    print(f"Found {len(images)} images to test.")
    
    async with app_lifespan(
        _server=None, # type: ignore
        model_id=DEFAULT_FLORENCE2_MODEL,
        moondream_model_id=DEFAULT_MOONDREAM_MODEL,
        moondream_revision=DEFAULT_MOONDREAM_REVISION,
        sam2_model_id=DEFAULT_SAM2_MODEL,
        aesthetic_model_id=DEFAULT_AESTHETIC_MODEL,
        grounding_dino_model_id=DEFAULT_GROUNDING_DINO_MODEL,
        reasoner_provider="none",
        reasoner_model="llama3",
        idle_timeout=0,
    ) as app:
        
        # We will directly call the tool functions registered on the server.
        # However, it's easier to just call the underlying app functions with the right types,
        # or we can mock the request context.
        class MockReqCtx:
            lifespan_context = app
        class MockCtx:
            request_context = MockReqCtx()
            
        ctx = MockCtx() # type: ignore
        
        # Let's import the tool functions
        from fusion_vision_mcp.__init__ import caption, count_objects, score_aesthetics, detect_objects
        
        print("\n--- CAPTIONING ---")
        for img_path in images:
            try:
                res = caption(ctx, str(img_path))
                print(f"[{img_path.name}] {res[0]}")
            except Exception as e:
                print(f"[{img_path.name}] ERROR: {e}")

        print("\n--- COUNTING (person) ---")
        for img_path in images:
            try:
                res = count_objects(ctx, str(img_path), "person")
                print(f"[{img_path.name}] found {res['count']} person(s)")
            except Exception as e:
                print(f"[{img_path.name}] ERROR: {e}")
                
        print("\n--- AESTHETICS ---")
        for img_path in images:
            try:
                res = score_aesthetics(ctx, str(img_path))
                print(f"[{img_path.name}] Score: {res[0]['score']}")
            except Exception as e:
                print(f"[{img_path.name}] ERROR: {e}")
                
if __name__ == "__main__":
    asyncio.run(test_battery())
