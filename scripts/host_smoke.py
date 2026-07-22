#!/usr/bin/env python
"""MCP-host smoke: spawn the mudm-mcp stdio server and drive every tool through the real MCP protocol
(initialize → tools/list → tools/call), against the live catalog (https://ngrf.mudm.io by default).

This is what an MCP host (Claude Desktop/Code, Cursor, …) does — it exercises the stdio transport, JSON-RPC,
schema validation and tool dispatch, not just the in-process Python wrappers.

    uv run python scripts/host_smoke.py
"""
import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Spawn the installed console entry directly (no `uv run` wrapper → clean stdio, no warning on the pipe).
SERVER = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "mudm-mcp")


def _unwrap(result):
    """Pull the tool's dict out of a CallToolResult (structuredContent, else JSON in a text block)."""
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        return sc.get("result", sc) if set(sc) == {"result"} else sc
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except Exception:
                return text
    return None


async def main() -> None:
    params = StdioServerParameters(command=SERVER, args=[], env=os.environ.copy())
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            info = await s.initialize()
            print("INITIALIZED:", info.serverInfo.name, info.serverInfo.version)
            names = sorted(t.name for t in (await s.list_tools()).tools)
            print("TOOLS (%d): %s" % (len(names), names))

            ld = _unwrap(await s.call_tool("list_datasets", {"filter": {"technique": "Xenium"}}))
            print("list_datasets(technique=Xenium) -> count =", ld["count"])

            d = _unwrap(await s.call_tool("describe_dataset", {"dataset_id": "xenium-kidney"}))
            print("describe_dataset(xenium-kidney) -> %s / %s | %d columns" %
                  (d["organism"], d["technique"], len(d["columns"])))

            a = _unwrap(await s.call_tool("get_access", {"dataset_id": "xenium-kidney"}))
            print("get_access(xenium-kidney) -> full_res_zoom=%s n_parts=%s" %
                  (a["download"]["full_res_zoom"], a["download"]["n_parts"]))

            dl = _unwrap(await s.call_tool("download", {"dataset_id": "xenium-mouse-ileum"}))
            print("download(xenium-mouse-ileum) -> %d leaves, %d bytes, zoom=%s" %
                  (dl["n_files"], dl["bytes"], dl["zoom"]))

            q = _unwrap(await s.call_tool(
                "query", {"dataset_id": "xenium-mouse-ileum",
                          "sql": "SELECT count(*) AS n, max(zoom) AS z FROM features"}))
            print("query(count, max zoom) -> ", q["rows"])

            sp = _unwrap(await s.call_tool(
                "sample_patches", {"dataset_id": "xenium-mouse-ileum", "n": 1, "max_features": 1}))
            p = sp["patches"][0]
            f = p["features"][0]
            print("sample_patches -> tile (%d,%d) %d feats; f0 geom=%s tags=%s coords[:1]=%s" %
                  (p["tile_x"], p["tile_y"], p["n_features"], f["geom_type"], f.get("tags"), f["coords"][:1]))
    print("\nHOST SMOKE OK — every tool drove end-to-end through the MCP stdio protocol against the live catalog.")


if __name__ == "__main__":
    asyncio.run(main())
