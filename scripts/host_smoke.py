#!/usr/bin/env python
"""MCP-host smoke: spawn the mudm-mcp stdio server and drive the two demo flows through the real MCP
protocol (initialize -> tools/list -> tools/call) against the live catalog (https://ngrf.mudm.io).

This is what an MCP host (Claude Code/Desktop, Cursor, ...) does: it exercises the stdio transport,
JSON-RPC, schema validation and tool dispatch, not just the in-process Python functions. It fails
(exit 1) if a call errors unexpectedly, an expected value is off, or any response exceeds the size budget.

    uv run python scripts/host_smoke.py                      # the project's own .venv entry point
    uv run python scripts/host_smoke.py uvx mudm-mcp         # any other launch command, e.g. the PyPI one

Uses a throwaway cache directory unless MUDM_CACHE_DIR is set.
"""
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BUDGET_CHARS = 20_000
EXPECTED_TOOLS = {"search_datasets", "describe_dataset", "get_access", "query_facets", "count_features",
                  "download", "query", "sample_patches", "get_lod"}
DEFAULT_SERVER = [str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "mudm-mcp")]

BRAIN = "xenium-mouse-brain-coronal"
BOX_UM = [2000, 2000, 2500, 2500]

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    if not ok:
        failures.append(what)
        print(f"    !! {what}")


async def call(session: ClientSession, tool: str, args: dict, expect_error: bool = False):
    t0 = time.time()
    res = await session.call_tool(tool, args)
    text = "".join(getattr(b, "text", "") or "" for b in res.content)
    shown = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
    print(f"  {tool}({shown[:110]})  ->  {len(text):,} chars, {time.time() - t0:.1f} s"
          + ("  [error result]" if res.is_error else ""))
    check(len(text) <= BUDGET_CHARS, f"{tool}: response of {len(text):,} chars exceeds {BUDGET_CHARS:,}")
    check(bool(res.is_error) == expect_error, f"{tool}: is_error={res.is_error}, expected {expect_error}: {text[:200]}")
    if res.is_error:
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


async def main(server_cmd: list[str]) -> int:
    env = os.environ.copy()
    env.setdefault("MUDM_CACHE_DIR", tempfile.mkdtemp(prefix="mudm-mcp-smoke-"))
    params = StdioServerParameters(command=server_cmd[0], args=server_cmd[1:], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            info = await s.initialize()
            print(f"server: {info.server_info.name} {info.server_info.version}; "
                  f"instructions: {len(info.instructions or '')} chars; cache: {env['MUDM_CACHE_DIR']}")
            listed = (await s.list_tools()).tools
            names = {t.name for t in listed}
            print(f"tools ({len(names)}): {', '.join(sorted(names))}")
            check(names == EXPECTED_TOOLS, f"tool set differs: {sorted(names ^ EXPECTED_TOOLS)}")
            check(len(json.dumps([t.model_dump(mode='json') for t in listed])) < 30_000, "tool list is very large")

            print("\nFlow A: find a mouse brain spatial transcriptomics dataset, describe it, count cells in a region")
            r = await call(s, "search_datasets", {"q": "mouse brain"})
            check(BRAIN in [d["id"] for d in r["datasets"]], f"{BRAIN} not found by 'mouse brain'")
            r = await call(s, "describe_dataset", {"dataset_id": BRAIN})
            check(r["units"] == {**r["units"], "vertex_unit": "pixel", "um_per_unit": 0.2125}, "unexpected units")
            print(f"    {r['feature_count']:,} cells; units {r['units']['vertex_unit']} x {r['units']['um_per_unit']}; "
                  f"layers {[l['id'] for l in r['layers']]}; facets {[f['name'] for f in r['facets']]}")
            r = await call(s, "count_features", {"dataset_id": BRAIN, "bbox_um": BOX_UM})
            print(f"    {r['count']:,} of {r['total_features']:,} cells in {BOX_UM} um (zoom {r['zoom']})")
            check(r["count"] == 769 and r["total_features"] == 131_889, f"count_features gave {r['count']}/{r['total_features']}")
            r = await call(s, "query_facets", {"dataset_id": BRAIN, "sql":
                           "SELECT gene, sum(count) AS total FROM facets GROUP BY 1 ORDER BY total DESC LIMIT 5"})
            print(f"    top genes: {r['rows']}")

            print("\nFlow B: how many Kenyon cells in the hemibrain?")
            r = await call(s, "search_datasets", {"q": "hemibrain"})
            check(r["datasets"][0]["id"] == "hemibrain", "hemibrain not first")
            r = await call(s, "query_facets", {"dataset_id": "hemibrain", "sql":
                           "SELECT cell_type, count(*) AS n FROM facets WHERE cell_type LIKE 'KC%' GROUP BY 1 ORDER BY n DESC"})
            total = sum(row[1] for row in r["rows"])
            print(f"    {total:,} neurons in {r['n_rows']} KC types; top: {r['rows'][:3]}")
            check(r["rows"][0] == ["KCg-m", 590] and total == 1846, f"Kenyon cells: {r['rows'][:1]} total {total}")

            print("\nAccess + local analysis")
            await call(s, "get_access", {"dataset_id": "hemibrain"})
            r = await call(s, "download", {"dataset_id": "hemibrain"})
            check(r.get("status") == "needs_confirmation", "hemibrain download was not gated")
            print(f"    gated: {r.get('hint', '')[:120]}")
            r = await call(s, "download", {"dataset_id": "xenium-mouse-ileum"})
            check(r.get("status") == "ok", "ileum download failed")
            await call(s, "get_lod", {"dataset_id": "xenium-mouse-ileum"})
            r = await call(s, "query", {"dataset_id": "xenium-mouse-ileum", "sql": "SELECT * FROM features"})
            check(isinstance(r, dict) and "truncated" in r, "query result has no truncated flag")
            await call(s, "sample_patches", {"dataset_id": "xenium-mouse-ileum", "n": 16, "max_features": 64,
                                             "include_coords": True})

            print("\nErrors an agent can act on")
            for tool, args, needle in [
                ("describe_dataset", {"dataset_id": "hemi-brain"}, "Did you mean: hemibrain"),
                ("query_facets", {"dataset_id": "hemibrain", "sql": "SELECT brain_regions FROM facets"}, "Available columns"),
                ("query", {"dataset_id": "xenium-mouse-ileum", "sql": "SELECT * FROM read_csv('/etc/hosts')"}, "disabled"),
                ("count_features", {"dataset_id": "hemibrain", "bbox_um": [0, 0, 1, 1]}, "query_facets"),
            ]:
                text = await call(s, tool, args, expect_error=True)
                check(needle in str(text), f"{tool} error lacks {needle!r}: {str(text)[:160]}")
                print(f"    {str(text)[:150]}")
    if failures:
        print(f"\nHOST SMOKE FAILED ({len(failures)}):\n  - " + "\n  - ".join(failures))
        return 1
    print("\nHOST SMOKE OK: both demo flows ran through the MCP stdio protocol against the live catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:] or DEFAULT_SERVER)))
