"""mudm-mcp — local MCP server entry point (SCAFFOLD).

This is a *client* of the static muDM catalog (default https://ngrf.mudm.io, produced by ../mudm-data). It runs
locally under the agent's MCP host (stdio); it hosts no backend and does all fetch/query/analysis on the user's
machine (local-first). The tool contracts below are the TARGET surface; implement them PLAN-DRIVEN (TDD) per
`.claude/plans/active.md` Phase 2. Design of record:
`../mudm-data/.claude/specs/2026-06-23-local-first-agent-showcase-design.md`.

Tools (target):
  list_datasets(filter)            — browse/search the catalog (catalog.json / catalog.parquet)
  describe_dataset(id)             — WHAT it contains: modality, assay, organism, layers, column schema, CRS model
  get_access(id)                   — HOW to access: URLs, the download manifest, the muDM-geometry decode note
  download(id, region=None, zoom="max")  — local-first fetch of FULL-RES tiles once (max-zoom; dedup-aware)
  query(id, sql)                   — DuckDB over the downloaded tiles, locally
  sample_patches(id, ...) / get_lod(id, zoom)  — tile-as-patch / multi-LOD access for ML (and the (b) demo)

NOTE: kept import-light on purpose — it must not import the `mcp` SDK at module load before `uv sync`. The SDK
wiring happens inside main()/the tool registration once Phase 2 begins.
"""
from __future__ import annotations

import argparse

DEFAULT_BASE_URL = "https://ngrf.mudm.io"

# Coordinates are image-pixel [X,Y,Z] in micrometres (NOT lon/lat). features.parquet is muDM BINARY tile geometry
# (NOT GeoParquet), Hive-partitioned by zoom, with the per-feature payload replicated across zooms → read max-zoom
# for full resolution and decode per the muDM geometry doc.

_NOT_IMPLEMENTED = (
    "mudm-mcp is scaffolding only — no tools are implemented yet. Implement plan-driven (TDD) per "
    ".claude/plans/active.md (Phase 1: tool contracts → Phase 2: per-tool implementation)."
)


def main(argv: list[str] | None = None) -> int:
    """Console entry (`mudm-mcp`). Will build + run the stdio MCP server once tools are implemented."""
    ap = argparse.ArgumentParser(prog="mudm-mcp", description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="catalog origin to read (default: %(default)s)")
    ap.parse_args(argv)
    raise SystemExit(_NOT_IMPLEMENTED)


if __name__ == "__main__":
    raise SystemExit(main())
