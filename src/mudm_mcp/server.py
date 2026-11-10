"""mudm-mcp — local stdio MCP server. A *client* of the static muDM catalog (default
https://ngrf.mudm.io): it discovers, downloads, and **locally** analyzes the spatial data. It hosts no
backend — all fetch/query/decode runs on the user's machine (local-first).

features.parquet is muDM BINARY tile geometry (NOT GeoParquet) — decode via
mudm_mcp.geometry.decode_geometry. Stored vertex units differ per dataset.
"""
from __future__ import annotations

import argparse
import functools
import json
import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import __version__, facets, spatial, tools
from .catalog_client import CatalogClient, DatasetNotFound

DEFAULT_BASE_URL = "https://ngrf.mudm.io"

INSTRUCTIONS = """\
muDM is a public catalog of ~100 spatial-omics and connectome datasets (Xenium, CODEX, CosMx, MERSCOPE,
MIBI, Visium; Drosophila EM connectomes; 3D reference anatomy) served as static tiles from ngrf.mudm.io.
This server runs locally and reads that catalog.

Typical flow:
1. search_datasets(q="mouse brain") to find a dataset id.
2. describe_dataset(id) for layers, columns, units, facet stores and per-zoom sizes.
3. query_facets(id, sql) for per-cell / per-neuron attributes (gene counts, markers, cell types). No download.
4. count_features(id, bbox_um) to count cells in a region given in micrometres (2D datasets).
5. download(id) then query / sample_patches / get_lod for local analysis. Downloads over 500 MB need confirm_large=true.

Rules: geometry is muDM binary (not GeoParquet/WKB), so SQL cannot filter on coordinates; use count_features
or sample_patches. Stored vertex units differ per dataset (pixels, micrometres, nanometres or metres);
describe_dataset reports them and the tools that take *_um arguments convert for you. SQL is read-only and
results are capped (max_rows).
"""

# Tools that only read the remote catalog / local cache. `download` writes (possibly large) files to the
# local cache, so it is NOT marked read-only: hosts may auto-approve read-only tools.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True)
WRITES_CACHE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True,
                               open_world_hint=True)

mcp = MCPServer("mudm-mcp", instructions=INSTRUCTIONS, version=__version__,
                website_url="https://ngrf.mudm.io")
_client: CatalogClient | None = None


def tool_errors(fn):
    """Turn expected failures into ``ToolError`` with a message the agent can act on (mcp 2.x hides the
    text of other exceptions from the client). Unexpected exceptions propagate unchanged."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except DatasetNotFound as exc:
            hint = f" Did you mean: {', '.join(exc.suggestions)}?" if exc.suggestions else ""
            raise ToolError(f"Unknown dataset '{exc.dataset_id}'.{hint} "
                            "Use search_datasets to list ids.") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise ToolError(f"{fn.__name__}: {exc}") from exc
        except Exception as exc:
            if type(exc).__module__.split(".")[0] in ("duckdb", "httpx", "_duckdb"):
                raise ToolError(f"{fn.__name__}: {exc}") from exc
            raise
    return wrapper


def _get_client() -> CatalogClient:
    """The catalog client, configured from MUDM_BASE_URL / MUDM_CACHE_DIR (set by ``main``)."""
    global _client
    if _client is None:
        _client = CatalogClient(os.environ.get("MUDM_BASE_URL", DEFAULT_BASE_URL),
                                os.environ.get("MUDM_CACHE_DIR") or None)
    return _client


def tool(annotations: ToolAnnotations):
    """Register ``fn`` as an MCP tool and return the plain (dict-returning, error-mapped) function.

    Over the wire the result is ONE compact JSON text block. The SDK's default for a dict is the same data
    twice (pretty-printed text + structured content), which roughly triples the characters a host has to
    read; the size budget the tools keep (20,000 chars) is measured on this compact form."""
    def register(fn):
        safe = tool_errors(fn)

        @functools.wraps(safe)
        def wire(*args, **kwargs) -> str:
            return json.dumps(safe(*args, **kwargs), separators=(",", ":"), ensure_ascii=False)

        mcp.tool(annotations=annotations, structured_output=False)(wire)
        return safe
    return register


@tool(READ_ONLY)
def search_datasets(q: str = "", modality: str | None = None, organism: str | None = None,
                    technique: str | None = None, tissue: str | None = None, limit: int = 25) -> dict:
    """Find datasets in the muDM catalog. Start here.

    ``q`` is free text; every word must match (case-insensitive) the id, label, summary, organism (Latin
    or common name: mouse, human, fly), technique, tissue, condition or tags. Optional substring filters:
    ``modality`` ("2d" spatial omics / "3d" connectomes + anatomy), ``organism``, ``technique``, ``tissue``.
    Examples: search_datasets(q="mouse brain") · search_datasets(technique="xenium", tissue="breast") ·
    search_datasets(modality="3d"). Returns up to ``limit`` summaries and the total match ``count``."""
    return tools.search_datasets(_get_client(), q=q, modality=modality, organism=organism,
                                 technique=technique, tissue=tissue, limit=limit)


@tool(READ_ONLY)
def describe_dataset(dataset_id: str) -> dict:
    """What a dataset contains: assay/organism/tissue, license + citation, vertex ``units`` (stored unit
    and the factor to micrometres), ``layers``, the real ``parquet_columns``, per-feature ``attributes``
    and where each lives (a key of the ``tags`` map, or a facet column), ``facets`` (queryable with
    query_facets, no download) and per-zoom sizes (``lod``)."""
    return tools.describe_dataset(_get_client(), dataset_id)


@tool(READ_ONLY)
def get_access(dataset_id: str) -> dict:
    """How to read a dataset directly (outside this server): the parquet URL template with per-zoom part
    counts and estimated sizes, facet file URLs, raster / vector / 3D tile endpoints, bulk archives, the
    vertex units and a minimal Python decode snippet. Use describe_dataset first for what the data is."""
    return tools.get_access(_get_client(), dataset_id)


@tool(READ_ONLY)
def query_facets(dataset_id: str, sql: str, facet: str | None = None, max_rows: int = 200) -> dict:
    """Query a dataset's per-cell / per-neuron attribute table with SQL — NO download (the parquet file is
    read in place). Best first tool for "how many / which cell types / which genes" questions.

    Write ONE DuckDB SELECT against the view ``facets``. What the table holds depends on the assay (see
    describe_dataset → ``facets``; DESCRIBE facets also works):
    - connectomes / anatomy, ``properties`` (one row per neuron): id, cell_type, instance, pre, post, …
      SELECT cell_type, count(*) AS n FROM facets WHERE cell_type LIKE 'KC%' GROUP BY 1 ORDER BY n DESC
    - Xenium, ``expression`` (long: cell_id, gene, count):
      SELECT gene, sum(count) AS total FROM facets GROUP BY 1 ORDER BY total DESC LIMIT 20
    - CODEX / CosMx / MERSCOPE / MIBI, ``markers`` (wide: one row per cell, one column per marker / gene).
    The result includes the file's real ``schema``; rows are capped at ``max_rows`` (``truncated``)."""
    return facets.query_facets(_get_client(), dataset_id, sql, facet=facet, max_rows=max_rows)


@tool(WRITES_CACHE)
def count_features(dataset_id: str, bbox_um: list[float] | None = None, layer: str | None = None,
                   group_by: str | None = None, zoom: int | str | None = None,
                   confirm_large: bool = False) -> dict:
    """Count cells (or another layer's features) of a 2D dataset inside a rectangle given in MICROMETRES.

    ``bbox_um`` = [xmin, ymin, xmax, ymax] in µm (omit to count everything); see describe_dataset →
    ``bounds_um`` for the dataset's extent. Stored pixel coordinates are converted for you — never pass
    pixels. A feature counts when its centroid is inside the box. ``layer`` defaults to "cells";
    ``group_by`` is a ``tags`` key such as "cell_type" (see describe_dataset → ``tag_keys``).
    Example: count_features("xenium-mouse-brain-coronal", bbox_um=[2000, 2000, 2500, 2500])
    Downloads the coarsest zoom of that layer on first use (usually a few MB; same 500 MB gate as
    download). Not for 3D datasets: use query_facets there."""
    return spatial.count_features(_get_client(), dataset_id, bbox_um=bbox_um, layer=layer, zoom=zoom,
                                  group_by=group_by, confirm_large=confirm_large)


@tool(WRITES_CACHE)
def download(dataset_id: str, zoom: int | str | None = None, layers: list[str] | None = None,
             confirm_large: bool = False) -> dict:
    """Download one zoom level of a dataset's feature parquet into the local cache, for query /
    sample_patches / get_lod. Often unnecessary: query_facets and count_features need little or no download.

    Defaults: 2D datasets -> full resolution, ``cells`` layer only; 3D datasets (neuron meshes) -> the
    coarsest zoom. ``zoom`` may be a number from describe_dataset's ``lod`` or "max". Downloads estimated
    above 500 MB are NOT started: the result has ``status: "needs_confirmation"`` with the size and cheaper
    options; ask the user before retrying with ``confirm_large=true``. Already-downloaded files are skipped."""
    return tools.download(_get_client(), dataset_id, zoom=zoom, layers=layers, confirm_large=confirm_large)


@tool(READ_ONLY)
def query(dataset_id: str, sql: str, max_rows: int = 200) -> dict:
    """Run ONE read-only SELECT (DuckDB) over a dataset's DOWNLOADED feature parquet, exposed as the view
    ``features``. Call ``download`` first. Columns: tile_x, tile_y, feature_id, geom_type, positions,
    indices, tags, zoom (see describe_dataset). ``tags`` is a MAP: use tags['cell_type'], CAST numbers.
    Geometry (``positions``) is binary and cannot be filtered in SQL: use count_features for regions.
    Example: SELECT tags['cell_type'] AS t, count(DISTINCT feature_id) AS n FROM features GROUP BY 1 ORDER BY n DESC
    Results are capped at ``max_rows`` (default 200, max 5000); ``truncated`` tells you if more exist."""
    return tools.query(_get_client(), dataset_id, sql, max_rows=max_rows)


@tool(READ_ONLY)
def sample_patches(dataset_id: str, n: int = 2, zoom: int | str = "max", max_features: int = 8,
                   include_coords: bool = False) -> dict:
    """Sample spatial PATCHES (tiles) from the downloaded data — the tile-as-patch unit for ML. Call
    ``download`` first. Returns the ``n`` densest tiles, each with up to ``max_features`` features:
    feature_id, geom_type, tags and a geometry summary in micrometres (n_vertices, bbox_um, centroid_um).
    Set ``include_coords=true`` only when you need the vertices themselves (stored units, first 64 each)."""
    return tools.sample_patches(_get_client(), dataset_id, n=n, zoom=zoom, max_features=max_features,
                                include_coords=include_coords)


@tool(READ_ONLY)
def get_lod(dataset_id: str, zoom: int | str = "max") -> dict:
    """Summarize one level-of-detail (zoom) of the DOWNLOADED data: row / tile / unique-feature counts
    and the geometry-type breakdown. Call ``download`` first (for that zoom)."""
    return tools.get_lod(_get_client(), dataset_id, zoom=zoom)


def main(argv: list[str] | None = None) -> int:
    """Console entry (`mudm-mcp`): set the catalog origin + cache, then run the stdio MCP server."""
    ap = argparse.ArgumentParser(prog="mudm-mcp", description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("MUDM_BASE_URL", DEFAULT_BASE_URL),
                    help="catalog origin to read (default: %(default)s)")
    ap.add_argument("--cache-dir", default=os.environ.get("MUDM_CACHE_DIR"),
                    help="local cache/download dir (default: ~/.cache/mudm-mcp/<host>)")
    a = ap.parse_args(argv)
    os.environ["MUDM_BASE_URL"] = a.base_url
    if a.cache_dir:
        os.environ["MUDM_CACHE_DIR"] = a.cache_dir
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
