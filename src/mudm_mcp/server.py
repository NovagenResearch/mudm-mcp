"""mudm-mcp — local stdio MCP server. A *client* of the static muDM catalog (default
https://ngrf.mudm.io, produced by ../mudm-data): it discovers, downloads, and **locally** analyzes the
spatial data. It hosts no backend — all fetch/query/decode runs on the user's machine (local-first).

Tools: list_datasets · describe_dataset (WHAT) · get_access (HOW) · download (local-first full-res) ·
query (local DuckDB) · sample_patches / get_lod (tile-as-patch + decoded geometry for ML).

Coordinates are image-pixel [X,Y,Z] in micrometres (NOT lon/lat); features.parquet is muDM BINARY tile
geometry (NOT GeoParquet) — decode via mudm_mcp.geometry.decode_geometry.
"""
from __future__ import annotations

import argparse
import os

from mcp.server.fastmcp import FastMCP

from . import tools
from .catalog_client import CatalogClient

DEFAULT_BASE_URL = "https://ngrf.mudm.io"

mcp = FastMCP("mudm-mcp")
_client: CatalogClient | None = None


def _get_client() -> CatalogClient:
    """The catalog client, configured from MUDM_BASE_URL / MUDM_CACHE_DIR (set by ``main``)."""
    global _client
    if _client is None:
        _client = CatalogClient(os.environ.get("MUDM_BASE_URL", DEFAULT_BASE_URL),
                                os.environ.get("MUDM_CACHE_DIR") or None)
    return _client


@mcp.tool()
def list_datasets(filter: dict | None = None) -> dict:
    """Discover/filter the muDM spatial dataset catalog. ``filter`` accepts ``modality`` (2d/3d),
    ``organism``, ``sex``, ``technique``, ``prep``, ``condition``, ``tissue`` and ``q`` (free-text)."""
    return tools.list_datasets(_get_client(), filter)


@mcp.tool()
def describe_dataset(dataset_id: str) -> dict:
    """Deduce WHAT a dataset contains: modality, assay/organism/tissue, layers, the column schema, the
    coordinate model (image-pixel µm, NOT lon/lat), counts, and the muDM-geometry caveats."""
    return tools.describe_dataset(_get_client(), dataset_id)


@mcp.tool()
def get_access(dataset_id: str) -> dict:
    """Deduce HOW to access a dataset: typed assets (raster/vector/features/tiles3d), the download index
    (features parts/zooms/full_res_zoom), a local DuckDB snippet, and the decode/coordinate notes."""
    return tools.get_access(_get_client(), dataset_id)


@mcp.tool()
def download(dataset_id: str, zoom: str = "max", layers: list[str] | None = None) -> dict:
    """Local-first: fetch a dataset's full-resolution (max-zoom) features.parquet leaves into the local
    cache. Returns the local path + leaf list. (3D/mesh datasets have no parquet and will raise.)"""
    return tools.download(_get_client(), dataset_id, layers=layers, zoom=zoom)


@mcp.tool()
def query(dataset_id: str, sql: str) -> dict:
    """Run DuckDB locally over the DOWNLOADED parquet (view ``features``). Call ``download`` first. Binary
    geometry columns return as ``<binary NB>`` placeholders — use ``sample_patches`` to decode geometry."""
    return tools.query(_get_client(), dataset_id, sql)


@mcp.tool()
def sample_patches(dataset_id: str, n: int = 6, zoom: str = "max", max_features: int = 64) -> dict:
    """Return up to ``n`` spatial PATCHES (tiles) from the downloaded data — the tile-as-patch unit for ML.
    Each patch lists features with decoded geometry (``coords``). Call ``download`` first."""
    return tools.sample_patches(_get_client(), dataset_id, n=n, zoom=zoom, max_features=max_features)


@mcp.tool()
def get_lod(dataset_id: str, zoom: str = "max") -> dict:
    """Summarize one level-of-detail (zoom) of the downloaded data: feature/tile/unique counts + geom-type
    breakdown (the multi-LOD access primitive). Call ``download`` first."""
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
