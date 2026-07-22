"""download (local-first fetch) + query (local DuckDB), against local fixtures (no network)."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mudm_mcp import tools
from mudm_mcp.catalog_client import CatalogClient


def _dl_fixture(tmp_path):
    """A local catalog whose features.parquet has one max-zoom leaf."""
    root = tmp_path / "site"
    leaf = root / "tiles2d" / "demo" / "features.parquet" / "zoom=9"
    leaf.mkdir(parents=True)
    (leaf / "cells_part_000.mu.parquet").write_bytes(b"PARQUET-BYTES")
    (root / "datasets" / "demo").mkdir(parents=True)
    (root / "pyramids.json").write_text(json.dumps({"version": "1.0", "pyramids": [
        {"id": "demo", "kind": "2d", "tilejson": "datasets/demo/tilejson.json",
         "features": "tiles2d/demo/features.parquet/"}]}))
    (root / "datasets" / "demo" / "tilejson.json").write_text(json.dumps({
        "tilejson": "3.0.0", "vector_layers": [],
        "assets": [{"role": "features", "href": "https://ex/tiles2d/demo/features.parquet/",
                    "parts": ["zoom=8/cells_part_000.mu.parquet", "zoom=9/cells_part_000.mu.parquet"],
                    "zooms": [8, 9], "full_res_zoom": 9}]}))
    return root


def test_download_fetches_full_res_only(tmp_path):
    c = CatalogClient(_dl_fixture(tmp_path), cache_dir=tmp_path / "cache")
    r = tools.download(c, "demo")            # zoom defaults to "max" → full_res_zoom 9
    assert r["zoom"] == 9 and r["n_files"] == 1   # only the z9 leaf (z8 exists in the manifest but is skipped)
    out = Path(r["files"][0])
    assert out.read_bytes() == b"PARQUET-BYTES"
    assert out.name == "cells_part_000.mu.parquet" and "zoom=9" in str(out)


def test_download_parquetless_raises(tmp_path):
    root = _dl_fixture(tmp_path)
    # strip the features asset → simulate a 3D/mesh dataset
    (root / "datasets" / "demo" / "tilejson.json").write_text(json.dumps({"tilejson": "3.0.0", "vector_layers": []}))
    c = CatalogClient(root, cache_dir=tmp_path / "cache")
    try:
        tools.download(c, "demo")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "parquet-less" in str(e) or "no downloadable" in str(e)


def _write_mudm_parquet(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = pa.table({
        "tile_x": pa.array([1, 2], pa.uint16()),
        "feature_id": pa.array([10, 20], pa.uint32()),
        "geom_type": pa.array([3, 1], pa.uint8()),
        "positions": pa.array([b"\x00\x00\x80?", b"\x00\x00"], pa.large_binary()),
        "tags": pa.array([[("layer_type", "cells")], [("layer_type", "transcripts")]],
                         pa.map_(pa.string(), pa.string())),
        "zoom": pa.array([9, 9], pa.int32()),
    })
    pq.write_table(t, path)


def test_query_over_downloaded_parquet(tmp_path):
    c = CatalogClient(tmp_path / "site", cache_dir=tmp_path / "cache")
    _write_mudm_parquet(c.cache_dir / "demo" / "features" / "zoom=9" / "cells_part_000.mu.parquet")
    r = tools.query(c, "demo", "SELECT count(*) AS n, max(zoom) AS z FROM features")
    assert r["rows"][0] == [2, 9]
    # tags map stays usable; binary geometry is placeholdered (JSON-safe)
    r2 = tools.query(c, "demo", "SELECT feature_id, positions FROM features ORDER BY feature_id")
    assert r2["rows"][0][0] == 10
    assert str(r2["rows"][0][1]).startswith("<binary")


def test_query_without_download_raises(tmp_path):
    c = CatalogClient(tmp_path / "site", cache_dir=tmp_path / "cache")
    try:
        tools.query(c, "demo", "SELECT 1")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "download" in str(e)
