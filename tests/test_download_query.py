"""download (local-first fetch) + query (local DuckDB), against local fixtures (no network)."""
import json
from pathlib import Path

import pyarrow as pa
import pytest
import pyarrow.parquet as pq

from mudm_mcp import tools
from mudm_mcp.catalog_client import CatalogClient


def _client(site, tmp_path):
    return CatalogClient(site, cache_dir=tmp_path / "cache")


def test_download_defaults_2d_cells_full_res(site, tmp_path):
    r = tools.download(_client(site, tmp_path), "demo2d")
    assert r["status"] == "ok" and r["zoom"] == 1 and r["layers"] == ["cells"]
    assert r["n_files"] == 1 and r["n_skipped"] == 0 and "files" not in r
    leaf = Path(r["path"]) / "zoom=1" / "cells_part_000.mu.parquet"
    src = site / "tiles2d/demo2d/features.parquet/zoom=1/cells_part_000.mu.parquet"
    assert leaf.read_bytes() == src.read_bytes() and r["bytes"] == src.stat().st_size
    assert not (Path(r["path"]) / "zoom=1" / "nuclei_part_000.mu.parquet").exists()


def test_download_explicit_layers_and_zoom(site, tmp_path):
    r = tools.download(_client(site, tmp_path), "demo2d", zoom="max", layers=["cells", "nuclei"])
    assert r["n_files"] == 2 and r["zoom"] == 1
    assert tools.download(_client(site, tmp_path), "demo2d", zoom=0)["zoom"] == 0
    assert tools.download(_client(site, tmp_path), "demo2d", zoom="0")["n_skipped"] == 1   # "0" from a JSON client


def test_download_defaults_3d_coarse_via_viewer_descriptor(site, tmp_path):
    r = tools.download(_client(site, tmp_path), "demo3d")      # features only in the viewer descriptor
    assert r["status"] == "ok" and r["zoom"] == 0 and r["layers"] == [] and r["n_files"] == 1
    assert tools.download(_client(site, tmp_path), "demo3d", zoom="max")["zoom"] == 1


def test_download_over_cap_needs_confirmation(site, tmp_path):
    c = _client(site, tmp_path)
    r = tools.download(c, "demo2d", max_bytes=10)
    assert r["status"] == "needs_confirmation" and r["est_bytes"] > 10 and r["n_files"] == 1
    assert "confirm_large=true" in r["hint"] and "available: 0" in r["hint"]
    assert not (tmp_path / "cache" / "demo2d").exists()        # nothing was fetched
    assert tools.download(c, "demo2d", max_bytes=10, confirm_large=True)["status"] == "ok"


def test_download_3d_gate_points_to_query_facets_first(site, tmp_path):
    r = tools.download(_client(site, tmp_path), "demo3d", max_bytes=10)
    assert r["status"] == "needs_confirmation" and r["hint"].index("query_facets") < r["hint"].index("confirm_large")


def test_download_unknown_size_is_gated(site, tmp_path):
    r = tools.download(_client(site, tmp_path), "big3d")       # 12,000 parts whose sizes cannot be read
    assert r["status"] == "needs_confirmation" and r["est_bytes"] is None and r["n_files"] == 12_000
    assert "unknown size" in r["hint"]


def test_download_is_resumable(site, tmp_path):
    c = _client(site, tmp_path)
    first = tools.download(c, "demo2d")
    again = tools.download(c, "demo2d")
    assert again["n_skipped"] == 1 and again["bytes"] == first["bytes"]


def test_download_rejects_unknown_zoom_and_layer(site, tmp_path):
    c = _client(site, tmp_path)
    with pytest.raises(ValueError, match=r"valid zooms: 0, 1"):
        tools.download(c, "demo2d", zoom=7)
    with pytest.raises(ValueError, match=r"valid zooms: 0, 1"):
        tools.download(c, "demo2d", zoom="finest")
    with pytest.raises(ValueError, match=r"available: cells, nuclei"):
        tools.download(c, "demo2d", layers=["transcripts"])


def test_download_without_features_raises(site, tmp_path):
    (site / "datasets" / "big3d" / "tilejson.json").write_text(json.dumps({"tilejson": "3.0.0", "assets": []}))
    with pytest.raises(ValueError, match="no downloadable features.parquet"):
        tools.download(_client(site, tmp_path), "big3d")


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


def test_query_is_capped_and_sandboxed(site, tmp_path):
    c = _client(site, tmp_path)
    tools.download(c, "demo2d")
    r = tools.query(c, "demo2d", "SELECT feature_id FROM features ORDER BY 1", max_rows=2)
    assert r["rows"] == [[1], [2]] and r["truncated"] is True
    r = tools.query(c, "demo2d", "SELECT tags['cell_type'] AS t, count(DISTINCT feature_id) FROM features GROUP BY 1 ORDER BY 1;")
    assert r["rows"] == [["A", 2], ["B", 2]] and r["truncated"] is False
    with pytest.raises(ValueError, match="single SELECT"):
        tools.query(c, "demo2d", "COPY features TO 'x.csv'")
    with pytest.raises(Exception, match="(?i)permission|disabled"):
        tools.query(c, "demo2d", "SELECT * FROM read_csv('/etc/hosts')")
