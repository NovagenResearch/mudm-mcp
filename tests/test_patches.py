"""get_lod + sample_patches over a synthetic parquet with real (decodable) geometry."""
import struct

import pyarrow as pa
import pyarrow.parquet as pq

from mudm_mcp import tools
from mudm_mcp.catalog_client import CatalogClient


def _write(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    poly = struct.pack("<6f", 0, 0, 10, 0, 5, 10)  # 3-vertex polygon ring
    point = struct.pack("<2f", 3, 4)               # 1 vertex
    t = pa.table({
        "tile_x": pa.array([0, 0], pa.uint16()),
        "tile_y": pa.array([0, 0], pa.uint16()),
        "feature_id": pa.array([1, 2], pa.uint32()),
        "geom_type": pa.array([3, 1], pa.uint8()),
        "positions": pa.array([poly, point], pa.large_binary()),
        "indices": pa.array([b"", b""], pa.large_binary()),
        "ring_lengths": pa.array([[3], []], pa.list_(pa.uint32())),
        "tags": pa.array([[("layer_type", "cells")], [("layer_type", "transcripts")]],
                         pa.map_(pa.string(), pa.string())),
        "zoom": pa.array([9, 9], pa.int32()),
    })
    pq.write_table(t, path)


def test_get_lod_summary(tmp_path):
    c = CatalogClient(tmp_path / "site", cache_dir=tmp_path / "cache")
    _write(c.cache_dir / "demo" / "features" / "zoom=9" / "part.mu.parquet")
    r = tools.get_lod(c, "demo")
    assert r["zoom"] == 9 and r["features"] == 2 and r["tiles"] == 1 and r["unique_features"] == 2
    assert r["geom_types"] == {1: 1, 3: 1}


def test_sample_patches_decodes_geometry(tmp_path):
    c = CatalogClient(tmp_path / "site", cache_dir=tmp_path / "cache")   # no catalog: offline directory use
    _write(c.cache_dir / "demo" / "features" / "zoom=9" / "part.mu.parquet")
    r = tools.sample_patches(c, "demo", dims=2, include_coords=True)
    assert r["n_patches"] == 1 and r["units"]["basis"].startswith("unknown") and r["coords_unit"] == "micrometre"
    p = r["patches"][0]
    assert p["tile_x"] == 0 and p["tile_y"] == 0 and p["n_features"] == 2 and len(p["features"]) == 2
    by_id = {f["feature_id"]: f for f in p["features"]}
    assert by_id[1]["geom_type"] == 3 and by_id[1]["n_rings"] == 1 and len(by_id[1]["coords"]) == 3
    assert by_id[1]["coords"][1] == [10.0, 0.0]
    assert by_id[1]["tags"]["layer_type"] == "cells"
    assert by_id[2]["geom_type"] == 1 and by_id[2]["coords"] == [[3.0, 4.0]]


def _downloaded(site, tmp_path, dataset, **kw):
    c = CatalogClient(site, cache_dir=tmp_path / "cache")
    tools.download(c, dataset, **kw)
    return c


def test_sample_patches_summarises_in_micrometres_by_default(site, tmp_path):
    c = _downloaded(site, tmp_path, "demo2d", zoom=0)
    r = tools.sample_patches(c, "demo2d")
    assert r["units"]["um_per_unit"] == 0.2125 and "coords_unit" not in r
    by_id = {f["feature_id"]: f for f in r["patches"][0]["features"]}
    f1 = by_id[1]                                   # square centred at pixel (100, 100), half-side 5 px
    assert "coords" not in f1 and f1["n_vertices"] == 4 and f1["tags"] == {"cell_type": "A"}
    assert f1["centroid_um"] == [21.25, 21.25]
    assert f1["bbox_um"] == [20.188, 20.188, 22.312, 22.312]


def test_sample_patches_include_coords_are_stored_units(site, tmp_path):
    c = _downloaded(site, tmp_path, "demo2d", zoom=0)
    r = tools.sample_patches(c, "demo2d", include_coords=True)
    f1 = {f["feature_id"]: f for f in r["patches"][0]["features"]}[1]
    assert r["coords_unit"] == "pixel" and f1["coords"][0] == [95.0, 95.0]


def test_sample_patches_3d_nanometres(site, tmp_path):
    c = _downloaded(site, tmp_path, "demo3d")
    f = {x["feature_id"]: x for x in tools.sample_patches(c, "demo3d")["patches"][0]["features"]}[1]
    assert f["n_triangles"] == 1 and f["tags"] == {"cellType": "KCg-m"}
    assert f["bbox_um"] == [1.0, 0.0, 0.0, 1.3, 0.3, 0.3]            # 1000..1300 nm -> 1.0..1.3 um


def test_sample_patches_clamps_and_stays_small(site, tmp_path):
    import json
    c = _downloaded(site, tmp_path, "demo2d", layers=["cells", "nuclei"])
    r = tools.sample_patches(c, "demo2d", n=10**6, max_features=10**6, include_coords=True)
    assert r["n_patches"] <= 16 and len(json.dumps(r)) < 20_000
    assert tools.sample_patches(c, "demo2d", n=0, max_features=0)["patches"][0]["features"].__len__() == 1


def test_sample_patches_3d_schema_no_ring_lengths(tmp_path, monkeypatch):
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from mudm_mcp import tools
    # one mesh feature: 3 verts (x,y,z) + 1 triangle. NO ring_lengths AND NO in-file `zoom` column —
    # zoom lives only in the zoom=4/ path, so _open_features picks hive=true and _resolve_zoom must
    # CAST (the DuckDB-1.5.4 fix); an in-file `zoom` column would mask that regression.
    verts = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype="<f4").tobytes()
    tris = np.array([0, 1, 2], dtype="<u4").tobytes()
    t = pa.table({"tile_x": pa.array([0], pa.uint16()),
                  "tile_y": pa.array([0], pa.uint16()), "tile_d": pa.array([0], pa.uint16()),
                  "feature_id": pa.array([1], pa.uint32()), "geom_type": pa.array([7], pa.uint8()),
                  "positions": pa.array([verts], pa.large_binary()), "indices": pa.array([tris], pa.large_binary()),
                  "tags": pa.array([[("cell_type", "MN")]], pa.map_(pa.utf8(), pa.utf8()))})
    root = tmp_path / "demo" / "features" / "zoom=4"; root.mkdir(parents=True)
    pq.write_table(t, root / "default_part_000.mu.parquet")
    class _C:  # minimal client: descriptor says 3d, features root = tmp
        def descriptor(self, _id): return {"mudm:kind": "3d"}
    r = tools.sample_patches(_C(), "demo", dims=None, downloaded_dir=str(tmp_path / "demo"))
    feat = r["patches"][0]["features"][0]
    assert len(feat["centroid_um"]) == 3         # 3D geometry summary
    assert feat.get("n_triangles") == 1          # mesh decoded
