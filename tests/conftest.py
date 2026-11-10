"""Shared offline fixtures: a small local catalog (`site`) shaped like the live one.

Datasets:
  demo2d  Xenium-like 2D. Vertices in PIXELS (scale 0.2125 um/px). Layers cells + nuclei, zooms 0..1.
          zoom 0 holds each of the 4 cells exactly once (as live: the coarsest zoom has every feature);
          zoom 1 holds cell 3 split across two tiles (5 rows, 4 ids). Long-layout gene facet.
  demo3d  EM-connectome-like 3D (nanometres). The agent descriptor LACKS the features asset; the viewer
          descriptor (tiles/demo3d/tilejson.json) has it with relative hrefs. Wide properties facet.
  big3d   3D with 60,000 part names (no files) — for response-size tests.
"""
import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

TAGS = pa.map_(pa.string(), pa.string())


def _rect(x0, y0, x1, y1) -> bytes:
    return struct.pack("<8f", x0, y0, x1, y0, x1, y1, x0, y1)


def write_2d_leaf(path: Path, rows: list[tuple]) -> None:
    """rows: (tile_x, tile_y, feature_id, positions_bytes, n_vertices, tags_dict). Polygons, live 2D schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "tile_x": pa.array([r[0] for r in rows], pa.uint16()),
        "tile_y": pa.array([r[1] for r in rows], pa.uint16()),
        "feature_id": pa.array([r[2] for r in rows], pa.uint32()),
        "geom_type": pa.array([3] * len(rows), pa.uint8()),
        "positions": pa.array([r[3] for r in rows], pa.large_binary()),
        "indices": pa.array([b""] * len(rows), pa.large_binary()),
        "ring_lengths": pa.array([[r[4]] for r in rows], pa.list_(pa.uint32())),
        "tags": pa.array([list(r[5].items()) for r in rows], TAGS),
    }), path)


def write_3d_leaf(path: Path, rows: list[tuple]) -> None:
    """rows: (feature_id, [x,y,z]*3 vertex floats, tags_dict). One-triangle meshes, live 3D schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    pq.write_table(pa.table({
        "tile_x": pa.array([0] * n, pa.uint16()), "tile_y": pa.array([0] * n, pa.uint16()),
        "tile_d": pa.array([0] * n, pa.uint16()),
        "feature_id": pa.array([r[0] for r in rows], pa.uint32()),
        "geom_type": pa.array([7] * n, pa.uint8()),
        "positions": pa.array([struct.pack("<9f", *r[1]) for r in rows], pa.large_binary()),
        "indices": pa.array([struct.pack("<3I", 0, 1, 2)] * n, pa.large_binary()),
        "tags": pa.array([list(r[2].items()) for r in rows], TAGS),
    }), path)


def _square(cx, cy, half=5):
    return _rect(cx - half, cy - half, cx + half, cy + half)


CELLS_PX = {1: (100, 100), 2: (200, 100), 3: (1000, 1000), 4: (2000, 2000)}  # centroids, pixels
CELL_TYPE = {1: "A", 2: "A", 3: "B", 4: "B"}


def build_site(root: Path) -> Path:
    base = str(root)
    # ---- demo2d -------------------------------------------------------------------------------
    f2 = root / "tiles2d" / "demo2d" / "features.parquet"
    write_2d_leaf(f2 / "zoom=0" / "cells_part_000.mu.parquet",
                  [(0, 0, i, _square(*CELLS_PX[i]), 4, {"cell_type": CELL_TYPE[i]}) for i in CELLS_PX])
    z1 = [(tx, ty, i, _square(*CELLS_PX[i]), 4, {"cell_type": CELL_TYPE[i]})
          for i, (tx, ty) in ((1, (0, 0)), (2, (0, 0)), (4, (1, 1)))]
    z1 += [(0, 0, 3, _rect(995, 995, 1000, 1005), 4, {"cell_type": "B"}),   # cell 3 split over 2 tiles
           (1, 0, 3, _rect(1000, 995, 1005, 1005), 4, {"cell_type": "B"})]
    write_2d_leaf(f2 / "zoom=1" / "cells_part_000.mu.parquet", z1)
    write_2d_leaf(f2 / "zoom=1" / "nuclei_part_000.mu.parquet",
                  [(0, 0, i, _square(*CELLS_PX[i], half=2), 4, {}) for i in CELLS_PX])
    fac2 = root / "tiles2d" / "demo2d" / "facets" / "expression.parquet"
    fac2.parent.mkdir(parents=True)
    pq.write_table(pa.table({"cell_id": [1, 1, 2, 3, 3, 4],
                             "gene": ["Gad1", "Slc17a7", "Gad1", "Gad1", "Slc17a7", "Slc17a7"],
                             "count": [5, 1, 3, 2, 7, 9]}), fac2)
    parts2 = ["zoom=0/cells_part_000.mu.parquet", "zoom=1/cells_part_000.mu.parquet",
              "zoom=1/nuclei_part_000.mu.parquet"]
    d2 = {
        "tilejson": "3.0.0", "name": "Demo mouse brain", "minzoom": 0, "maxzoom": 1,
        "mudm:kind": "2d", "mudm:platform": "xenium",
        "mudm:coordinate_space": {"space": "image-pixel-XYZ", "units": "micrometre"},
        "mudm:bounds_um": [0, 0, 500, 500],
        "vector_layers": [{"id": "cells", "description": "polygon",
                           "fields": {"tile_x": "integer", "tile_y": "integer", "feature_id": "integer",
                                      "geom_type": "text", "positions": "text", "indices": "text",
                                      "ring_lengths": "text", "tags": "text", "zoom": "integer",
                                      "cell_type": "string", "gene": "vector<int>"},
                           "fieldenums": {"gene": [f"Gene{i}" for i in range(5000)],
                                          "cell_type": ["A", "B"]}}],
        "multiscale": {"axes": [{"name": "x", "type": "space", "unit": "micrometer"},
                                {"name": "y", "type": "space", "unit": "micrometer"}],
                       "coordinateTransformations": [{"type": "scale", "scale": [0.2125, 0.2125]}]},
        "assets": [
            {"role": "raster", "href": f"{base}/tiles2d/demo2d/raster/{{z}}/{{x}}/{{y}}.png",
             "min_zoom": 0, "max_zoom": 1},
            {"role": "features", "href": f"{base}/tiles2d/demo2d/features.parquet/",
             "parts": parts2, "zooms": [0, 1], "full_res_zoom": 1},
            {"role": "facets", "href": f"{base}/tiles2d/demo2d/facets/expression.parquet", "facet": "gene",
             "layout": "long", "key": "cell_id", "columns": ["cell_id", "gene", "count"], "layer": "cells"}]}
    # ---- demo3d (features only in the viewer descriptor) --------------------------------------
    f3 = root / "tiles" / "demo3d" / "features.parquet"
    neurons = [(i, [1000 * i, 0, 0, 1000 * i + 300, 0, 0, 1000 * i, 300, 300],
                {"cellType": ct}) for i, ct in enumerate(["KCg-m", "KCg-m", "KCab-s", "MBON01", "DN1a"], 1)]
    write_3d_leaf(f3 / "zoom=0" / "part_000.mu.parquet", neurons)
    write_3d_leaf(f3 / "zoom=1" / "part_000.mu.parquet", neurons)
    fac3 = root / "tiles" / "demo3d" / "facets" / "properties.parquet"
    fac3.parent.mkdir(parents=True)
    pq.write_table(pa.table({"id": [1, 2, 3, 4, 5],
                             "cell_type": ["KCg-m", "KCg-m", "KCab-s", "MBON01", "DN1a"],
                             "pre": [10, 20, 30, 40, 50], "post": [1, 2, 3, 4, 5]}), fac3)
    layers3 = [{"id": "default", "description": "mesh",
                "fields": {"cell_type": "string", "brain_regions": "string", "pre": "number", "post": "number"},
                "fieldranges": {"pre": [10, 50], "post": [1, 5]},
                "fieldenums": {"cell_type": ["DN1a", "KCab-s", "KCg-m", "MBON01"]}}]
    facet3 = {"role": "facets", "key": "id", "layout": "wide",
              "columns": ["id", "brain_regions", "cell_type", "pre", "post"]}   # brain_regions: advertised, absent
    d3_agent = {"tilejson": "3.0.0", "name": "Demo connectome", "minzoom": 0, "maxzoom": 1,
                "mudm:kind": "3d", "mudm:coordinate_space": {"space": "image-pixel-XYZ", "meters_per_unit": None},
                "vector_layers": layers3,
                "assets": [{"role": "tiles3d", "href": f"{base}/tiles/demo3d/tileset.json"},
                           {**facet3, "href": f"{base}/tiles/demo3d/facets/properties.parquet"}]}
    d3_viewer = {"tilejson": "3.0.0", "minzoom": 0, "maxzoom": 1, "vector_layers": layers3,
                 "assets": [{**facet3, "href": "facets/properties.parquet"},
                            {"role": "features", "href": "features.parquet/",
                             "parts": ["zoom=0/part_000.mu.parquet", "zoom=1/part_000.mu.parquet"],
                             "zooms": [0, 1], "full_res_zoom": 1}]}
    # ---- big3d (names only) -------------------------------------------------------------------
    big_parts = [f"zoom={z}/part_{i:05d}.mu.parquet" for z in range(5) for i in range(12_000)]
    d_big = {"tilejson": "3.0.0", "name": "Big connectome", "minzoom": 0, "maxzoom": 4, "mudm:kind": "3d",
             "vector_layers": [{"id": "default", "description": "mesh", "fields": {"cell_type": "string"},
                                "fieldenums": {"cell_type": [f"type{i}" for i in range(6000)]}}],
             "assets": [{"role": "tiles3d", "href": f"{base}/tiles/big3d/tileset.json"},
                        {"role": "features", "href": f"{base}/tiles/big3d/features.parquet/",
                         "parts": big_parts, "zooms": [0, 1, 2, 3, 4], "full_res_zoom": 4}]}

    for ds, desc in (("demo2d", d2), ("demo3d", d3_agent), ("big3d", d_big)):
        (root / "datasets" / ds).mkdir(parents=True, exist_ok=True)
        (root / "datasets" / ds / "tilejson.json").write_text(json.dumps(desc))
    (root / "tiles" / "demo3d" / "tilejson.json").write_text(json.dumps(d3_viewer))
    (root / "pyramids.json").write_text(json.dumps({"version": "1.0", "pyramids": [
        {"id": "demo2d", "label": "Demo mouse brain", "kind": "2d", "tilejson": "datasets/demo2d/tilejson.json",
         "features": "tiles2d/demo2d/features.parquet/", "feature_count": 4, "featured": True,
         "summary": "A demo coronal section.", "license": "CC-BY-4.0", "tags": ["transcriptomics"],
         "organism": "Mus musculus", "sex": "Unspecified", "technique": "Xenium", "prep": "FFPE",
         "condition": "n/a", "tissue": "Brain", "citation": "Doe 2024"},
        {"id": "demo3d", "label": "Demo fly connectome", "kind": "3d", "tilejson": "datasets/demo3d/tilejson.json",
         "features": "tiles/demo3d/features.parquet/", "feature_count": 5,
         "organism": "Drosophila melanogaster", "technique": "EM connectomics", "tissue": "Brain / CNS"},
        {"id": "big3d", "label": "Big fly connectome", "kind": "3d", "tilejson": "datasets/big3d/tilejson.json",
         "features": "tiles/big3d/features.parquet/", "feature_count": 25000,
         "organism": "Drosophila melanogaster", "technique": "EM connectomics", "tissue": "Brain / CNS"}]}))
    return root


@pytest.fixture
def site(tmp_path) -> Path:
    return build_site(tmp_path / "site")


@pytest.fixture
def stub_httpx_get(monkeypatch):
    """``stub_httpx_get(body)`` replaces ``httpx.get``; returns the list of URLs requested so far."""
    def install(body: bytes, fail: bool = False) -> list[str]:
        import httpx
        calls: list[str] = []

        class _Resp:
            content = body

            def raise_for_status(self):
                return None

        def fake_get(url, **_kw):
            calls.append(url)
            if fail:
                raise httpx.ConnectError("offline")
            return _Resp()

        monkeypatch.setattr(httpx, "get", fake_get)
        return calls
    return install
