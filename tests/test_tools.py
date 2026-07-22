"""list_datasets / describe_dataset / get_access over a local catalog fixture (no network)."""
import json

from mudm_mcp import tools
from mudm_mcp.catalog_client import CatalogClient


def _fixture(tmp_path):
    root = tmp_path / "site"
    (root / "datasets" / "demo").mkdir(parents=True)
    (root / "pyramids.json").write_text(json.dumps({"version": "1.0", "pyramids": [
        {"id": "demo", "label": "Demo Kidney", "kind": "2d", "tilejson": "datasets/demo/tilejson.json",
         "features": "tiles2d/demo/features.parquet/", "feature_count": 474686, "summary": "A demo kidney.",
         "license": "CC-BY-4.0", "tags": ["transcriptomics"], "organism": "Homo sapiens", "sex": "Unspecified",
         "technique": "Xenium", "prep": "FFPE", "condition": "n/a", "tissue": "Kidney", "citation": "Doe 2024"},
        {"id": "brain", "label": "Fly Brain", "kind": "3d", "tilejson": "datasets/brain/tilejson.json",
         "organism": "Drosophila melanogaster", "technique": "EM connectomics", "tissue": "Brain / CNS"}]}))
    (root / "datasets" / "demo" / "tilejson.json").write_text(json.dumps({
        "tilejson": "3.0.0", "name": "Demo Kidney", "minzoom": 0, "maxzoom": 9, "mudm:kind": "2d",
        "mudm:coordinate_space": {"space": "image-pixel-XYZ", "units": "micrometre"},
        "mudm:bounds_um": [0, 0, 100, 50],
        "vector_layers": [{"id": "cells", "description": "polygon",
                           "fields": {"feature_id": "integer", "geom_type": "text"},
                           "fielddescriptions": {"feature_id": "Stable per-feature id.",
                                                 "geom_type": "muDM geometry type."}}],
        "multiscale": {"axes": [{"name": "x", "type": "space", "unit": "micrometer"}]},
        "assets": [
            {"role": "raster", "href": "https://ex/tiles2d/demo/raster/{z}/{x}/{y}.png", "min_zoom": 0, "max_zoom": 9},
            {"role": "vector", "href": "https://ex/tiles2d/demo/vectors/{z}/{x}/{y}.pbf"},
            {"role": "features", "href": "https://ex/tiles2d/demo/features.parquet/",
             "parts": ["zoom=9/cells_part_000.mu.parquet"], "zooms": [0, 9], "full_res_zoom": 9}]}))
    return root


def test_list_datasets_filters(tmp_path):
    c = CatalogClient(_fixture(tmp_path))
    assert tools.list_datasets(c)["count"] == 2
    assert tools.list_datasets(c, {"modality": "3d"})["datasets"][0]["id"] == "brain"
    assert tools.list_datasets(c, {"technique": "Xenium"})["count"] == 1
    assert tools.list_datasets(c, {"q": "kidney"})["count"] == 1
    assert tools.list_datasets(c, {"organism": "Mus musculus"})["count"] == 0


def test_describe_dataset_what_plus_schema(tmp_path):
    c = CatalogClient(_fixture(tmp_path))
    d = tools.describe_dataset(c, "demo")
    assert d["kind"] == "2d" and d["organism"] == "Homo sapiens" and d["technique"] == "Xenium"
    assert d["coordinate_space"]["units"] == "micrometre"
    assert {col["name"] for col in d["columns"]} >= {"feature_id", "geom_type"}
    assert d["layers"] == [{"id": "cells", "type": "polygon"}]
    assert any("NOT lon/lat" in c2 for c2 in d["caveats"])


def test_get_access_how(tmp_path):
    c = CatalogClient(_fixture(tmp_path))
    a = tools.get_access(c, "demo")
    assert a["download"]["full_res_zoom"] == 9
    assert a["download"]["parts"] == ["zoom=9/cells_part_000.mu.parquet"]
    assert a["download"]["n_parts"] == 1
    assert a["raster"]["max_zoom"] == 9
    assert "decode_geometry" in a["decode_note"]
