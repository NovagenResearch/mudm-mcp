"""CatalogClient over a local `file://`-style build dir (no network)."""
import json

import pytest

from mudm_mcp.catalog_client import CatalogClient


def _fixture(tmp_path):
    root = tmp_path / "site"
    (root / "datasets" / "demo").mkdir(parents=True)
    (root / "pyramids.json").write_text(json.dumps({"version": "1.0", "pyramids": [
        {"id": "demo", "label": "Demo", "kind": "2d", "tilejson": "datasets/demo/tilejson.json",
         "organism": "Homo sapiens", "technique": "Xenium"}]}))
    (root / "datasets" / "demo" / "tilejson.json").write_text(json.dumps({
        "tilejson": "3.0.0", "mudm:kind": "2d", "vector_layers": [{"id": "cells"}],
        "assets": [{"role": "features", "href": "https://x/features.parquet/",
                    "parts": ["zoom=0/cells_part_000.mu.parquet"], "zooms": [0], "full_res_zoom": 0}]}))
    (root / "datasets" / "demo" / "croissant.json").write_text(json.dumps({"@type": "sc:Dataset", "name": "demo"}))
    return root


def test_reads_entries_descriptor_croissant(tmp_path):
    c = CatalogClient(_fixture(tmp_path))
    assert [e["id"] for e in c.entries()] == ["demo"]
    e = c.entry("demo")
    assert e["kind"] == "2d" and e["organism"] == "Homo sapiens"
    assert c.descriptor("demo")["mudm:kind"] == "2d"
    assert c.croissant("demo")["name"] == "demo"


def test_features_asset_is_the_download_index(tmp_path):
    c = CatalogClient(_fixture(tmp_path))
    feat = c.features_asset(c.descriptor("demo"))
    assert feat["full_res_zoom"] == 0
    assert feat["parts"] == ["zoom=0/cells_part_000.mu.parquet"]


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(KeyError):
        CatalogClient(_fixture(tmp_path)).entry("nope")
