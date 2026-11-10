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


def test_features_agent_descriptor_preferred(site):
    f = CatalogClient(site).features("demo2d")
    assert f["source"] == "agent" and f["base"] == "tiles2d/demo2d/features.parquet"
    assert f["full_res_zoom"] == 1 and f["zooms"] == [0, 1] and len(f["parts"]) == 3


def test_features_falls_back_to_viewer_descriptor(site):
    f = CatalogClient(site).features("demo3d")
    assert f["source"] == "viewer" and f["base"] == "tiles/demo3d/features.parquet"
    assert f["full_res_zoom"] == 1 and f["parts"][0] == "zoom=0/part_000.mu.parquet"


def test_features_fallback_without_entry_features_field(site):
    doc = json.loads((site / "pyramids.json").read_text())
    for e in doc["pyramids"]:
        e.pop("features", None)
    (site / "pyramids.json").write_text(json.dumps(doc))
    c = CatalogClient(site)
    assert c.features("demo3d")["base"] == "tiles/demo3d/features.parquet"      # probed tiles/<id>/
    assert c.features("demo2d")["base"] == "tiles2d/demo2d/features.parquet"    # stripped from the href


def test_features_none_when_no_descriptor_has_them(tmp_path):
    root = _fixture(tmp_path)
    (root / "datasets" / "demo" / "tilejson.json").write_text(json.dumps({"tilejson": "3.0.0", "assets": []}))
    assert CatalogClient(root).features("demo") is None


def test_features_foreign_href_without_entry_base_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="cannot locate features.parquet"):   # href on another origin
        CatalogClient(_fixture(tmp_path)).features("demo")


def test_estimate_bytes_exact_for_few_parts(site):
    c = CatalogClient(site)
    f = c.features("demo2d")
    parts = [p for p in f["parts"] if p.startswith("zoom=1/")]
    assert c.estimate_bytes(f["base"], parts) == sum((site / f["base"] / p).stat().st_size for p in parts)


def test_estimate_bytes_extrapolates_from_a_sample(site):
    c = CatalogClient(site)
    one = (site / "tiles2d/demo2d/features.parquet/zoom=1/cells_part_000.mu.parquet").stat().st_size
    assert c.estimate_bytes("tiles2d/demo2d/features.parquet", ["zoom=1/cells_part_000.mu.parquet"] * 10) == one * 10


def test_estimate_bytes_none_when_sizes_unavailable(site):
    c = CatalogClient(site)
    f = c.features("big3d")                       # 60,000 names, no files behind them
    assert c.estimate_bytes(f["base"], f["parts"]) is None and c.estimate_bytes(f["base"], []) == 0


def test_url_is_origin_relative(site):
    assert CatalogClient(site).url("tiles", "x/", "a.parquet") == f"{site}/tiles/x/a.parquet"
    assert CatalogClient("https://example.test/").url("a", "b") == "https://example.test/a/b"


def test_json_cache_expires(tmp_path, stub_httpx_get):
    calls = stub_httpx_get(b'{"pyramids": []}')
    fresh = CatalogClient("https://example.test", tmp_path / "a")
    fresh.pyramids(); fresh.pyramids()
    assert len(calls) == 1                         # second read served from the cache
    expiring = CatalogClient("https://example.test", tmp_path / "b", json_ttl_s=0)
    expiring.pyramids(); expiring.pyramids()
    assert len(calls) == 3                         # ttl 0: refetched every time


def test_stale_cache_is_served_when_offline(tmp_path, stub_httpx_get):
    stub_httpx_get(b'{"pyramids": [{"id": "x"}]}')
    c = CatalogClient("https://example.test", tmp_path, json_ttl_s=0)
    c.pyramids()
    stub_httpx_get(b"", fail=True)
    assert c.entries() == [{"id": "x"}]            # expired + offline -> stale copy, not an error


def test_download_file_is_atomic(site, tmp_path, monkeypatch):
    c = CatalogClient(site)
    out = tmp_path / "leaf.parquet"

    def boom(*_a):
        raise OSError("interrupted")
    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError):
        c.download_file(["tiles2d/demo2d/features.parquet", "zoom=1/cells_part_000.mu.parquet"], out)
    assert not out.exists() and not list(tmp_path.glob("leaf.parquet*"))


def test_download_file_copies(site, tmp_path):
    src = site / "tiles2d/demo2d/features.parquet/zoom=1/cells_part_000.mu.parquet"
    n = CatalogClient(site).download_file(["tiles2d/demo2d/features.parquet/", "zoom=1/cells_part_000.mu.parquet"],
                                          tmp_path / "o" / "leaf.parquet")
    assert n == src.stat().st_size and (tmp_path / "o" / "leaf.parquet").read_bytes() == src.read_bytes()


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(KeyError):
        CatalogClient(_fixture(tmp_path)).entry("nope")
