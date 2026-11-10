"""Every tool response must fit an MCP host's output budget: at most 20,000 characters of JSON, even
for worst-case arguments and for a dataset with 60,000 parts and thousands of enum values."""
import json

import pytest

from mudm_mcp import facets, spatial, tools
from mudm_mcp.catalog_client import CatalogClient

BUDGET = 20_000


@pytest.fixture
def client(site, tmp_path):
    c = CatalogClient(site, cache_dir=tmp_path / "cache")
    tools.download(c, "demo2d", layers=["cells", "nuclei"])
    tools.download(c, "demo2d", zoom=0)
    tools.download(c, "demo3d", zoom="max")
    return c


CALLS = {
    "search_datasets()": lambda c: tools.search_datasets(c),
    "search_datasets(limit=max)": lambda c: tools.search_datasets(c, limit=10**6),
    "describe_dataset(demo2d)": lambda c: tools.describe_dataset(c, "demo2d"),
    "describe_dataset(demo3d)": lambda c: tools.describe_dataset(c, "demo3d"),
    "describe_dataset(big3d)": lambda c: tools.describe_dataset(c, "big3d"),
    "get_access(demo2d)": lambda c: tools.get_access(c, "demo2d"),
    "get_access(demo3d)": lambda c: tools.get_access(c, "demo3d"),
    "get_access(big3d)": lambda c: tools.get_access(c, "big3d"),
    "download(demo2d)": lambda c: tools.download(c, "demo2d"),
    "download(big3d) gated": lambda c: tools.download(c, "big3d", zoom="max"),
    "query SELECT *": lambda c: tools.query(c, "demo2d", "SELECT * FROM features", max_rows=5000),
    "query 3d SELECT *": lambda c: tools.query(c, "demo3d", "SELECT * FROM features"),
    "get_lod": lambda c: tools.get_lod(c, "demo2d"),
    "sample_patches default": lambda c: tools.sample_patches(c, "demo2d"),
    "sample_patches worst": lambda c: tools.sample_patches(c, "demo2d", n=16, max_features=64, include_coords=True),
    "sample_patches 3d worst": lambda c: tools.sample_patches(c, "demo3d", n=16, max_features=64, include_coords=True),
    "query_facets SELECT *": lambda c: facets.query_facets(c, "demo2d", "SELECT * FROM facets", max_rows=5000),
    "query_facets wide": lambda c: facets.query_facets(c, "demo3d", "SELECT * FROM facets"),
    "count_features": lambda c: spatial.count_features(c, "demo2d", bbox_um=[0, 0, 50, 50]),
    "count_features grouped": lambda c: spatial.count_features(c, "demo2d", group_by="cell_type"),
}


@pytest.mark.parametrize("name", CALLS)
def test_response_fits_the_budget(name, client):
    out = CALLS[name](client)
    assert len(json.dumps(out)) <= BUDGET, name


def test_wide_tables_and_many_groups_stay_bounded(tmp_path, site):
    """Synthetic worst cases the small fixture cannot show: a 400-column facet table and 5,000 tag values."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from conftest import _square, write_2d_leaf

    wide = site / "tiles" / "demo3d" / "facets" / "properties.parquet"
    pq.write_table(pa.table({f"marker_{i:03d}": [float(i)] * 300 for i in range(400)}), wide)
    c = CatalogClient(site, cache_dir=tmp_path / "cache")
    r = facets.query_facets(c, "demo3d", "SELECT * FROM facets")
    assert r["n_columns"] == 400 and len(r["schema"]) == 60 and r["truncated"] and len(json.dumps(r)) <= BUDGET

    leaf = site / "tiles2d" / "demo2d" / "features.parquet" / "zoom=0" / "cells_part_000.mu.parquet"
    write_2d_leaf(leaf, [(0, 0, i, _square(10 + i, 10), 4, {"cell_type": f"type-{i:05d}-" + "x" * 40})
                         for i in range(5000)])
    r = spatial.count_features(c, "demo2d", group_by="cell_type")
    assert r["count"] == 5000 and len(r["groups"]) == 50 and "50 largest of 5000" in r["note"]
    assert len(json.dumps(r)) <= BUDGET
    assert len(json.dumps(tools.query(c, "demo2d", "SELECT * FROM features", max_rows=5000))) <= BUDGET
    assert len(json.dumps(tools.sample_patches(c, "demo2d", n=16, max_features=64, include_coords=True))) <= BUDGET
