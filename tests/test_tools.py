"""search_datasets / describe_dataset / get_access over the local `site` fixture (no network)."""
import json

from mudm_mcp import tools
from mudm_mcp.catalog_client import CatalogClient


# --- search_datasets -------------------------------------------------------------------------------
def test_search_without_arguments_lists_everything_featured_first(site):
    r = tools.search_datasets(CatalogClient(site))
    assert r["count"] == 3 and r["returned"] == 3 and r["offset"] == 0
    assert [d["id"] for d in r["datasets"]] == ["demo2d", "big3d", "demo3d"]
    assert set(r["datasets"][0]) == {"id", "label", "kind", "organism", "technique", "tissue", "feature_count"}


def test_search_tokens_are_anded_case_insensitive(site):
    c = CatalogClient(site)
    assert [d["id"] for d in tools.search_datasets(c, q="MOUSE brain")["datasets"]] == ["demo2d"]
    assert tools.search_datasets(c, q="mouse kidney")["count"] == 0


def test_search_matches_common_organism_names(site):
    c = CatalogClient(site)          # the catalog says "Mus musculus" / "Drosophila melanogaster"
    assert tools.search_datasets(c, q="fly connectome")["count"] == 2
    assert tools.search_datasets(c, q="human")["count"] == 0


def test_search_filters_are_case_insensitive_substrings(site):
    c = CatalogClient(site)
    assert tools.search_datasets(c, technique="xenium")["count"] == 1
    assert tools.search_datasets(c, modality="3d")["count"] == 2
    assert tools.search_datasets(c, organism="drosophila", tissue="cns")["count"] == 2
    assert tools.search_datasets(c, q="demo", modality="3D")["datasets"][0]["id"] == "demo3d"


def test_search_limit_caps_pages_and_says_so(site):
    c = CatalogClient(site)
    r = tools.search_datasets(c, limit=1)
    assert r["count"] == 3 and r["returned"] == 1 and "2 more match" in r["note"] and "offset=1" in r["note"]
    page2 = tools.search_datasets(c, limit=1, offset=1)
    assert [d["id"] for d in page2["datasets"]] == ["big3d"] and page2["offset"] == 1
    assert tools.search_datasets(c, limit=1, offset=2).get("note") is None
    assert tools.search_datasets(c, offset=99)["returned"] == 0
    assert tools.search_datasets(c, limit=10**6)["returned"] == 3          # clamped, not an error


def test_search_organism_filter_accepts_common_names(site):
    c = CatalogClient(site)          # catalog stores "Mus musculus" / "Drosophila melanogaster"
    assert tools.search_datasets(c, organism="mouse")["count"] == 1
    assert tools.search_datasets(c, organism="Mus")["count"] == 1
    assert tools.search_datasets(c, organism="fly", q="connectome")["count"] == 2
    assert tools.search_datasets(c, organism="human")["count"] == 0


def test_search_overview_and_empty_result_help(site):
    c = CatalogClient(site)
    r = tools.search_datasets(c)
    assert r["overview"]["modality"] == {"3d": 2, "2d": 1}
    assert r["overview"]["technique"] == {"EM connectomics": 2, "Xenium": 1}
    assert r["overview"]["organism"] == {"Drosophila melanogaster": 2, "Mus musculus": 1}
    empty = tools.search_datasets(c, q="zebrafish retina")
    assert empty["count"] == 0 and "fewer" in empty["note"] and "overview" not in empty
    assert empty["catalog_overview"]["tissue"] == {"Brain / CNS": 2, "Brain": 1}


def test_search_sort_by_size(site):
    c = CatalogClient(site)
    assert [d["id"] for d in tools.search_datasets(c, sort="size")["datasets"]] == ["big3d", "demo3d", "demo2d"]
    import pytest
    with pytest.raises(ValueError, match="sort must be"):
        tools.search_datasets(c, sort="newest")


# --- describe_dataset ------------------------------------------------------------------------------
def test_describe_reports_units_columns_facets_lod(site):
    d = tools.describe_dataset(CatalogClient(site), "demo2d")
    assert d["kind"] == "2d" and d["organism"] == "Mus musculus" and d["technique"] == "Xenium"
    assert d["units"]["vertex_unit"] == "pixel" and d["units"]["um_per_unit"] == 0.2125
    assert d["bounds_um"] == [0, 0, 500, 500]
    assert [c["name"] for c in d["parquet_columns"]] == [
        "tile_x", "tile_y", "feature_id", "geom_type", "positions", "indices", "ring_lengths", "tags"]
    assert d["tag_keys"] == ["cell_type"]
    assert d["facets"] == [{"name": "expression", "about": "gene", "layout": "long", "key": "cell_id",
                            "n_columns": 3, "columns": ["cell_id", "gene", "count"]}]
    assert [l["zoom"] for l in d["lod"]] == [0, 1] and [l["n_parts"] for l in d["lod"]] == [1, 2]
    assert all(l["est_bytes"] > 0 for l in d["lod"]) and d["full_res_zoom"] == 1
    assert d["layers"] == [{"id": "cells", "geometry": "polygon"}, {"id": "nuclei", "geometry": "polygon"}]
    assert any("multiply by 0.2125" in c for c in d["caveats"])


def test_describe_attributes_say_where_they_live_and_are_bounded(site):
    d = tools.describe_dataset(CatalogClient(site), "demo2d")
    by = {a["name"]: a for a in d["attributes"]}
    assert set(by) == {"cell_type", "gene"}            # real parquet columns and `zoom` are not attributes
    assert by["cell_type"]["in"] == "tags" and by["cell_type"]["examples"] == ["A", "B"]
    assert by["gene"]["in"] == "facet:expression"
    assert by["gene"]["n_values"] == 5000 and len(by["gene"]["examples"]) == 8
    assert len(json.dumps(d)) < 20_000


def test_describe_3d_uses_viewer_fallback_and_flags_mismatched_names(site):
    d = tools.describe_dataset(CatalogClient(site), "demo3d")
    assert d["units"]["vertex_unit"] == "nanometre" and d["bounds_um"] is None
    assert [l["zoom"] for l in d["lod"]] == [0, 1]
    assert "tile_d" in {c["name"] for c in d["parquet_columns"]} and d["tag_keys"] == ["cellType"]
    by = {a["name"]: a for a in d["attributes"]}
    assert by["cell_type"]["in"] == "facet:properties"      # the descriptor's name is the FACET column
    assert by["pre"]["range"] == [10, 50]
    assert d["layers"] == [{"id": "default", "geometry": "mesh"}]


def test_describe_survives_unreadable_parts(site):
    d = tools.describe_dataset(CatalogClient(site), "big3d")     # 60,000 part names, no files
    assert d["parquet_columns"] is None and any("footer" in c for c in d["caveats"])
    assert [l["n_parts"] for l in d["lod"]] == [12_000] * 5 and d["lod"][0]["est_bytes"] is None
    assert d["attributes"][0]["n_values"] == 6000 and len(json.dumps(d)) < 20_000


# --- get_access -------------------------------------------------------------------------------------
def test_get_access_2d(site):
    a = tools.get_access(CatalogClient(site), "demo2d")
    f = a["features"]
    assert f["base_url"] == f"{site}/tiles2d/demo2d/features.parquet"
    assert f["url_template"].endswith("/zoom={zoom}/{part_file}") and f["full_res_zoom"] == 1
    assert [(z["zoom"], z["n_parts"]) for z in f["zooms"]] == [(0, 1), (1, 2)]
    assert f["layers"] == ["cells", "nuclei"] and f["example_parts"] == ["zoom=0/cells_part_000.mu.parquet"]
    assert a["facets"][0]["url"].endswith("facets/expression.parquet")
    assert a["raster"][0]["max_zoom"] == 1 and a["units"]["vertex_unit"] == "pixel"
    assert "reshape(-1, 2) * 0.2125" in a["python_snippet"] and "decode_geometry" in a["decode_note"]


def test_get_access_never_lists_parts(site):
    a = tools.get_access(CatalogClient(site), "big3d")
    assert len(json.dumps(a)) < 20_000 and "parts" not in a["features"] and "assets" not in a
    assert sum(z["n_parts"] for z in a["features"]["zooms"]) == 60_000
    assert len(a["features"]["example_parts"]) == 3 and "tilejson.json" in a["features"]["part_list"]
    assert a["tiles3d"]["tileset_url"].endswith("tileset.json")


def test_get_access_3d_via_viewer_descriptor(site):
    a = tools.get_access(CatalogClient(site), "demo3d")
    assert a["features"]["part_list"].endswith("tiles/demo3d/tilejson.json")     # where the parts really are
    assert a["facets"][0]["url"].endswith("tiles/demo3d/facets/properties.parquet")
    assert "reshape(-1, 3) * 0.001" in a["python_snippet"]
