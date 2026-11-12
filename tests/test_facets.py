"""query_facets over the local `site` fixture; one live check of the demo query."""
import json

import pytest

from mudm_mcp.catalog_client import CatalogClient, DatasetNotFound
from mudm_mcp.facets import query_facets


def test_query_facets_wide_via_agent_descriptor(site):
    r = query_facets(CatalogClient(site), "demo3d", "SELECT count(*) FROM facets WHERE cell_type LIKE 'KC%'")
    assert r["rows"] == [[3]] and r["facet"] == "properties" and r["layout"] == "wide"
    assert [c["name"] for c in r["schema"]] == ["id", "cell_type", "pre", "post"]
    assert r["missing_columns"] == ["brain_regions"]          # advertised by the descriptor, not in the file
    assert r["source"].endswith("tiles/demo3d/facets/properties.parquet")


def test_query_facets_via_viewer_descriptor_relative_href(site):
    agent = site / "datasets" / "demo3d" / "tilejson.json"
    doc = json.loads(agent.read_text())
    doc["assets"] = [a for a in doc["assets"] if a["role"] != "facets"]
    agent.write_text(json.dumps(doc))
    r = query_facets(CatalogClient(site), "demo3d",
                     "SELECT cell_type, count(*) AS n FROM facets GROUP BY 1 ORDER BY n DESC, 1 LIMIT 1")
    assert r["rows"] == [["KCg-m", 2]]


def test_query_facets_long(site):
    r = query_facets(CatalogClient(site), "demo2d",
                     "SELECT gene, sum(count) AS total FROM facets GROUP BY 1 ORDER BY 1;")
    assert r["rows"] == [["Gad1", 10], ["Slc17a7", 17]] and r["facet"] == "expression"
    assert "missing_columns" not in r and r["truncated"] is False


def test_query_facets_is_capped(site):
    r = query_facets(CatalogClient(site), "demo2d", "SELECT * FROM facets", max_rows=4)
    assert r["n_rows"] == 4 and r["truncated"] is True


def test_query_facets_bad_column_lists_available(site):
    with pytest.raises(ValueError, match=r"brain_regions.*Available columns: id, cell_type, pre, post"):
        query_facets(CatalogClient(site), "demo3d", "SELECT brain_regions FROM facets")
    with pytest.raises(ValueError, match="named `facets`"):
        query_facets(CatalogClient(site), "demo3d", "SELECT * FROM neurons")


def test_query_facets_store_name_is_an_alias_of_the_view(site):
    c = CatalogClient(site)          # what a model writes after reading describe_dataset's facet name
    assert query_facets(c, "demo3d", "SELECT count(*) FROM properties")["rows"] == [[5]]
    r = query_facets(c, "demo2d", "SELECT gene, sum(count) FROM expression GROUP BY 1 ORDER BY 1")
    assert r["rows"] == [["Gad1", 10], ["Slc17a7", 17]]
    with pytest.raises(Exception, match="(?i)single SELECT"):
        query_facets(c, "demo3d", "DROP VIEW properties")


def test_query_facets_store_selection(site):
    c = CatalogClient(site)
    assert query_facets(c, "demo3d", "SELECT 1 FROM facets LIMIT 1", facet="properties")["n_rows"] == 1
    with pytest.raises(ValueError, match="available: properties"):
        query_facets(c, "demo3d", "SELECT 1", facet="expression")


def test_query_facets_no_store_and_unknown_dataset(site):
    with pytest.raises(ValueError, match="no facet store"):
        query_facets(CatalogClient(site), "big3d", "SELECT 1")
    with pytest.raises(DatasetNotFound):
        query_facets(CatalogClient(site), "nope", "SELECT 1")


def test_query_facets_is_sandboxed(site):
    import duckdb
    c = CatalogClient(site)
    with pytest.raises(duckdb.Error):
        query_facets(c, "demo3d", f"SELECT * FROM '{site}/tiles2d/demo2d/facets/expression.parquet'")
    with pytest.raises(ValueError, match="single SELECT"):
        query_facets(c, "demo3d", "COPY facets TO 'x.csv'")


@pytest.mark.live
def test_live_kenyon_cells(tmp_path):
    c = CatalogClient("https://ngrf.mudm.io", tmp_path)
    r = query_facets(c, "hemibrain", "SELECT cell_type, count(*) AS n FROM facets WHERE cell_type LIKE 'KC%' "
                                     "GROUP BY 1 ORDER BY n DESC LIMIT 3")
    assert r["rows"] == [["KCg-m", 590], ["KCab-m", 342], ["KCab-s", 223]]
    assert "brain_regions" in r["missing_columns"]


@pytest.mark.live
def test_live_gene_expression_without_download(tmp_path):
    c = CatalogClient("https://ngrf.mudm.io", tmp_path)
    r = query_facets(c, "xenium-mouse-brain-coronal", "SELECT count(DISTINCT cell_id) FROM facets WHERE gene = 'Gad1'")
    assert r["rows"][0][0] > 1000 and not list(tmp_path.rglob("*.parquet"))
