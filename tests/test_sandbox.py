"""The SQL sandbox: one readable source, single SELECT only, bounded results."""
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mudm_mcp.sandbox import jsonable, open_sandbox, run_select


@pytest.fixture
def features_dir(site):
    return site / "tiles2d" / "demo2d" / "features.parquet"


@pytest.mark.parametrize("sql", [
    "SELECT * FROM read_csv('/etc/hosts')",
    "SELECT * FROM read_text('/etc/hosts')",
    "SELECT * FROM '{outside}'",
    "SELECT * FROM read_parquet('https://ngrf.mudm.io/tiles/hemibrain/facets/properties.parquet')",
    "COPY (SELECT 1) TO '{tmp}/x.csv'",
    "COPY features TO '{tmp}/x.parquet'",
    "SELECT 1; SELECT 2",
    "ATTACH '{tmp}/x.db'",
    "SET enable_external_access=true",
    "RESET enable_external_access",
    "CREATE TABLE t AS SELECT 1",
    "INSTALL spatial",
    "PRAGMA enable_profiling",
    "EXPLAIN SELECT 1",
    "FROM glob('/etc/*')",
    "DROP VIEW features",
    "",
])
def test_sandbox_blocks(sql, features_dir, site, tmp_path):
    con = open_sandbox("features", str(features_dir), is_dir=True)
    sql = sql.format(tmp=tmp_path, outside=site / "tiles" / "demo3d" / "facets" / "properties.parquet")
    with pytest.raises((ValueError, duckdb.Error)):
        run_select(con, sql)
    assert not list(tmp_path.glob("x.*"))
    assert run_select(con, "SELECT count(*) FROM features")["rows"] == [[13]]   # still usable afterwards


def test_sandbox_directory_view_and_trailing_semicolon(features_dir):
    con = open_sandbox("features", str(features_dir), is_dir=True, hive=True)
    r = run_select(con, "SELECT zoom, count(*) AS n FROM features GROUP BY 1 ORDER BY 1;")
    assert r == {"columns": ["zoom", "n"], "n_rows": 2, "rows": [[0, 4], [1, 9]], "truncated": False}


def test_describe_and_summarize_are_allowed(features_dir):
    con = open_sandbox("features", str(features_dir), is_dir=True)
    assert "feature_id" in [r[0] for r in run_select(con, "DESCRIBE features")["rows"]]
    assert run_select(con, "SUMMARIZE SELECT feature_id FROM features")["n_rows"] == 1


def test_sandbox_single_file(site):
    con = open_sandbox("facets", str(site / "tiles" / "demo3d" / "facets" / "properties.parquet"))
    assert run_select(con, "SELECT count(*) FROM facets WHERE cell_type LIKE 'KC%'")["rows"] == [[3]]
    with pytest.raises(duckdb.Error):
        run_select(con, f"SELECT * FROM '{site}/tiles2d/demo2d/facets/expression.parquet'")


def test_run_select_truncates_and_flags(features_dir):
    con = open_sandbox("features", str(features_dir), is_dir=True)
    r = run_select(con, "SELECT * FROM features", max_rows=2)
    assert r["n_rows"] == 2 and r["truncated"] is True and "max_rows" in r["note"]
    assert str(r["rows"][0][r["columns"].index("positions")]).startswith("<binary ")
    assert run_select(con, "SELECT 1 FROM features LIMIT 1000000", max_rows=2)["n_rows"] == 2
    assert run_select(con, "SELECT * FROM features", max_rows=10**9)["truncated"] is False   # clamped, 13 rows


def test_run_select_caps_characters(tmp_path):
    path = tmp_path / "wide.parquet"
    pq.write_table(pa.table({"s": ["x" * 400] * 200}), path)
    r = run_select(open_sandbox("t", str(path)), "SELECT * FROM t")
    assert r["truncated"] and r["n_rows"] < 200 and "Stopped after" in r["note"]


def test_jsonable_renders_awkward_values():
    import datetime
    import decimal
    assert jsonable(b"abc") == "<binary 3B>" and jsonable(float("nan")) is None
    assert jsonable("y" * 600).endswith("...") and len(jsonable("y" * 600)) == 503
    assert jsonable({"k": [1, b"z", datetime.date(2026, 1, 2), decimal.Decimal("1.5")]}) == \
        {"k": [1, "<binary 1B>", "2026-01-02", "1.5"]}


def test_run_select_budget_covers_wide_and_nested_results(tmp_path):
    import json
    path = tmp_path / "wide.parquet"
    pq.write_table(pa.table({f"c{i:03d}": ["v" * 450] * 50 for i in range(300)}), path)
    r = run_select(open_sandbox("t", str(path)), "SELECT * FROM t")
    assert r["n_rows"] >= 1 and r["truncated"] and len(json.dumps(r)) < 16_500   # one row alone would be 135k
    big_map = pa.array([[(f"k{i}", "z" * 400) for i in range(90)]] * 3, pa.map_(pa.string(), pa.string()))
    pq.write_table(pa.table({"tags": big_map}), path)
    r = run_select(open_sandbox("t", str(path)), "SELECT * FROM t")
    assert len(json.dumps(r)) < 16_500 and str(r["rows"][0][0]).endswith("...")
    pq.write_table(pa.table({f"c{i}": [1] for i in range(501)}), path)
    with pytest.raises(ValueError, match="501 columns"):
        run_select(open_sandbox("t", str(path)), "SELECT * FROM t")
