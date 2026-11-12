"""query_facets — SQL over a dataset's facet store (a per-cell / per-neuron attribute table published
as ONE parquet file next to the tiles), read in place over HTTP range requests: no download.

Facet stores by assay: Xenium ``expression`` (long: cell_id, gene, count) · CODEX / CosMx / MERSCOPE /
MIBI ``markers`` (wide: one column per marker or gene) · connectomes and reference anatomy ``properties``
(wide: one row per neuron / structure with cell_type, synapse counts, …). Visium has none.
"""
from __future__ import annotations

from .catalog_client import CatalogClient
from .sandbox import open_sandbox, run_select
from .tools import facet_assets

MAX_SCHEMA_COLUMNS = 60


def _names(cols: list[str], cap: int = MAX_SCHEMA_COLUMNS) -> str:
    more = f", … (+{len(cols) - cap} more)" if len(cols) > cap else ""
    return ", ".join(cols[:cap]) + more


def query_facets(client: CatalogClient, dataset_id: str, sql: str, facet: str | None = None,
                 max_rows: int = 200) -> dict:
    """Run ONE read-only SELECT against a dataset's facet store, exposed as the view ``facets`` (the
    store's own name, e.g. ``expression`` or ``properties``, works as an alias).

    ``facet`` picks the store by name when a dataset has several. Returns the bounded result of
    ``run_select`` plus ``facet``, ``source`` (the file's URL) and ``schema`` — the file's REAL columns,
    which can differ from what the descriptor advertises (``missing_columns`` lists advertised-but-absent
    ones)."""
    import duckdb  # lazy

    client.entry(dataset_id)  # unknown id -> DatasetNotFound with suggestions
    stores = facet_assets(client, dataset_id)
    if not stores:
        raise ValueError(f"{dataset_id} has no facet store (Visium datasets have none). "
                         "describe_dataset lists a dataset's facets; download + query reads the features.")
    names = [s["name"] for s in stores]
    if facet:
        store = next((s for s in stores if s["name"] == facet), None)
        if store is None:
            raise ValueError(f"{dataset_id} has no facet store named {facet!r}; available: {', '.join(names)}")
    elif len(stores) > 1:
        raise ValueError(f"{dataset_id} has several facet stores; pass facet= one of: {', '.join(names)}")
    else:
        store = stores[0]

    con = open_sandbox("facets", store["url"])
    try:
        # Agents often write FROM <store name> (e.g. FROM expression) because that is the name
        # describe_dataset shows: accept it as an alias of the `facets` view instead of failing.
        if store["name"].isidentifier() and store["name"].lower() != "facets":
            con.execute(f'CREATE VIEW "{store["name"]}" AS SELECT * FROM facets')
        schema = [{"name": r[0], "type": r[1]} for r in con.execute("DESCRIBE facets").fetchall()]
        cols = [c["name"] for c in schema]
        try:
            result = run_select(con, sql, max_rows=max_rows)
        except (duckdb.BinderException, duckdb.CatalogException) as exc:
            first = str(exc).split("\n", 1)[0]
            raise ValueError(f"{first} The view is named `facets`. Available columns: {_names(cols)}") from exc
    finally:
        con.close()
    out = {**result, "facet": store["name"], "layout": store["layout"], "source": store["url"],
           "n_columns": len(schema), "schema": schema[:MAX_SCHEMA_COLUMNS]}
    missing = [c for c in store["columns"] if c not in cols]
    if missing:
        out["missing_columns"] = missing[:MAX_SCHEMA_COLUMNS]
    return out
