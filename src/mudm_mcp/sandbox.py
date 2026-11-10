"""A locked-down DuckDB connection for agent-written SQL, and a bounded SELECT runner.

The SQL comes from an LLM, so the connection may read exactly ONE source (a parquet file, a directory of
parquet leaves, or one https URL) and nothing else: no other local files, no other URLs, no writes
(COPY/ATTACH/EXPORT), no extension installs, and the configuration is locked so none of it can be
switched back on. Results are capped in rows and in characters so they fit an MCP host's output budget.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

MAX_ROWS_LIMIT = 5000
MAX_CELL_CHARS = 500
MAX_COLUMNS = 500
MAX_RESULT_CHARS = 15_000  # serialized rows budget (the whole tool response must stay under 20k)


def _q(text: str) -> str:
    return str(text).replace("'", "''")


def _ca_bundle() -> str | None:
    try:
        import certifi
        return certifi.where()
    except Exception:
        return None


def open_sandbox(view: str, source: str, *, is_dir: bool = False, hive: bool | None = None):
    """A DuckDB connection whose only readable data is ``source``, exposed as the view ``view``.

    ``source``: a local parquet file, a local directory (``is_dir=True`` → every ``*.parquet`` below
    it), or an http(s) URL of one parquet file. ``hive``: force ``hive_partitioning`` on/off (default:
    DuckDB's auto-detection). The caller closes the connection."""
    import duckdb  # lazy

    source = str(source)
    remote = source.startswith(("http://", "https://"))
    con = duckdb.connect()
    try:
        if remote:
            con.execute("INSTALL httpfs")  # before the lockdown: installing needs file + network access
            con.execute("LOAD httpfs")
        con.execute("SET allow_persistent_secrets=false")  # else CREATE VIEW probes ~/.duckdb after lockdown
        if is_dir:
            root = str(Path(source).resolve())
            con.execute(f"SET allowed_directories=['{_q(root)}/']")
            scan_src = f"{root}/**/*.parquet"
        else:
            scan_src = source if remote else str(Path(source).resolve())
            allowed = [scan_src]
            if remote:
                # DuckDB >= 2.0 reads a CA bundle from disk for TLS, which the lockdown would forbid:
                # allow exactly one bundle (certifi's public root list) and point httpfs at it.
                bundle = _ca_bundle()
                if bundle:
                    allowed.append(bundle)
                    try:
                        con.execute(f"SET ca_cert_file='{_q(bundle)}'")
                    except duckdb.Error:
                        pass  # older httpfs without the setting needs no bundle
            con.execute("SET allowed_paths=[" + ", ".join(f"'{_q(a)}'" for a in allowed) + "]")
        con.execute("SET enable_external_access=false")
        opts = "" if hive is None else f", hive_partitioning={'true' if hive else 'false'}"
        con.execute(f'CREATE VIEW "{view}" AS SELECT * FROM read_parquet(\'{_q(scan_src)}\'{opts})')
        con.execute("SET lock_configuration=true")
    except BaseException:
        con.close()
        raise
    return con


def jsonable(v: Any, cap: int = MAX_CELL_CHARS) -> Any:
    """A JSON-safe, size-bounded rendering of one DuckDB value: strings are cut at ``cap`` characters,
    nested values (maps, lists, structs) at ``4 * cap`` characters of their JSON."""
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<binary {len(v)}B>"  # muDM geometry blobs: decode via sample_patches / count_features
    if isinstance(v, str):
        return v if len(v) <= cap else v[:cap] + "..."
    if isinstance(v, (dict, list, tuple)):
        import json
        items = ({str(k): jsonable(x, cap) for k, x in list(v.items())[:100]} if isinstance(v, dict)
                 else [jsonable(x, cap) for x in v[:100]])
        text = json.dumps(items)
        return items if len(text) <= 4 * cap else text[:4 * cap] + "..."
    return jsonable(str(v), cap)  # dates, decimals, UUIDs, …


def _single_select(con, sql: str) -> str:
    import duckdb  # lazy

    text = str(sql or "").strip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    if not text:
        raise ValueError("Empty SQL. Only a single SELECT statement is allowed.")
    try:
        statements = con.extract_statements(text)
    except duckdb.Error as exc:
        raise ValueError(f"SQL syntax error: {exc}") from exc
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError("Only a single SELECT statement is allowed.")
    return text


def run_select(con, sql: str, max_rows: int = 200) -> dict:
    """Run one SELECT and return ``{"columns", "n_rows", "rows", "truncated"[, "note"]}``.

    At most ``max_rows`` rows (clamped to 1..5000) and about 15,000 characters of rows come back;
    ``truncated`` says whether more rows existed. A trailing ``;`` is accepted; anything that is not
    exactly one SELECT raises ``ValueError``."""
    import json

    text = _single_select(con, sql)
    max_rows = max(1, min(int(max_rows), MAX_ROWS_LIMIT))
    rel = con.execute(text)
    cols = [d[0] for d in rel.description]
    if len(cols) > MAX_COLUMNS:
        raise ValueError(f"The result has {len(cols)} columns; select at most {MAX_COLUMNS} "
                         "(name the columns you need instead of SELECT *).")
    # The column header and every cell share one character budget, so a wide result gets shorter cells
    # and fewer rows rather than an oversized response.
    budget = MAX_RESULT_CHARS - len(json.dumps(cols))
    cap = max(16, min(MAX_CELL_CHARS, budget // (4 * max(1, len(cols)))))
    fetched = rel.fetchmany(max_rows + 1)
    rows, used, note = [], 0, None
    for raw in fetched[:max_rows]:
        row = [jsonable(v, cap) for v in raw]
        used += len(json.dumps(row))  # ASCII-escaped: the worst case
        if rows and used > budget:
            note = (f"Stopped after {len(rows)} rows to keep the response small; select fewer columns, "
                    "aggregate, or add LIMIT/OFFSET.")
            break
        rows.append(row)
    truncated = len(fetched) > max_rows or note is not None
    out = {"columns": cols, "n_rows": len(rows), "rows": rows, "truncated": truncated}
    if truncated:
        out["note"] = note or (f"More than {max_rows} rows matched; only the first {max_rows} are shown. "
                               "Aggregate, filter, or raise max_rows (up to 5000).")
    return out
