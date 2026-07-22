"""The mudm-mcp tool implementations — thin, JSON-returning functions over CatalogClient + geometry.

Discovery/understanding tools (`list_datasets`, `describe_dataset`, `get_access`) are pure reads of the
static muDM catalog. They surface the SAME canonical facet values a human sees (carried as open metadata on
each PyramidEntry) and the column schema from the descriptor's TileLayer fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .catalog_client import CatalogClient
from .geometry import decode_geometry

# Facet keys carried (as open metadata) on each catalog entry; "modality" maps to the entry's `kind`.
_FACET_KEYS = ("organism", "sex", "technique", "prep", "condition", "tissue")
_SUMMARY_KEYS = ("id", "label", "kind", "organism", "sex", "technique", "condition", "tissue", "license")
_COORDS_NOTE = "Coordinates are image-pixel [X,Y,Z] in micrometres, NOT lon/lat."
_DECODE_NOTE = ("features.parquet is muDM binary tile geometry (NOT GeoParquet/WKB) — decode with "
                "mudm_mcp.geometry.decode_geometry, not ST_*/shapely.wkb.")


def list_datasets(client: CatalogClient, filter: dict | None = None) -> dict:
    """Discover + filter the catalog. `filter` accepts `modality` (→ `kind`), the facet keys, and `q`
    (free-text over id/label/summary/facets/tags). Reads `pyramids.json` only — no download."""
    filt = {k: v for k, v in (filter or {}).items() if v not in (None, "")}
    q = str(filt.pop("q", "")).lower()
    if "modality" in filt:
        filt["kind"] = filt.pop("modality")
    out: list[dict] = []
    for e in client.entries():
        if any(str(e.get(k)) != str(v) for k, v in filt.items()):
            continue
        if q:
            hay = " ".join(str(e.get(k, "")) for k in
                           ("id", "label", "summary", "organism", "technique", "tissue", "condition", "sex"))
            hay = (hay + " " + " ".join(map(str, e.get("tags") or []))).lower()
            if q not in hay:
                continue
        out.append({k: e.get(k) for k in _SUMMARY_KEYS})
    return {"count": len(out), "datasets": out}


def describe_dataset(client: CatalogClient, dataset_id: str) -> dict:
    """Deduce WHAT a dataset is: semantics (catalog entry) + the TileModel descriptor's column schema
    (TileLayer fields), layers, coordinate model, counts, and the muDM-specific caveats."""
    e = client.entry(dataset_id)
    d = client.descriptor(dataset_id)
    layers = d.get("vector_layers", []) or []
    columns: list[dict] = []
    seen: set[str] = set()
    for layer in layers:
        descs = layer.get("fielddescriptions") or {}
        for name, dtype in (layer.get("fields") or {}).items():
            if name not in seen:
                seen.add(name)
                columns.append({"name": name, "dtype": dtype, "desc": descs.get(name)})
    return {
        "id": dataset_id,
        "label": e.get("label"),
        "kind": e.get("kind"),
        "summary": e.get("summary"),
        "license": e.get("license"),
        "citation": e.get("citation"),
        **{k: e.get(k) for k in _FACET_KEYS},  # canonical facet values (== website)
        "coordinate_space": d.get("mudm:coordinate_space"),
        "bounds_um": d.get("mudm:bounds_um"),
        "feature_count": e.get("feature_count"),
        "layers": [{"id": layer.get("id"), "type": layer.get("description")} for layer in layers],
        "columns": columns,
        "caveats": [_COORDS_NOTE, _DECODE_NOTE],
    }


def get_access(client: CatalogClient, dataset_id: str) -> dict:
    """Deduce HOW to access a dataset: the descriptor's typed `assets[]`, the download index (the
    `features` asset's parts/zooms/full_res_zoom), a local DuckDB snippet, and the decode/coords notes."""
    d = client.descriptor(dataset_id)
    assets = d.get("assets", []) or []
    by_role = {a.get("role"): a for a in assets}
    out: dict[str, Any] = {"assets": assets, "coords_note": _COORDS_NOTE, "decode_note": _DECODE_NOTE}
    feat = by_role.get("features")
    if feat:
        out["download"] = {
            "features_base_url": feat.get("href"),
            "full_res_zoom": feat.get("full_res_zoom"),
            "zooms": feat.get("zooms"),
            "n_parts": len(feat.get("parts") or []),
            "parts": feat.get("parts"),
        }
        out["local_query"] = {
            "after": f'download("{dataset_id}")',
            "duckdb_snippet": ("SELECT feature_id, geom_type, tags FROM read_parquet("
                               f"'<cache>/{dataset_id}/features/**/*.mu.parquet', hive_partitioning=1) LIMIT 10"),
        }
    if by_role.get("raster"):
        r = by_role["raster"]
        out["raster"] = {"tile_url_template": r.get("href"),
                         "min_zoom": r.get("min_zoom"), "max_zoom": r.get("max_zoom")}
    if by_role.get("tiles3d"):
        out["tiles3d"] = {"tileset_url": by_role["tiles3d"].get("href")}
    return out


def _part_layer(part: str) -> str | None:
    """Layer from a leaf filename: 'zoom=9/cells_part_000.mu.parquet'→'cells'; bare 'part_061…'→None."""
    name = part.rsplit("/", 1)[-1]
    i = name.find("_part_")
    return name[:i] if i > 0 else None


def download(client: CatalogClient, dataset_id: str, layers: list[str] | None = None,
             zoom: str | int = "max", dest: str | None = None) -> dict:
    """Local-first fetch of a dataset's `features.parquet` leaves into the local cache. Defaults to the
    FULL-RESOLUTION (max-zoom) partition — the per-feature payload is replicated across zooms, so a single
    zoom IS the whole dataset (no redundant zoom copies). Idempotent-ish (overwrites). 3D / parquet-less
    datasets raise. Returns the local path + the fetched leaf list + byte total."""
    entry = client.entry(dataset_id)
    feat = client.features_asset(client.descriptor(dataset_id))
    if not feat or not feat.get("parts"):
        raise ValueError(f"{dataset_id}: no downloadable features.parquet (mesh / parquet-less dataset)")
    base = (entry.get("features") or "").rstrip("/")  # site-relative base → origin-agnostic (http or local)
    target = feat.get("full_res_zoom") if zoom in ("max", None) else int(zoom)
    sel = [p for p in feat["parts"] if target is None or p.startswith(f"zoom={target}/")]
    if layers:
        want = set(layers)
        sel = [p for p in sel if _part_layer(p) in want or _part_layer(p) is None] or sel
    dest_root = Path(dest) if dest else (Path(client.cache_dir) / dataset_id / "features")
    files, total = [], 0
    for part in sel:
        out = dest_root / part  # preserves the zoom=<z>/ hive structure
        total += client.download_file([base, part], out)
        files.append(str(out))
    return {"dataset": dataset_id, "zoom": target, "path": str(dest_root),
            "n_files": len(files), "bytes": total, "files": files}


def _jsonable(v: Any) -> Any:
    # Binary geometry columns (positions/indices) aren't JSON; placeholder them — decode via sample_patches.
    return f"<binary {len(v)}B>" if isinstance(v, (bytes, bytearray)) else v


def _features_root(client: CatalogClient, dataset_id: str, downloaded_dir: str | None) -> Path:
    return Path(downloaded_dir) if downloaded_dir else (Path(client.cache_dir) / dataset_id / "features")


def _open_features(root: Path):
    """A DuckDB connection exposing a ``features`` view over the downloaded leaves. Picks
    ``hive_partitioning`` by peeking the footer: the leaves live in zoom=<z>/ dirs, but the in-file schema
    VARIES — some carry a `zoom` COLUMN (→ hive OFF, else the path-derived BIGINT `zoom` collides with the
    file's int32 `zoom`), others have zoom ONLY in the path (→ hive ON so `zoom` exists). Caller closes."""
    import duckdb  # lazy
    import pyarrow.parquet as _pq  # lazy
    leaves = sorted(Path(root).rglob("*.parquet"))
    if not leaves:
        raise FileNotFoundError(f"no downloaded parquet under {root} — call download() first")
    hive = "false" if "zoom" in _pq.read_schema(leaves[0]).names else "true"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW features AS SELECT * FROM read_parquet("
                f"'{Path(root) / '**' / '*.parquet'}', hive_partitioning={hive})")
    return con


def _resolve_zoom(con, zoom):
    # CAST(...) sidesteps a DuckDB 1.5.x internal error ("index N within vector of size N") on max() over a
    # single-partition hive `zoom` column — which is the common case, since download() fetches only full-res.
    return con.execute("SELECT max(CAST(zoom AS INTEGER)) FROM features").fetchone()[0] if zoom in ("max", None) else int(zoom)


def query(client: CatalogClient, dataset_id: str, sql: str, downloaded_dir: str | None = None) -> dict:
    """Run DuckDB locally over the DOWNLOADED parquet (view ``features``). Errors if nothing is downloaded.
    Binary geometry columns come back as ``<binary NB>`` placeholders (decode via ``sample_patches``)."""
    con = _open_features(_features_root(client, dataset_id, downloaded_dir))
    try:
        rel = con.execute(sql)
        cols = [d[0] for d in rel.description]
        rows = [[_jsonable(v) for v in row] for row in rel.fetchall()]
    finally:
        con.close()
    return {"columns": cols, "n_rows": len(rows), "rows": rows}


def get_lod(client: CatalogClient, dataset_id: str, zoom: str | int = "max",
            downloaded_dir: str | None = None) -> dict:
    """Summarize ONE level-of-detail (zoom) of the downloaded data: feature/tile/unique counts +
    geom-type breakdown. The multi-LOD access primitive (also the (b)-experiment hook)."""
    con = _open_features(_features_root(client, dataset_id, downloaded_dir))
    try:
        z = _resolve_zoom(con, zoom)
        feats, tiles, uniq = con.execute(
            "SELECT count(*), count(DISTINCT (tile_x, tile_y)), count(DISTINCT feature_id) "
            f"FROM features WHERE zoom = {z}").fetchone()
        by_geom = con.execute(
            f"SELECT geom_type, count(*) FROM features WHERE zoom = {z} GROUP BY 1 ORDER BY 1").fetchall()
    finally:
        con.close()
    return {"dataset": dataset_id, "zoom": int(z), "features": int(feats), "tiles": int(tiles),
            "unique_features": int(uniq), "geom_types": {int(g): int(c) for g, c in by_geom}}


def sample_patches(client: CatalogClient, dataset_id: str, n: int = 6, zoom: str | int = "max",
                   max_features: int = 64, decode: bool = True, dims: int | None = None,
                   downloaded_dir: str | None = None) -> dict:
    """Return up to ``n`` spatial PATCHES (tiles) from the downloaded data — the tile-as-patch unit for ML.
    Densest tiles first; each patch lists up to ``max_features`` features (feature_id, geom_type, tags, and —
    when ``decode`` — geometry decoded to ``coords`` via ``decode_geometry``)."""
    if dims is None:
        try:
            dims = 3 if client.descriptor(dataset_id).get("mudm:kind") == "3d" else 2
        except Exception:
            dims = 2
    con = _open_features(_features_root(client, dataset_id, downloaded_dir))
    try:
        z = _resolve_zoom(con, zoom)
        tiles = con.execute(
            f"SELECT tile_x, tile_y, count(*) AS n FROM features WHERE zoom = {z} "
            f"GROUP BY 1, 2 ORDER BY n DESC, tile_x, tile_y LIMIT {int(n)}").fetchall()
        colnames = {d[0] for d in con.execute("SELECT * FROM features LIMIT 0").description}
        has_rings = "ring_lengths" in colnames
        sel = "feature_id, geom_type, positions, indices, " + ("ring_lengths, " if has_rings else "") + "tags"
        patches = []
        for tx, ty, cnt in tiles:
            rows = con.execute(
                f"SELECT {sel} FROM features "
                f"WHERE zoom = {z} AND tile_x = {tx} AND tile_y = {ty} LIMIT {int(max_features)}").fetchall()
            feats = []
            for row in rows:
                if has_rings:
                    fid, gt, pos, idx, rings, tags = row
                else:
                    fid, gt, pos, idx, tags = row; rings = []
                feat = {"feature_id": int(fid), "geom_type": int(gt), "tags": dict(tags) if tags else {}}
                if decode:
                    g = decode_geometry(bytes(pos or b""), bytes(idx or b""), list(rings or []), int(gt), dims=dims)
                    feat["coords"] = g["coords"].tolist()
                    if "rings" in g:
                        feat["n_rings"] = len(g["rings"])
                    if "triangles" in g:
                        feat["n_triangles"] = len(g["triangles"])
                feats.append(feat)
            patches.append({"tile_x": int(tx), "tile_y": int(ty), "zoom": int(z),
                            "n_features": int(cnt), "features": feats})
    finally:
        con.close()
    return {"dataset": dataset_id, "zoom": int(z), "n_patches": len(patches), "patches": patches}
