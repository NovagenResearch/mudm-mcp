"""The mudm-mcp tool implementations — thin, JSON-returning functions over CatalogClient + geometry.

Discovery/understanding tools (`search_datasets`, `describe_dataset`, `get_access`) are pure reads of the
static muDM catalog. Every response is kept small (bounded lists, counts instead of enumerations) so it
fits an MCP host's output budget.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .catalog_client import CatalogClient
from .geometry import decode_geometry
from .sandbox import jsonable, open_sandbox, run_select
from .units import resolve_units

# Facet keys carried (as open metadata) on each catalog entry; "modality" maps to the entry's `kind`.
_FACET_KEYS = ("organism", "sex", "technique", "prep", "condition", "tissue")
_SUMMARY_KEYS = ("id", "label", "kind", "organism", "technique", "tissue", "feature_count")
_SEARCH_KEYS = ("id", "label", "summary", "organism", "technique", "tissue", "condition", "sex", "prep")
_DECODE_NOTE = ("features.parquet is muDM binary tile geometry (NOT GeoParquet/WKB): SQL cannot filter on "
                "coordinates. Use count_features / sample_patches, or decode with "
                "mudm_mcp.geometry.decode_geometry.")
# Common names people type that the catalog's Latin binomials do not contain.
_COMMON_NAMES = {"mus musculus": "mouse", "homo sapiens": "human",
                 "drosophila melanogaster": "fly fruit-fly fruitfly drosophila"}
MAX_LISTED = 30  # cap for attribute / column / tag-key lists in one response


def units_note(units: dict) -> str:
    if units["basis"].startswith("unknown"):
        return ("Vertex unit could not be determined; micrometres are assumed. Check coordinates against "
                "the dataset bounds. Not lon/lat.")
    if units["um_per_unit"] == 1.0:
        return "Vertices are stored in micrometres (image space, not lon/lat)."
    return (f"Vertices are stored in {units['vertex_unit']}s; multiply by {units['um_per_unit']:g} for "
            "micrometres (image space, not lon/lat). Tools taking *_um arguments convert for you.")


def _haystack(e: dict) -> str:
    hay = " ".join(str(e.get(k) or "") for k in _SEARCH_KEYS)
    hay = (hay + " " + " ".join(map(str, e.get("tags") or []))).lower()
    return hay + " " + _COMMON_NAMES.get(str(e.get("organism") or "").lower(), "")


MAX_SEARCH_LIMIT = 50
_OVERVIEW_KEYS = ("kind", "technique", "organism", "tissue")


def _overview(entries: list[dict], top: int = 12) -> dict:
    """Value counts of the main catalog facets over ``entries`` (largest first, ``top`` per facet)."""
    out = {}
    for key in _OVERVIEW_KEYS:
        counts: dict[str, int] = {}
        for e in entries:
            v = str(e.get(key) or "unspecified")
            counts[v] = counts.get(v, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        out["modality" if key == "kind" else key] = dict(ranked[:top])
        if len(ranked) > top:
            out[("modality" if key == "kind" else key)]["(other values)"] = sum(n for _, n in ranked[top:])
    return out


def search_datasets(client: CatalogClient, q: str = "", modality: str | None = None,
                    organism: str | None = None, technique: str | None = None, tissue: str | None = None,
                    limit: int = 25, offset: int = 0, sort: str = "featured") -> dict:
    """Find datasets. ``q`` is free text: every whitespace-separated word must occur (case-insensitive)
    in the id, label, summary, organism (Latin or common name), technique, tissue, condition or tags.
    ``modality`` ("2d"/"3d"), ``organism`` (Latin or common name), ``technique``, ``tissue`` are
    case-insensitive substring filters. Returns the total match ``count``, an ``overview`` (how the
    matches split by modality / technique / organism / tissue) and up to ``limit`` (max 50) summaries
    from ``offset``; ``sort`` is "featured" (default) or "size" (most features first)."""
    tokens = str(q or "").lower().split()
    filters = {"kind": modality, "technique": technique, "tissue": tissue}
    filters = {k: str(v).lower() for k, v in filters.items() if v not in (None, "")}
    organism = str(organism or "").lower().strip()
    if str(sort).lower() not in ("featured", "size"):
        raise ValueError(f"sort must be 'featured' or 'size', got {sort!r}")
    entries = client.entries()
    hits: list[dict] = []
    for e in entries:
        if any(v not in str(e.get(k) or "").lower() for k, v in filters.items()):
            continue
        if organism:
            latin = str(e.get("organism") or "").lower()
            if organism not in latin and organism not in _COMMON_NAMES.get(latin, "").split():
                continue
        if tokens:
            hay = _haystack(e)
            if not all(t in hay for t in tokens):
                continue
        hits.append(e)
    if str(sort).lower() == "size":
        hits.sort(key=lambda e: (-(e.get("feature_count") or 0), str(e.get("id"))))
    else:
        hits.sort(key=lambda e: (not e.get("featured"), str(e.get("id"))))
    limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    offset = max(0, int(offset))
    shown = [{k: e.get(k) for k in _SUMMARY_KEYS} for e in hits[offset:offset + limit]]
    out: dict[str, Any] = {"count": len(hits), "returned": len(shown), "offset": offset, "datasets": shown}
    if hits:
        out["overview"] = _overview(hits)
        rest = len(hits) - offset - len(shown)
        if rest > 0:
            out["note"] = (f"{rest} more match: narrow the search, or page with offset={offset + len(shown)} "
                           f"(limit is at most {MAX_SEARCH_LIMIT}).")
    else:
        out["note"] = ("No dataset matches. Words in q must ALL occur and filters are substrings, so try "
                       "fewer or simpler terms. catalog_overview shows the values that exist.")
        out["catalog_overview"] = _overview(entries)
    return out


def _part_layer(part: str) -> str | None:
    """Layer from a leaf filename: 'zoom=9/cells_part_000.mu.parquet'→'cells'; bare 'part_061…'→None."""
    name = part.rsplit("/", 1)[-1]
    i = name.find("_part_")
    return name[:i] if i > 0 else None


def _part_zoom(part: str) -> int:
    return int(part.split("/", 1)[0].split("=", 1)[1])


def part_layers(parts: list[str]) -> list[str]:
    """Distinct layer prefixes found in the part names, in first-seen order (empty for unprefixed 3D parts)."""
    return list(dict.fromkeys(l for l in map(_part_layer, parts) if l))


def lod_table(client: CatalogClient, feat: dict, sample: int = 3) -> list[dict]:
    """Per-zoom ``{"zoom", "n_parts", "est_bytes"}`` for a features index (sizes sampled concurrently)."""
    by_zoom: dict[int, list[str]] = {}
    for p in feat["parts"]:
        by_zoom.setdefault(_part_zoom(p), []).append(p)
    zooms = sorted(by_zoom)
    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(lambda z: client.estimate_bytes(feat["base"], by_zoom[z], sample), zooms))
    return [{"zoom": z, "n_parts": len(by_zoom[z]), "est_bytes": s} for z, s in zip(zooms, sizes)]


def _peek_part(client: CatalogClient, base: str, part: str) -> tuple[list[dict], list[str]]:
    """``(parquet columns, tag keys of the first row)`` of one leaf, reading only the footer + one small
    column chunk (works over HTTP range requests)."""
    import duckdb  # lazy
    src = client.url(base, part).replace("'", "''")
    scan = f"read_parquet('{src}', hive_partitioning=false)"  # the file's own columns, not the zoom= dir
    con = duckdb.connect()
    try:
        if client.is_http:
            con.execute("INSTALL httpfs; LOAD httpfs")
        cols = [{"name": r[0], "type": r[1]} for r in
                con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()]
        keys: list[str] = []
        if any(c["name"] == "tags" for c in cols):
            row = con.execute(f"SELECT map_keys(tags) FROM {scan} LIMIT 1").fetchone()
            keys = [str(k) for k in (row[0] if row and row[0] else [])]
    finally:
        con.close()
    return cols, keys


def facet_assets(client: CatalogClient, dataset_id: str, descriptor: dict | None = None) -> list[dict]:
    """The dataset's facet stores (per-cell / per-neuron attribute tables), each as
    ``{"name", "url", "layout", "key", "columns", "about"}`` with ``url`` absolute (or a local path).
    Read from the agent descriptor, else from the viewer descriptor (whose hrefs are relative)."""
    descriptor = descriptor if descriptor is not None else client.descriptor(dataset_id)
    assets = [a for a in descriptor.get("assets") or [] if a.get("role") == "facets"]
    rel_dir = None
    if not assets:
        found = client.viewer_descriptor(dataset_id)
        if found:
            rel_dir, assets = found[0], [a for a in found[1].get("assets") or [] if a.get("role") == "facets"]
    out = []
    for a in assets:
        href = str(a.get("href") or "")
        url = client.url(rel_dir, href) if rel_dir is not None and "://" not in href else href
        name = href.rstrip("/").rsplit("/", 1)[-1].removesuffix(".parquet")
        out.append({"name": name, "url": url, "layout": a.get("layout"), "key": a.get("key"),
                    "columns": list(a.get("columns") or []), "about": a.get("facet") or name})
    return out


def _capped(items: list, cap: int = MAX_LISTED) -> list:
    return items[:cap]


def describe_dataset(client: CatalogClient, dataset_id: str) -> dict:
    """Deduce WHAT a dataset is: catalog semantics, vertex units, layers, the REAL parquet columns (read
    from a leaf's footer), per-feature attributes and where each one lives (`tags` map key vs facet
    column), facet stores, and per-zoom sizes. Bounded: long lists are capped and counted."""
    e = client.entry(dataset_id)
    d = client.descriptor(dataset_id)
    units = resolve_units(e, d)
    feat = client.features(dataset_id)
    caveats = [units_note(units), _DECODE_NOTE]

    vlayers = d.get("vector_layers") or []
    geometry = {str(l.get("id")): l.get("description") for l in vlayers}
    layer_ids = (part_layers(feat["parts"]) if feat else []) or list(geometry)
    default_geom = next(iter(geometry.values()), None)
    layers = [{"id": i, "geometry": geometry.get(i, default_geom if len(geometry) == 1 else None)}
              for i in layer_ids]

    columns, tag_keys, lod = None, [], []
    if feat:
        lod = lod_table(client, feat)
        coarse = [p for p in feat["parts"] if _part_zoom(p) == feat["zooms"][0]]
        try:
            columns, tag_keys = _peek_part(client, feat["base"], coarse[0])
        except Exception as exc:  # the footer read is a convenience; never fail the description on it
            caveats.append(f"Could not read a parquet footer ({type(exc).__name__}); parquet_columns unknown.")
    else:
        caveats.append("No features.parquet is published for this dataset.")

    facets = facet_assets(client, dataset_id, d)
    facet_of = {c: f["name"] for f in facets for c in f["columns"]}
    col_names = {c["name"] for c in columns or []}
    attributes, seen = [], set()
    for layer in vlayers:
        enums, ranges = layer.get("fieldenums") or {}, layer.get("fieldranges") or {}
        for name, dtype in (layer.get("fields") or {}).items():
            if name in col_names or name in seen or name == "zoom":  # zoom is the partition key
                continue
            seen.add(name)
            where = ("tags" if name in tag_keys else f"facet:{facet_of[name]}" if name in facet_of
                     else "tags" if not tag_keys and d.get("mudm:kind") == "2d" else "descriptor only")
            a: dict[str, Any] = {"name": name, "type": dtype, "in": where}
            if name in enums:
                a["n_values"], a["examples"] = len(enums[name]), [str(v)[:40] for v in enums[name][:8]]
            if name in ranges:
                a["range"] = ranges[name]
            attributes.append(a)
    if any(a["in"] == "tags" for a in attributes) or tag_keys:
        caveats.append("`tags` is a MAP(VARCHAR, VARCHAR) per feature: read a key as tags['<key>'] and CAST "
                       "numbers. Keys may be spelled differently from the facet columns.")
    out_facets = [{**{k: f[k] for k in ("name", "about", "layout", "key")},
                   "n_columns": len(f["columns"]), "columns": _capped(f["columns"])} for f in facets]
    return {
        "id": dataset_id,
        "label": e.get("label"),
        "kind": e.get("kind"),
        "summary": e.get("summary"),
        "license": e.get("license"),
        "citation": e.get("citation"),
        **{k: e.get(k) for k in _FACET_KEYS},
        "feature_count": e.get("feature_count"),
        "bounds_um": d.get("mudm:bounds_um"),
        "units": units,
        "layers": layers,
        "parquet_columns": columns,
        "tag_keys": _capped(tag_keys),
        "n_attributes": len(attributes),
        "attributes": _capped(attributes),
        "facets": out_facets,
        "lod": lod,
        "full_res_zoom": feat["full_res_zoom"] if feat else None,
        "caveats": caveats,
    }


DEFAULT_MAX_BYTES = 500_000_000  # download gate: larger fetches need confirm_large=True


def get_access(client: CatalogClient, dataset_id: str) -> dict:
    """Deduce HOW to access a dataset without this server: where the parquet leaves live (URL template,
    per-zoom part counts + estimated bytes — never the part list itself), the facet files, raster / 3D
    tile endpoints, bulk archives, the vertex units and a minimal Python decode snippet."""
    e = client.entry(dataset_id)
    d = client.descriptor(dataset_id)
    units = resolve_units(e, d)
    by_role: dict[str, list[dict]] = {}
    for a in d.get("assets") or []:
        by_role.setdefault(str(a.get("role")), []).append(a)
    out: dict[str, Any] = {"id": dataset_id,
                           "descriptor_url": client.url("datasets", dataset_id, "tilejson.json")}
    feat = client.features(dataset_id)
    if feat:
        base_url = client.url(feat["base"])
        example = [p for p in feat["parts"] if _part_zoom(p) == feat["zooms"][0]][:3]
        out["features"] = {
            "base_url": base_url,
            "url_template": base_url + "/zoom={zoom}/{part_file}",
            "full_res_zoom": feat["full_res_zoom"],
            "zooms": lod_table(client, feat),
            "layers": part_layers(feat["parts"]),
            "example_parts": example,
            "part_list": ("The complete list is the `parts` array of the role=features asset in "
                          + (out["descriptor_url"] if feat["source"] == "agent"
                             else client.url(feat["base"].rsplit("/", 1)[0], "tilejson.json"))),
            "note": ("Hive-partitioned by zoom. Every zoom holds the same features at a different level "
                     "of detail; the coarsest zoom is the cheapest complete copy."),
        }
        dims = 3 if (d.get("mudm:kind") or e.get("kind")) == "3d" else 2
        scale = "" if units["um_per_unit"] == 1.0 else f" * {units['um_per_unit']:g}"
        out["python_snippet"] = (
            "import io, httpx, numpy as np, pyarrow.parquet as pq\n"
            f"t = pq.read_table(io.BytesIO(httpx.get('{base_url}/{example[0]}').content))\n"
            f"xyz_um = np.frombuffer(t['positions'][0].as_py(), '<f4').reshape(-1, {dims}){scale}"
            "  # vertices of feature 0, micrometres")
    facets = facet_assets(client, dataset_id, d)
    if facets:
        out["facets"] = [{"name": f["name"], "url": f["url"], "layout": f["layout"], "key": f["key"],
                          "n_columns": len(f["columns"])} for f in facets]
    if by_role.get("raster"):
        out["raster"] = [{"tile_url_template": r.get("href"), "channel": r.get("channel"),
                          "min_zoom": r.get("min_zoom"), "max_zoom": r.get("max_zoom")}
                         for r in by_role["raster"][:8]]
    if by_role.get("vector"):
        out["vector_tiles"] = {"tile_url_template": by_role["vector"][0].get("href")}
    if by_role.get("tiles3d"):
        out["tiles3d"] = {"tileset_url": by_role["tiles3d"][0].get("href")}
    if by_role.get("download"):
        out["bulk_downloads"] = [{"url": b.get("href"), "bytes": b.get("bytes"), "artifact": b.get("artifact")}
                                 for b in by_role["download"][:8]]
    out.update(units=units, units_note=units_note(units), decode_note=_DECODE_NOTE)
    return out


def plan_download(client: CatalogClient, dataset_id: str, zoom: str | int | None = None,
                  layers: list[str] | None = None) -> dict:
    """Resolve a download request to concrete parts: ``{feat, kind, zoom, layers, parts, est_bytes}``.
    Defaults — ``zoom``: full resolution for 2D, the COARSEST zoom for 3D (meshes are large; ``"max"``
    asks for full resolution); ``layers``: ``["cells"]`` when a 2D dataset has that layer, else all."""
    feat = client.features(dataset_id)
    if not feat:
        raise ValueError(f"{dataset_id}: no downloadable features.parquet is published for this dataset")
    kind = client.descriptor(dataset_id).get("mudm:kind") or client.entry(dataset_id).get("kind")
    if zoom is None or zoom == "":
        target = feat["zooms"][0] if kind == "3d" else feat["full_res_zoom"]
    elif str(zoom).lower() == "max":
        target = feat["full_res_zoom"]
    else:
        try:
            target = int(zoom)
        except (TypeError, ValueError):
            target = None
        if target not in feat["zooms"]:
            raise ValueError(f"zoom {zoom!r} is not available for {dataset_id}; "
                             f"valid zooms: {', '.join(map(str, feat['zooms']))} (or \"max\")")
    sel = [p for p in feat["parts"] if _part_zoom(p) == target]
    available = part_layers(sel)
    if layers:
        missing = [l for l in layers if l not in available]
        if missing:
            raise ValueError(f"layer(s) {', '.join(missing)} not found in {dataset_id} at zoom {target}; "
                             f"available: {', '.join(available) or '(single unnamed layer)'}")
        chosen = list(layers)
    else:
        chosen = ["cells"] if kind != "3d" and "cells" in available else available
    if available:
        sel = [p for p in sel if _part_layer(p) in chosen]
    return {"feat": feat, "kind": kind, "zoom": target, "layers": chosen, "parts": sel,
            "est_bytes": client.estimate_bytes(feat["base"], sel)}


def download(client: CatalogClient, dataset_id: str, zoom: str | int | None = None,
             layers: list[str] | None = None, max_bytes: int = DEFAULT_MAX_BYTES,
             confirm_large: bool = False, dest: str | None = None, plan: dict | None = None) -> dict:
    """Fetch one zoom of a dataset's `features.parquet` leaves into the local cache (see
    ``plan_download`` for the zoom/layer defaults). Size-gated: when the estimate exceeds ``max_bytes``
    (or cannot be determined) and ``confirm_large`` is false, NOTHING is fetched and a
    ``status: "needs_confirmation"`` result explains the options. Resumable: complete files already in
    the cache are skipped. Returns the local path and totals, not a file list."""
    plan = plan or plan_download(client, dataset_id, zoom, layers)  # `plan`: a precomputed plan_download()
    feat, est = plan["feat"], plan["est_bytes"]
    summary = {"dataset": dataset_id, "zoom": plan["zoom"], "layers": plan["layers"],
               "n_files": len(plan["parts"])}
    if not confirm_large and (est is None or est > max_bytes):
        coarser = [z for z in feat["zooms"] if z < plan["zoom"]]
        options = ["query_facets answers per-neuron / per-cell attribute questions with no download"
                   if plan["kind"] == "3d" else
                   "count_features and query_facets need little or no download"]
        if coarser:
            options.append(f"a coarser zoom is smaller (available: {', '.join(map(str, coarser))})")
        options.append("pass confirm_large=true to download anyway")
        size = "an unknown size" if est is None else f"about {est / 1e6:,.0f} MB"
        return {"status": "needs_confirmation", **summary, "est_bytes": est, "max_bytes": int(max_bytes),
                "hint": f"This would download {len(plan['parts'])} files, {size} (limit "
                        f"{max_bytes / 1e6:,.0f} MB). Options: " + "; ".join(options) + "."}
    dest_root = Path(dest) if dest else (Path(client.cache_dir) / dataset_id / "features")

    def fetch(part: str) -> tuple[int, bool]:
        out = dest_root / part  # preserves the zoom=<z>/ hive structure
        if out.is_file():  # complete by construction: download_file renames only finished files
            return out.stat().st_size, True
        return client.download_file([feat["base"], part], out), False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, plan["parts"]))
    return {"status": "ok", **summary, "path": str(dest_root),
            "n_skipped": sum(1 for _, skipped in results if skipped),
            "bytes": sum(size for size, _ in results)}


def _features_root(client: CatalogClient, dataset_id: str, downloaded_dir: str | None) -> Path:
    return Path(downloaded_dir) if downloaded_dir else (Path(client.cache_dir) / dataset_id / "features")


def _open_features(root: Path):
    """A sandboxed DuckDB connection exposing a ``features`` view over the downloaded leaves (it can
    read nothing else). Picks ``hive_partitioning`` by peeking the footer: the leaves live in zoom=<z>/
    dirs, but the in-file schema VARIES — some carry a `zoom` COLUMN (→ hive OFF, else the path-derived
    BIGINT `zoom` collides with the file's int32 `zoom`), others have zoom ONLY in the path (→ hive ON so
    `zoom` exists). Caller closes."""
    import pyarrow.parquet as _pq  # lazy
    leaves = sorted(Path(root).rglob("*.parquet"))
    if not leaves:
        raise FileNotFoundError(f"no downloaded parquet under {root} — call download() first")
    hive = "zoom" not in _pq.read_schema(leaves[0]).names
    return open_sandbox("features", str(root), is_dir=True, hive=hive)


def _resolve_zoom(con, zoom):
    # CAST(...) sidesteps a DuckDB 1.5.x internal error ("index N within vector of size N") on max() over a
    # single-partition hive `zoom` column — which is the common case, since download() fetches one zoom.
    if zoom in ("max", None, ""):
        return con.execute("SELECT max(CAST(zoom AS INTEGER)) FROM features").fetchone()[0]
    return int(zoom)


def query(client: CatalogClient, dataset_id: str, sql: str, max_rows: int = 200,
          downloaded_dir: str | None = None) -> dict:
    """Run ONE read-only SELECT with DuckDB over the DOWNLOADED parquet (view ``features``). Errors if
    nothing is downloaded. The connection is sandboxed (no other files/URLs, no writes) and the result
    is capped (``max_rows``, ``truncated``). Binary geometry columns come back as ``<binary NB>``."""
    con = _open_features(_features_root(client, dataset_id, downloaded_dir))
    try:
        return run_select(con, sql, max_rows=max_rows)
    finally:
        con.close()


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


MAX_COORDS = 64            # vertices returned per feature when include_coords is set
PATCH_BUDGET_CHARS = 14_000


def _dataset_geometry(client, dataset_id: str) -> tuple[int, dict]:
    """``(dims, units)`` for a dataset; falls back to 2D / unknown-units when the catalog is unreachable
    (e.g. analysing an already-downloaded directory offline)."""
    try:
        d = client.descriptor(dataset_id)
        entry = client.entry(dataset_id) if hasattr(client, "entry") else {}
        kind = d.get("mudm:kind") or entry.get("kind")
        return (3 if kind == "3d" else 2), resolve_units(entry, d)
    except Exception:
        return 2, resolve_units({}, {})


def sample_patches(client: CatalogClient, dataset_id: str, n: int = 2, zoom: str | int = "max",
                   max_features: int = 8, include_coords: bool = False, dims: int | None = None,
                   downloaded_dir: str | None = None) -> dict:
    """Return up to ``n`` spatial PATCHES (tiles) from the downloaded data — the tile-as-patch unit for ML.
    Densest tiles first; each patch lists up to ``max_features`` features with ``feature_id``,
    ``geom_type``, ``tags`` and a geometry SUMMARY in micrometres (``n_vertices``, ``bbox_um``,
    ``centroid_um``). ``include_coords=True`` adds the decoded vertices in STORED units (first 64 per
    feature). ``n`` is clamped to 1..16, ``max_features`` to 1..64, and the response to ~14k chars."""
    import json

    import numpy as np
    n = max(1, min(int(n), 16))
    max_features = max(1, min(int(max_features), 64))
    auto_dims, units = _dataset_geometry(client, dataset_id)
    dims = int(dims) if dims else auto_dims
    f = units["um_per_unit"]
    con = _open_features(_features_root(client, dataset_id, downloaded_dir))
    truncated = False
    try:
        z = _resolve_zoom(con, zoom)
        tiles = con.execute(
            f"SELECT tile_x, tile_y, count(*) AS n FROM features WHERE zoom = {z} "
            f"GROUP BY 1, 2 ORDER BY n DESC, tile_x, tile_y LIMIT {n}").fetchall()
        colnames = {d[0] for d in con.execute("SELECT * FROM features LIMIT 0").description}
        has_rings = "ring_lengths" in colnames
        sel = "feature_id, geom_type, positions, indices, " + ("ring_lengths, " if has_rings else "") + "tags"
        patches, used = [], 0
        for tx, ty, cnt in tiles:
            rows = con.execute(
                f"SELECT {sel} FROM features "
                f"WHERE zoom = {z} AND tile_x = {tx} AND tile_y = {ty} LIMIT {max_features}").fetchall()
            feats = []
            for row in rows:
                if has_rings:
                    fid, gt, pos, idx, rings, tags = row
                else:
                    fid, gt, pos, idx, tags = row
                    rings = []
                g = decode_geometry(bytes(pos or b""), bytes(idx or b""), list(rings or []), int(gt), dims=dims)
                xyz = g["coords"]
                feat: dict[str, Any] = {"feature_id": int(fid), "geom_type": int(gt),
                                        "tags": jsonable(dict(tags) if tags else {}),
                                        "n_vertices": int(len(xyz))}
                if len(xyz):
                    um = xyz.astype(np.float64) * f
                    feat["bbox_um"] = [round(float(v), 3) for v in (*um.min(0), *um.max(0))]
                    feat["centroid_um"] = [round(float(v), 3) for v in um.mean(0)]
                if "rings" in g:
                    feat["n_rings"] = len(g["rings"])
                if "triangles" in g:
                    feat["n_triangles"] = len(g["triangles"])
                if include_coords:
                    feat["coords"] = xyz[:MAX_COORDS].tolist()
                    if len(xyz) > MAX_COORDS:
                        feat["coords_truncated"] = True
                used += len(json.dumps(feat))
                if feats and used > PATCH_BUDGET_CHARS:
                    truncated = True
                    break
                feats.append(feat)
            patches.append({"tile_x": int(tx), "tile_y": int(ty), "zoom": int(z),
                            "n_features": int(cnt), "features": feats})
            if truncated:
                break
    finally:
        con.close()
    out = {"dataset": dataset_id, "zoom": int(z), "units": units, "n_patches": len(patches), "patches": patches}
    if include_coords:
        out["coords_unit"] = units["vertex_unit"]
    if truncated:
        out["truncated"] = True
        out["note"] = "Stopped early to keep the response small; lower n / max_features or drop include_coords."
    return out
