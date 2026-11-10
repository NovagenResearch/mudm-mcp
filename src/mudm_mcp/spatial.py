"""count_features — how many cells (or other 2D features) lie in a region given in MICROMETRES.

muDM geometry is packed binary, so SQL cannot filter on coordinates. This reads the vertex buffers
directly, reduces every feature to its vertex centroid, converts stored units to micrometres with the
dataset's factor (see ``mudm_mcp.units``) and tests the centroid against the box.

It reads the COARSEST zoom by default: in a 2D pyramid that level holds every feature exactly once and
whole, in the same coordinate space, at a fraction of the bytes; at finer zooms features are clipped at
tile edges and repeated across tiles.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import tools
from .catalog_client import CatalogClient
from .units import resolve_units

MAX_GROUPS = 50
BBOX_HELP = "bbox_um must be [xmin, ymin, xmax, ymax] in micrometres with xmin <= xmax and ymin <= ymax"


def _check_bbox(bbox_um) -> list[float] | None:
    if bbox_um is None:
        return None
    try:
        box = [float(v) for v in bbox_um]
    except (TypeError, ValueError):
        raise ValueError(f"{BBOX_HELP}; got {bbox_um!r}") from None
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[0] > box[2] or box[1] > box[3]:
        raise ValueError(f"{BBOX_HELP}; got {bbox_um!r}")
    return box


def _centroid_sums(positions, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(coordinate sums [n, dims], vertex counts [n])`` of a (large_)binary arrow array of
    packed little-endian float32 vertices — vectorised over the array's offset + data buffers."""
    import pyarrow as pa

    n = len(positions)
    if n == 0:
        return np.zeros((0, dims)), np.zeros(0, dtype=np.int64)
    off_t = np.int64 if pa.types.is_large_binary(positions.type) else np.int32
    _validity, offsets_buf, data_buf = positions.buffers()
    offsets = np.frombuffer(offsets_buf, dtype=off_t)[positions.offset:positions.offset + n + 1].astype(np.int64)
    stride = 4 * dims
    counts = (offsets[1:] - offsets[:-1]) // stride
    sums = np.zeros((n, dims), dtype=np.float64)
    if data_buf is None or counts.sum() == 0:
        return sums, counts
    raw = np.frombuffer(data_buf, dtype=np.uint8)
    nonempty = np.flatnonzero(counts)
    lo, hi = int(offsets[0]), int(offsets[-1])
    if (hi - lo) == int(counts.sum()) * stride:  # every row is a whole number of vertices
        coords = raw[lo:hi].view("<f4").reshape(-1, dims).astype(np.float64)  # contiguous: one pass
        starts = ((offsets[:-1] - lo) // stride)[nonempty]
        sums[nonempty] = np.add.reduceat(coords, starts, axis=0)
    else:  # ragged tails (trailing bytes that are not whole vertices): per-row fallback
        for i in nonempty:
            a = raw[offsets[i]:offsets[i] + counts[i] * stride].view("<f4").reshape(-1, dims)
            sums[i] = a.sum(axis=0, dtype=np.float64)
    return sums, counts


def count_features(client: CatalogClient, dataset_id: str, bbox_um: list[float] | None = None,
                   layer: str | None = None, zoom: int | str | None = None, group_by: str | None = None,
                   max_bytes: int = tools.DEFAULT_MAX_BYTES, confirm_large: bool = False) -> dict:
    """Count the features of a 2D dataset whose vertex centroid lies inside ``bbox_um``
    (``[xmin, ymin, xmax, ymax]`` in micrometres; ``None`` counts all).

    ``layer`` defaults to ``cells`` when the dataset has it; ``zoom`` defaults to the coarsest level
    (complete and cheapest). ``group_by`` names a key of the per-feature ``tags`` map to break the count
    down by (e.g. ``cell_type``). The needed leaves are downloaded through the same size gate as
    ``download``; a ``needs_confirmation`` result is returned unchanged. 3D datasets are not supported."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    box = _check_bbox(bbox_um)
    entry = client.entry(dataset_id)
    descriptor = client.descriptor(dataset_id)
    if (descriptor.get("mudm:kind") or entry.get("kind")) == "3d":
        raise ValueError(
            f"{dataset_id} is a 3D dataset: count_features supports 2D datasets only (3D features are large "
            "meshes). Use query_facets for neuron / structure counts by attribute, e.g. "
            "SELECT cell_type, count(*) FROM facets GROUP BY 1.")
    units = resolve_units(entry, descriptor)
    feat = client.features(dataset_id)
    if not feat:
        raise ValueError(f"{dataset_id}: no features.parquet is published for this dataset")
    plan = tools.plan_download(client, dataset_id, feat["zooms"][0] if zoom in (None, "") else zoom,
                               [layer] if layer else None)
    got = tools.download(client, dataset_id, max_bytes=max_bytes, confirm_large=confirm_large, plan=plan)
    if got["status"] != "ok":
        return got

    root = Path(got["path"])
    want = ["feature_id", "positions"] + (["tags"] if group_by else [])
    ids_l, sums_l, counts_l, labels_l, tag_keys = [], [], [], [], []
    for part in plan["parts"]:
        t = pq.read_table(root / part, columns=want)
        ids_l.append(t.column("feature_id").to_numpy().astype(np.int64))
        for chunk in t.column("positions").chunks:
            s, c = _centroid_sums(chunk, 2)
            sums_l.append(s)
            counts_l.append(c)
        if group_by:
            if not tag_keys and t.num_rows:
                tag_keys = [str(k) for k, _ in (t.column("tags")[0].as_py() or [])]
            labels_l.append(pc.map_lookup(t.column("tags"), group_by, "first").to_numpy(zero_copy_only=False))
    ids = np.concatenate(ids_l) if ids_l else np.zeros(0, dtype=np.int64)
    sums = np.concatenate(sums_l) if sums_l else np.zeros((0, 2))
    counts = np.concatenate(counts_l) if counts_l else np.zeros(0, dtype=np.int64)

    # One centroid per FEATURE: at finer zooms a feature is split over tiles (several rows per id).
    uniq, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    n_vertices = np.bincount(inverse, weights=counts, minlength=len(uniq))
    xy = np.stack([np.bincount(inverse, weights=sums[:, d], minlength=len(uniq)) for d in (0, 1)], axis=1)
    has_geom = n_vertices > 0
    centroids_um = np.full((len(uniq), 2), np.nan)
    centroids_um[has_geom] = xy[has_geom] / n_vertices[has_geom, None] * units["um_per_unit"]

    if box is None:
        inside = has_geom.copy()
    else:
        inside = (has_geom & (centroids_um[:, 0] >= box[0]) & (centroids_um[:, 0] <= box[2])
                  & (centroids_um[:, 1] >= box[1]) & (centroids_um[:, 1] <= box[3]))
    extent = None
    if has_geom.any():
        lo, hi = centroids_um[has_geom].min(0), centroids_um[has_geom].max(0)
        extent = [round(float(v), 2) for v in (*lo, *hi)]

    notes = []
    if len(uniq) != len(ids):
        notes.append(f"Zoom {plan['zoom']} stores features split across tiles ({len(ids):,} rows for "
                     f"{len(uniq):,} features); centroids were merged per feature_id.")
    if box is not None and not inside.any() and extent is not None:
        outside = box[2] < extent[0] or box[0] > extent[2] or box[3] < extent[1] or box[1] > extent[3]
        span = f"Feature centroids span x {extent[0]}..{extent[2]}, y {extent[1]}..{extent[3]} µm."
        notes.append(f"The box lies outside the data. {span} bbox_um must be micrometres, not pixels."
                     if outside else
                     f"No centroid falls in the box: that region holds no features (e.g. a gap in the tissue). {span}")

    groups = None
    if group_by:
        labels = np.concatenate(labels_l)[first] if labels_l else np.zeros(0, dtype=object)
        if len(labels) and all(v is None for v in labels):
            raise ValueError(f"tags has no key {group_by!r} in {dataset_id}; available keys: "
                             f"{', '.join(tag_keys) or '(none)'}. Per-cell attributes that are not in tags "
                             "live in the facet store: use query_facets.")
        vals, n = np.unique(np.array(["(none)" if v is None else str(v) for v in labels[inside]], dtype=object),
                            return_counts=True)
        order = np.argsort(-n, kind="stable")
        groups = {str(vals[i]): int(n[i]) for i in order[:MAX_GROUPS]}
        if len(vals) > MAX_GROUPS:
            notes.append(f"groups shows the {MAX_GROUPS} largest of {len(vals)} values.")

    return {"status": "ok", "dataset": dataset_id, "layer": (plan["layers"] or [None])[0],
            "zoom": plan["zoom"], "count": int(inside.sum()), "total_features": int(has_geom.sum()),
            "bbox_um": box, "extent_um": extent, "bounds_um": descriptor.get("mudm:bounds_um"),
            "units": units, "method": "feature counted when its vertex centroid lies inside bbox_um",
            "groups": groups, "note": " ".join(notes) or None}
