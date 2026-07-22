"""Decode muDM binary tile geometry (the `features.parquet` `positions`/`indices`/`ring_lengths` columns).

This is muDM's OWN binary layout — **NOT GeoParquet/WKB**, so `ST_*`/`shapely.wkb` do not apply. Verified
against real `features.parquet` leaves (2026-06-23):
  - `positions` = packed little-endian float32, vertices interleaved `[x,y(,z)…]` (image-pixel µm, NOT lon/lat).
  - `indices`   = packed little-endian uint32 triangle indices — present only for mesh types; EMPTY for
                  point/line/polygon.
  - `ring_lengths` = per-ring vertex counts (polygons/lines); EMPTY for points.
  - point  `geom_type=1`: 8 bytes → 1 vertex, no rings.   polygon `geom_type=3`: packed verts + ring_lengths.
"""
from __future__ import annotations

import numpy as np

# OGC-style geometry-type codes used by muDM tile geometry.
POINT, LINESTRING, POLYGON = 1, 2, 3


def decode_geometry(
    positions: bytes,
    indices: bytes,
    ring_lengths,
    geom_type: int,
    dims: int | None = None,
) -> dict:
    """Decode one muDM tile-geometry row into NumPy arrays (zero-copy via ``frombuffer``).

    Args:
        positions: packed little-endian float32 vertex coordinates (`[x,y(,z)…]`).
        indices: packed little-endian uint32 triangle indices (mesh types; ``b""`` otherwise).
        ring_lengths: per-ring vertex counts (polygons/lines); empty/None for points.
        geom_type: OGC-style code (1=Point, 2=LineString, 3=Polygon; mesh types carry ``indices``).
        dims: 2 or 3. If None, inferred from ``ring_lengths`` (or a single-vertex point), defaulting to 2.

    Returns:
        ``{geom_type, dims, coords: float32[N,dims], rings?: [float32[ni,dims]], triangles?: uint32[T,3]}``.
    """
    floats = np.frombuffer(positions, dtype="<f4")
    rings = [int(n) for n in (ring_lengths or [])]
    if dims is None:
        total = sum(rings)
        if total:
            dims = len(floats) // total
        elif geom_type == POINT and len(floats):
            dims = len(floats)  # a single vertex
        else:
            dims = 2
    dims = int(dims) or 2
    coords = floats.reshape(-1, dims)
    out: dict = {"geom_type": int(geom_type), "dims": dims, "coords": coords}
    if rings:
        bounds = np.cumsum([0, *rings])
        out["rings"] = [coords[bounds[i]:bounds[i + 1]] for i in range(len(rings))]
    if indices:
        out["triangles"] = np.frombuffer(indices, dtype="<u4").reshape(-1, 3)
    return out
