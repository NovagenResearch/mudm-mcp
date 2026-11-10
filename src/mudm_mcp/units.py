"""Which unit a dataset's stored vertices are in, and the factor to micrometres.

The catalog descriptors do not (yet) state the vertex unit reliably: every 2D descriptor says
"micrometre" and carries a raster pixel-size scale, and 3D descriptors carry no transform at all. The
rules below were measured against the live data (stored coordinate extents vs. the declared micrometre
bounds) and are a client-side stopgap; an explicit ``mudm:vertex_unit`` / ``mudm:um_per_unit`` pair in the
descriptor always wins.
"""
from __future__ import annotations

UNKNOWN = "unknown: assumed micrometres, verify against the dataset bounds"


def _scale(descriptor: dict) -> float | None:
    for t in (descriptor.get("multiscale") or {}).get("coordinateTransformations") or []:
        if t.get("type") == "scale" and t.get("scale"):
            return float(t["scale"][0])
    return None


def resolve_units(entry: dict, descriptor: dict) -> dict:
    """``{"vertex_unit": "pixel"|"micrometre"|"nanometre"|"metre", "um_per_unit": float, "basis": str}``.
    Multiply a stored coordinate by ``um_per_unit`` to get micrometres. ``basis`` names the rule that
    fired and starts with ``"unknown"`` when none did."""
    unit, factor = descriptor.get("mudm:vertex_unit"), descriptor.get("mudm:um_per_unit")
    if unit and factor is not None:
        return {"vertex_unit": str(unit), "um_per_unit": float(factor), "basis": "stated by the descriptor"}
    kind = descriptor.get("mudm:kind") or entry.get("kind")
    technique = str(entry.get("technique") or "")
    if kind == "2d":
        if descriptor.get("mudm:platform") == "xenium":
            scale = _scale(descriptor)
            if scale is None:
                return {"vertex_unit": "micrometre", "um_per_unit": 1.0, "basis": UNKNOWN}
            return {"vertex_unit": "pixel", "um_per_unit": scale,
                    "basis": "Xenium: vertices are image pixels; scale is the descriptor's pixel size"}
        return {"vertex_unit": "micrometre", "um_per_unit": 1.0,
                "basis": "non-Xenium 2D: vertices are already micrometres"}
    if kind == "3d":
        if technique == "EM connectomics":
            return {"vertex_unit": "nanometre", "um_per_unit": 0.001,
                    "basis": "EM connectome: vertices are nanometres"}
        if technique == "3D reference anatomy":
            return {"vertex_unit": "metre", "um_per_unit": 1e6,
                    "basis": "3D reference anatomy: vertices are metres"}
    return {"vertex_unit": "micrometre", "um_per_unit": 1.0, "basis": UNKNOWN}
