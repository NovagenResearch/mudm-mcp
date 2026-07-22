"""decode_geometry on synthetic muDM buffers (layout verified against real features.parquet 2026-06-23)."""
import struct

import numpy as np

from mudm_mcp.geometry import decode_geometry


def _f32(*vals):
    return struct.pack("<%df" % len(vals), *vals)


def test_decode_point_2d():
    # geom_type=1: 8 bytes → one [x,y] vertex, no rings, no indices.
    g = decode_geometry(_f32(38530.0, 9346.0), b"", [], geom_type=1)
    assert g["geom_type"] == 1 and g["dims"] == 2
    np.testing.assert_allclose(g["coords"], [[38530.0, 9346.0]])
    assert "rings" not in g and "triangles" not in g


def test_decode_polygon_rings_2d():
    verts = [(0, 0), (10, 0), (10, 10), (0, 10), (2, 2), (4, 2), (2, 4)]  # outer 4 + hole 3
    buf = _f32(*[c for xy in verts for c in xy])
    g = decode_geometry(buf, b"", [4, 3], geom_type=3)
    assert g["coords"].shape == (7, 2)
    assert [r.shape for r in g["rings"]] == [(4, 2), (3, 2)]
    np.testing.assert_allclose(g["rings"][0][1], [10, 0])
    np.testing.assert_allclose(g["rings"][1][0], [2, 2])


def test_decode_mesh_triangles_3d():
    verts = _f32(0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1)  # 4 xyz verts
    idx = struct.pack("<6I", 0, 1, 2, 1, 2, 3)         # 2 triangles
    g = decode_geometry(verts, idx, [], geom_type=7, dims=3)
    assert g["dims"] == 3 and g["coords"].shape == (4, 3)
    assert g["triangles"].shape == (2, 3)
    np.testing.assert_array_equal(g["triangles"][1], [1, 2, 3])


def test_decode_infers_dims_from_rings():
    buf = _f32(0, 0, 1, 0, 1, 1)  # 3 verts, 2D; dims not passed
    g = decode_geometry(buf, b"", [3], geom_type=3)
    assert g["dims"] == 2 and g["coords"].shape == (3, 2)


def test_decode_empty_positions_is_safe():
    g = decode_geometry(b"", b"", [], geom_type=1)
    assert g["coords"].shape == (0, 2)
