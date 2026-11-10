"""Per-dataset vertex units (client-side rules; see mudm_mcp.units)."""
import pytest

from mudm_mcp.units import resolve_units


def _d2(platform, scale):
    return {"mudm:kind": "2d", "mudm:platform": platform,
            "multiscale": {"coordinateTransformations": [{"type": "scale", "scale": [scale, scale]}]}}


@pytest.mark.parametrize("entry, descriptor, unit, factor", [
    ({"kind": "2d", "technique": "Xenium"}, _d2("xenium", 0.2125), "pixel", 0.2125),
    ({"kind": "2d", "technique": "Xenium Prime"}, _d2("xenium", 0.2125), "pixel", 0.2125),
    ({"kind": "2d", "technique": "CODEX/PhenoCycler"}, _d2("codex", 0.3774), "micrometre", 1.0),
    ({"kind": "2d", "technique": "Visium"}, _d2("visium", 4.0034), "micrometre", 1.0),
    ({"kind": "2d", "technique": "MERSCOPE/MERFISH"}, _d2("merscope", 0.1086), "micrometre", 1.0),
    ({"kind": "3d", "technique": "EM connectomics"}, {"mudm:kind": "3d"}, "nanometre", 0.001),
    ({"kind": "3d", "technique": "3D reference anatomy"}, {"mudm:kind": "3d"}, "metre", 1e6),
])
def test_live_groups(entry, descriptor, unit, factor):
    u = resolve_units(entry, descriptor)
    assert (u["vertex_unit"], u["um_per_unit"]) == (unit, factor)
    assert not u["basis"].startswith("unknown")


def test_kind_falls_back_to_entry_when_descriptor_lacks_it():
    u = resolve_units({"kind": "3d", "technique": "EM connectomics"}, {})
    assert u["vertex_unit"] == "nanometre"


def test_explicit_descriptor_unit_wins():
    d = {**_d2("xenium", 0.2125), "mudm:vertex_unit": "micrometre", "mudm:um_per_unit": 1.0}
    u = resolve_units({"kind": "2d"}, d)
    assert (u["vertex_unit"], u["um_per_unit"]) == ("micrometre", 1.0) and "descriptor" in u["basis"]


def test_xenium_without_scale_is_flagged_unknown():
    u = resolve_units({"kind": "2d"}, {"mudm:kind": "2d", "mudm:platform": "xenium"})
    assert u["basis"].startswith("unknown")


def test_unknown_3d_technique_is_flagged():
    u = resolve_units({"kind": "3d", "technique": "Light-sheet"}, {"mudm:kind": "3d"})
    assert (u["vertex_unit"], u["um_per_unit"]) == ("micrometre", 1.0) and u["basis"].startswith("unknown")
