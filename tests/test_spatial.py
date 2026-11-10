"""count_features: micrometre boxes over stored-pixel geometry (offline `site` fixture)."""
import struct

import numpy as np
import pyarrow as pa
import pytest

from mudm_mcp.catalog_client import CatalogClient
from mudm_mcp.spatial import _centroid_sums, count_features


def _c(site, tmp_path):
    return CatalogClient(site, cache_dir=tmp_path / "cache")


def test_count_converts_pixels_to_um(site, tmp_path):
    r = count_features(_c(site, tmp_path), "demo2d", bbox_um=[0, 0, 50, 50])
    # cell centroids: (21.25, 21.25), (42.5, 21.25), (212.5, 212.5), (425, 425) µm. Read as µm, the stored
    # pixel values (100, 100) … would put NO cell in this box.
    assert r["status"] == "ok" and r["count"] == 2 and r["total_features"] == 4
    assert r["layer"] == "cells" and r["zoom"] == 0 and r["units"]["um_per_unit"] == 0.2125
    assert r["extent_um"] == [21.25, 21.25, 425.0, 425.0] and r["bounds_um"] == [0, 0, 500, 500]
    assert r["note"] is None and r["groups"] is None


def test_count_everything_and_box_edges_are_inclusive(site, tmp_path):
    c = _c(site, tmp_path)
    assert count_features(c, "demo2d")["count"] == 4
    assert count_features(c, "demo2d", bbox_um=[21.25, 21.25, 42.5, 21.25])["count"] == 2
    assert count_features(c, "demo2d", bbox_um=[200, 200, 450, 450])["count"] == 2


def test_count_merges_features_split_across_tiles(site, tmp_path):
    r = count_features(_c(site, tmp_path), "demo2d", zoom=1)      # zoom 1: cell 3 is stored as two halves
    assert r["count"] == 4 and r["total_features"] == 4 and "5 rows for 4 features" in r["note"]
    assert count_features(_c(site, tmp_path), "demo2d", zoom=1, bbox_um=[212, 212, 213, 213])["count"] == 1


def test_count_other_layer(site, tmp_path):
    r = count_features(_c(site, tmp_path), "demo2d", layer="nuclei", zoom=1, bbox_um=[0, 0, 50, 50])
    assert r["layer"] == "nuclei" and r["count"] == 2
    with pytest.raises(ValueError, match="available: cells"):
        count_features(_c(site, tmp_path), "demo2d", layer="nuclei")          # zoom 0 has cells only


def test_count_group_by_tag(site, tmp_path):
    c = _c(site, tmp_path)
    assert count_features(c, "demo2d", group_by="cell_type")["groups"] == {"A": 2, "B": 2}
    assert count_features(c, "demo2d", group_by="cell_type", bbox_um=[0, 0, 300, 300])["groups"] == {"A": 2, "B": 1}
    assert count_features(c, "demo2d", group_by="cell_type", zoom=1)["groups"] == {"A": 2, "B": 2}
    with pytest.raises(ValueError, match=r"no key 'celltype'.*available keys: cell_type.*query_facets"):
        count_features(c, "demo2d", group_by="celltype")


@pytest.mark.parametrize("bbox", [[0, 0, 50], [50, 0, 0, 50], [0, 50, 50, 0], [0, 0, float("nan"), 5],
                                  [0, 0, 0, 1, 1, 1], "0,0,5,5", ["a", 0, 1, 1], []])
def test_count_rejects_bad_bbox(site, tmp_path, bbox):
    with pytest.raises(ValueError, match="bbox_um must be"):
        count_features(_c(site, tmp_path), "demo2d", bbox_um=bbox)
    assert not (tmp_path / "cache").exists()                     # rejected before anything is fetched


def test_count_outside_bounds_explains(site, tmp_path):
    r = count_features(_c(site, tmp_path), "demo2d", bbox_um=[9000, 9000, 9500, 9500])
    assert r["count"] == 0 and "outside the data" in r["note"] and "not pixels" in r["note"]
    r = count_features(_c(site, tmp_path), "demo2d", bbox_um=[100, 100, 110, 110])
    assert r["count"] == 0 and "No centroid falls in the box" in r["note"] and "pixels" not in r["note"]


def test_count_respects_download_gate(site, tmp_path):
    c = _c(site, tmp_path)
    r = count_features(c, "demo2d", max_bytes=10)
    assert r["status"] == "needs_confirmation" and not (tmp_path / "cache").exists()
    assert count_features(c, "demo2d", max_bytes=10, confirm_large=True)["count"] == 4


def test_count_rejects_3d_and_points_to_facets(site, tmp_path):
    with pytest.raises(ValueError, match="2D datasets only.*query_facets"):
        count_features(_c(site, tmp_path), "demo3d", bbox_um=[0, 0, 1, 1])


@pytest.mark.parametrize("typ", [pa.large_binary(), pa.binary()])
def test_centroid_sums_handles_empty_null_sliced_and_ragged(typ):
    tri = struct.pack("<6f", 0, 0, 3, 0, 0, 3)
    pt = struct.pack("<2f", 5, 7)
    arr = pa.array([tri, b"", None, pt, tri], typ)
    sums, counts = _centroid_sums(arr, 2)
    assert counts.tolist() == [3, 0, 0, 1, 3]
    assert sums.tolist() == [[3, 3], [0, 0], [0, 0], [5, 7], [3, 3]]
    s2, c2 = _centroid_sums(arr.slice(3), 2)                     # sliced arrays keep a buffer offset
    assert c2.tolist() == [1, 3] and s2.tolist() == [[5, 7], [3, 3]]
    ragged = pa.array([tri + b"\x00\x00", pt], typ)               # 2 stray bytes after the last whole vertex
    s3, c3 = _centroid_sums(ragged, 2)
    assert c3.tolist() == [3, 1] and s3.tolist() == [[3, 3], [5, 7]]
    assert _centroid_sums(pa.array([], typ), 2)[1].tolist() == []
    assert np.all(_centroid_sums(pa.array([None, b""], typ), 2)[0] == 0)


@pytest.mark.live
@pytest.mark.parametrize("dataset", ["xenium-mouse-ileum", "cosmx-melanoma", "mibi-uterus", "visium-ffpe-breast"])
def test_live_units_calibration(dataset, tmp_path):
    """Coarsest zoom holds every catalogued feature once, and centroids (in µm) fall inside bounds_um."""
    c = CatalogClient("https://ngrf.mudm.io", tmp_path)
    r = count_features(c, dataset)
    assert r["zoom"] == 0 and r["note"] is None
    assert r["count"] == r["total_features"] == c.entry(dataset)["feature_count"]
    (x0, y0, x1, y1), (bx0, by0, bx1, by1) = r["extent_um"], r["bounds_um"]
    assert bx0 - 1 <= x0 and by0 - 1 <= y0 and x1 <= bx1 + 1 and y1 <= by1 + 1
    assert (x1 - x0) > 0.5 * (bx1 - bx0)          # and they fill it: the unit factor is not too small
