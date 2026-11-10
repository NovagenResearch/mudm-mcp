"""Contract checks against the LIVE catalog (https://ngrf.mudm.io). Deselected by default:

    uv run pytest -m live
"""
import json

import pytest

from mudm_mcp import tools
from mudm_mcp.catalog_client import CatalogClient
from mudm_mcp.units import resolve_units

pytestmark = pytest.mark.live
BUDGET = 20_000
BIG = ["hemibrain", "banc", "manc-full", "malecns-courtship", "xenium-mouse-brain-coronal", "codex-bone-marrow"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return CatalogClient("https://ngrf.mudm.io", tmp_path_factory.mktemp("live-cache"))


def test_catalog_is_listed_and_search_is_bounded(client):
    r = tools.search_datasets(client)
    assert r["count"] >= 100 and r["returned"] == 25 and len(json.dumps(r)) <= BUDGET
    assert len(json.dumps(tools.search_datasets(client, limit=100))) <= 2 * BUDGET   # explicit opt-in to more
    assert tools.search_datasets(client, q="mouse brain")["count"] >= 5
    assert tools.search_datasets(client, q="human breast")["count"] >= 5


@pytest.mark.parametrize("dataset", BIG)
def test_describe_and_access_fit_the_budget(client, dataset):
    d = tools.describe_dataset(client, dataset)
    a = tools.get_access(client, dataset)
    assert len(json.dumps(d)) <= BUDGET and len(json.dumps(a)) <= BUDGET
    assert d["parquet_columns"] and d["lod"] and all(z["est_bytes"] for z in d["lod"])
    assert a["features"]["zooms"] and "parts" not in a["features"]


def test_every_dataset_has_features_and_a_known_unit(client):
    unknown, no_features = [], []
    for e in client.entries():
        if resolve_units(e, client.descriptor(e["id"]))["basis"].startswith("unknown"):
            unknown.append(e["id"])
        if client.features(e["id"]) is None:
            no_features.append(e["id"])
    assert not unknown, f"no unit rule for: {unknown}"
    assert not no_features, f"no features.parquet index for: {no_features}"


def test_viewer_descriptor_fallback(client):
    for dataset in ("manc-full", "malecns-courtship"):
        assert client.features(dataset)["source"] in ("viewer", "agent")     # "agent" once mudm-data fixes it
        assert tools.get_access(client, dataset)["features"]["full_res_zoom"] == 4


def test_large_downloads_are_gated(client):
    for kwargs in ({}, {"zoom": "max"}):
        r = tools.download(client, "hemibrain", **kwargs)
        assert r["status"] == "needs_confirmation" and r["est_bytes"] > 500_000_000
    assert not (client.cache_dir / "hemibrain" / "features").exists()
