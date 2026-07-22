"""Read the static muDM catalog: a CLIENT of `pyramids.json` (mudm.PyramidJSON) + per-dataset
`tilejson.json` (mudm.TileModel) + derived `croissant.json`. No backend — fetches static files over
http(s) (cached to a local dir) or reads a local build dir (`file://` / plain path) for dev/tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse


def default_cache_dir(host: str) -> Path:
    return Path.home() / ".cache" / "mudm-mcp" / (host or "local")


class CatalogClient:
    """Resolve + cache the catalog artifacts. ``base_url`` may be an http(s) origin (default
    ``https://ngrf.mudm.io``) or a local directory (a `build_site --out` dir, as `file://…` or a path)."""

    def __init__(self, base_url: str, cache_dir: str | Path | None = None):
        self.base_url = str(base_url).rstrip("/")
        parsed = urlparse(self.base_url)
        self.is_http = parsed.scheme in ("http", "https")
        if self.is_http:
            self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir(parsed.netloc)
            self._local_root: Path | None = None
        else:  # file:// or a plain local path
            self._local_root = Path(parsed.path if parsed.scheme == "file" else self.base_url)
            self.cache_dir = Path(cache_dir) if cache_dir else self._local_root

    # --- fetch / cache ---
    def fetch(self, *parts: str) -> bytes:
        """Bytes of a catalog-relative path (e.g. ``fetch("datasets", id, "tilejson.json")``)."""
        rel = "/".join(str(p).strip("/") for p in parts if str(p))
        if not self.is_http:
            return (self._local_root / rel).read_bytes()
        cached = self.cache_dir / rel
        if cached.is_file():
            return cached.read_bytes()
        import httpx  # lazy: file:// dev/tests need no network/httpx
        resp = httpx.get(f"{self.base_url}/{rel}", follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(resp.content)
        return resp.content

    def fetch_json(self, *parts: str) -> dict:
        return json.loads(self.fetch(*parts))

    def download_file(self, rel_parts, out_path) -> int:
        """Stream a (possibly large) leaf file to ``out_path`` (http GET or local copy); returns bytes
        written. Unlike ``fetch``, this does not hold the body in memory or use the JSON cache — and the
        parquet leaves are served uncompressed (range-safe), so no Content-Encoding decoding is involved."""
        rel = "/".join(str(p).strip("/") for p in rel_parts if str(p))
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.is_http:
            data = (self._local_root / rel).read_bytes()
            out.write_bytes(data)
            return len(data)
        import httpx  # lazy
        with httpx.stream("GET", f"{self.base_url}/{rel}", follow_redirects=True, timeout=120.0) as resp:
            resp.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        return out.stat().st_size

    # --- catalog accessors ---
    def pyramids(self) -> dict:
        """The unified 2D+3D catalog (mudm.PyramidJSON)."""
        return self.fetch_json("pyramids.json")

    def entries(self) -> list[dict]:
        return self.pyramids().get("pyramids", [])

    def entry(self, dataset_id: str) -> dict:
        """The catalog entry (PyramidEntry: id/label/kind + OPEN facet metadata) for one dataset."""
        for e in self.entries():
            if e.get("id") == dataset_id:
                return e
        raise KeyError(f"dataset {dataset_id!r} not found in pyramids.json")

    def descriptor(self, dataset_id: str) -> dict:
        """The per-dataset muDM TileModel descriptor (assets[] + TileLayer schema + multiscale coords)."""
        return self.fetch_json("datasets", dataset_id, "tilejson.json")

    def croissant(self, dataset_id: str) -> dict:
        """The derived Croissant JSON-LD (schema.org/Dataset + cr:)."""
        return self.fetch_json("datasets", dataset_id, "croissant.json")

    def features_asset(self, descriptor: dict) -> dict | None:
        """The `features` Asset (download index: href + parts/zooms/full_res_zoom) from a descriptor, if any."""
        for a in descriptor.get("assets", []):
            if a.get("role") == "features":
                return a
        return None
