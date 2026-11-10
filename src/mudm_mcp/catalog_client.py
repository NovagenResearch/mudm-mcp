"""Read the static muDM catalog: a CLIENT of `pyramids.json` (mudm.PyramidJSON) + per-dataset
`tilejson.json` (mudm.TileModel) + derived `croissant.json`. No backend — fetches static files over
http(s) (cached to a local dir) or reads a local build dir (`file://` / plain path) for dev/tests.
"""
from __future__ import annotations

import difflib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_JSON_TTL_S = 24 * 3600.0


def default_cache_dir(host: str) -> Path:
    return Path.home() / ".cache" / "mudm-mcp" / (host or "local")


class DatasetNotFound(KeyError):
    """Unknown dataset id; ``suggestions`` holds up to 5 close catalog ids."""

    def __init__(self, dataset_id: str, known_ids: list[str]):
        super().__init__(f"dataset {dataset_id!r} not found in pyramids.json")
        self.dataset_id = dataset_id
        self.suggestions = difflib.get_close_matches(dataset_id, known_ids, n=5, cutoff=0.5)


def _zoom_of(part: str) -> int:
    """'zoom=9/cells_part_000.mu.parquet' -> 9."""
    return int(part.split("/", 1)[0].split("=", 1)[1])


class CatalogClient:
    """Resolve + cache the catalog artifacts. ``base_url`` may be an http(s) origin (default
    ``https://ngrf.mudm.io``) or a local directory (a `build_site --out` dir, as `file://…` or a path).

    Cached catalog JSON is refetched once it is older than ``json_ttl_s`` (default 24 h), so a long-lived
    install follows catalog updates; if the refetch fails (offline) the stale copy is served."""

    def __init__(self, base_url: str, cache_dir: str | Path | None = None,
                 json_ttl_s: float = DEFAULT_JSON_TTL_S):
        self.base_url = str(base_url).rstrip("/")
        self.json_ttl_s = float(json_ttl_s)
        parsed = urlparse(self.base_url)
        self.is_http = parsed.scheme in ("http", "https")
        if self.is_http:
            self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir(parsed.netloc)
            self._local_root: Path | None = None
        else:  # file:// or a plain local path
            self._local_root = Path(parsed.path if parsed.scheme == "file" else self.base_url)
            self.cache_dir = Path(cache_dir) if cache_dir else self._local_root

    # --- paths ---
    @staticmethod
    def _rel(parts) -> str:
        return "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))

    def url(self, *parts: str) -> str:
        """Absolute URL (http origin) or local path (local origin) of a site-relative path."""
        rel = self._rel(parts)
        return f"{self.base_url}/{rel}" if self.is_http else str(self._local_root / rel)

    # --- fetch / cache ---
    def fetch(self, *parts: str) -> bytes:
        """Bytes of a catalog-relative path (e.g. ``fetch("datasets", id, "tilejson.json")``)."""
        rel = self._rel(parts)
        if not self.is_http:
            return (self._local_root / rel).read_bytes()
        cached = self.cache_dir / rel
        have = cached.is_file()
        if have and (time.time() - cached.stat().st_mtime) < self.json_ttl_s:
            return cached.read_bytes()
        import httpx  # lazy: file:// dev/tests need no network/httpx
        try:
            resp = httpx.get(f"{self.base_url}/{rel}", follow_redirects=True, timeout=60.0)
            resp.raise_for_status()
        except httpx.HTTPError:
            if have:  # expired but unreachable: a stale catalog beats no catalog
                return cached.read_bytes()
            raise
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(cached.name + ".part")
        tmp.write_bytes(resp.content)
        os.replace(tmp, cached)
        return resp.content

    def fetch_json(self, *parts: str) -> dict:
        return json.loads(self.fetch(*parts))

    def _try_fetch_json(self, *parts: str) -> dict | None:
        """``fetch_json`` that returns None when the file does not exist (local miss / HTTP 4xx)."""
        try:
            return self.fetch_json(*parts)
        except FileNotFoundError:
            return None
        except Exception as exc:  # httpx.HTTPStatusError without importing httpx eagerly
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500:  # S3/CloudFront answer 403 for a missing key
                return None
            raise

    def download_file(self, rel_parts, out_path) -> int:
        """Stream a (possibly large) leaf file to ``out_path`` (http GET or local copy); returns bytes
        written. Atomic: the data goes to ``<out_path>.part`` and is renamed only when complete, so an
        interrupted download never leaves a truncated file under the final name. Unlike ``fetch`` this
        does not hold the body in memory or use the JSON cache (parquet leaves are served uncompressed)."""
        rel = self._rel(rel_parts)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        try:
            if not self.is_http:
                with open(self._local_root / rel, "rb") as src, open(tmp, "wb") as fh:
                    while chunk := src.read(1 << 20):
                        fh.write(chunk)
            else:
                import httpx  # lazy
                with httpx.stream("GET", f"{self.base_url}/{rel}", follow_redirects=True,
                                  timeout=120.0) as resp:
                    resp.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_bytes():
                            fh.write(chunk)
            size = tmp.stat().st_size
            os.replace(tmp, out)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return size

    # --- sizes ---
    def part_size(self, base: str, part: str) -> int | None:
        """Size in bytes of one leaf (local ``stat``; HTTP ``HEAD``, else a 1-byte ranged GET). None if
        it cannot be determined."""
        rel = self._rel([base, part])
        if not self.is_http:
            p = self._local_root / rel
            return p.stat().st_size if p.is_file() else None
        import httpx  # lazy
        url = f"{self.base_url}/{rel}"
        try:
            r = httpx.head(url, follow_redirects=True, timeout=30.0)
            if r.status_code == 200 and "content-length" in r.headers:
                return int(r.headers["content-length"])
            r = httpx.get(url, headers={"Range": "bytes=0-0"}, follow_redirects=True, timeout=30.0)
            total = r.headers.get("content-range", "").rsplit("/", 1)[-1]
            return int(total) if total.isdigit() else None
        except httpx.HTTPError:
            return None

    def estimate_bytes(self, base: str, parts: list[str], sample: int = 4) -> int | None:
        """Total size of ``parts``: exact when there are at most ``sample`` of them, else the mean of
        ``sample`` evenly spaced parts times the count. None if no size could be read."""
        if not parts:
            return 0
        if len(parts) <= sample:
            sizes = [self.part_size(base, p) for p in parts]
            return None if any(s is None for s in sizes) else sum(sizes)
        step = len(parts) / sample
        sizes = [s for s in (self.part_size(base, parts[int(i * step)]) for i in range(sample)) if s is not None]
        return round(sum(sizes) / len(sizes) * len(parts)) if sizes else None

    # --- catalog accessors ---
    def pyramids(self) -> dict:
        """The unified 2D+3D catalog (mudm.PyramidJSON)."""
        return self.fetch_json("pyramids.json")

    def entries(self) -> list[dict]:
        return self.pyramids().get("pyramids", [])

    def entry(self, dataset_id: str) -> dict:
        """The catalog entry (PyramidEntry: id/label/kind + OPEN facet metadata) for one dataset."""
        entries = self.entries()
        for e in entries:
            if e.get("id") == dataset_id:
                return e
        raise DatasetNotFound(dataset_id, [str(e.get("id")) for e in entries])

    def descriptor(self, dataset_id: str) -> dict:
        """The per-dataset muDM TileModel descriptor (assets[] + TileLayer schema + multiscale coords)."""
        return self.fetch_json("datasets", dataset_id, "tilejson.json")

    def croissant(self, dataset_id: str) -> dict:
        """The derived Croissant JSON-LD (schema.org/Dataset + cr:)."""
        return self.fetch_json("datasets", dataset_id, "croissant.json")

    def viewer_descriptor(self, dataset_id: str) -> tuple[str, dict] | None:
        """The VIEWER's descriptor ``(<site-relative dir>, tilejson)`` that sits next to the tiles
        (``tiles/<id>/`` for 3D, ``tiles2d/<id>/`` for 2D). Its asset hrefs are relative to that dir. Used
        as a fallback when the agent descriptor (``datasets/<id>/tilejson.json``) omits an asset."""
        entry = self.entry(dataset_id)
        feats = (entry.get("features") or "").strip("/")
        dirs = [feats.rsplit("/", 1)[0]] if "/" in feats else []
        dirs += [d for d in (f"tiles/{dataset_id}", f"tiles2d/{dataset_id}") if d not in dirs]
        for d in dirs:
            doc = self._try_fetch_json(d, "tilejson.json")
            if doc is not None:
                return d, doc
        return None

    @staticmethod
    def _asset(descriptor: dict, role: str) -> dict | None:
        return next((a for a in descriptor.get("assets") or [] if a.get("role") == role), None)

    def features(self, dataset_id: str) -> dict | None:
        """The download index of a dataset's ``features.parquet``:
        ``{base, parts, zooms, full_res_zoom, source}`` — ``base`` is the site-relative directory holding
        the ``zoom=<z>/…`` leaves, ``source`` is ``"agent"`` or ``"viewer"`` (which descriptor carried it).
        None when neither descriptor lists parquet parts."""
        entry = self.entry(dataset_id)
        entry_base = (entry.get("features") or "").strip("/")
        asset = self._asset(self.descriptor(dataset_id), "features")
        source, viewer_dir = "agent", None
        if not (asset and asset.get("parts")):
            found = self.viewer_descriptor(dataset_id)
            asset = self._asset(found[1], "features") if found else None
            if not (asset and asset.get("parts")):
                return None
            source, viewer_dir = "viewer", found[0]
        href = str(asset.get("href") or "")
        if entry_base:
            base = entry_base
        elif viewer_dir is not None:
            base = self._rel([viewer_dir, href or "features.parquet"])
        elif href.startswith(self.base_url + "/"):
            base = href[len(self.base_url) + 1:].strip("/")
        else:
            raise ValueError(f"{dataset_id}: cannot locate features.parquet (href {asset.get('href')!r} is "
                             f"outside {self.base_url})")
        parts = list(asset["parts"])
        zooms = sorted(int(z) for z in (asset.get("zooms") or {_zoom_of(p) for p in parts}))
        full = asset.get("full_res_zoom")
        return {"base": base, "parts": parts, "zooms": zooms,
                "full_res_zoom": int(full) if full is not None else zooms[-1], "source": source}
