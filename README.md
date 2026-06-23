# mudm-mcp

A **local [Model Context Protocol](https://modelcontextprotocol.io) server** that lets an LLM/agent **discover,
access, and locally analyze** the [muDM](https://github.com/NovagenResearch/mudm) public spatial-data catalog
(spatial transcriptomics/proteomics + 3D connectomes), served as static tiles on S3/CloudFront.

It runs **on your machine**, launched by your MCP host (Claude Desktop/Code, Cursor, …). It is a **client of the
static catalog** — it reads the public catalog, downloads the data you ask for, and analyzes it **locally** with
DuckDB/PyArrow. There is **no hosted backend**.

> Status: **early / scaffolding.** The tool surface below is the target design; see the development notes for
> what's implemented.

## What it does

| Tool | Purpose |
|---|---|
| `list_datasets` | browse/search the catalog |
| `describe_dataset` | what a dataset contains — modality, assay, organism, feature layers, column schema, coordinate model |
| `get_access` | how to access it — URLs, the download manifest, geometry-decoding notes |
| `download` | fetch a dataset's full-resolution tiles (or a region) locally |
| `query` | run SQL (DuckDB) over the downloaded tiles |
| `sample_patches` / `get_lod` | sample spatial patches / a chosen level-of-detail for ML |

## Install & configure

```bash
uv tool install mudm-mcp        # or: pip install mudm-mcp
```

Then register it in your MCP host (example, stdio):

```json
{
  "mcpServers": {
    "mudm": { "command": "mudm-mcp", "args": ["--base-url", "https://ngrf.mudm.io"] }
  }
}
```

## Data model notes

Datasets are muDM tile pyramids. Coordinates are **image-pixel `[X, Y, Z]` in micrometres**, *not* longitude/
latitude. The feature Parquet is **muDM binary tile geometry (not GeoParquet)**; `mudm-mcp` handles decoding and
de-duplication (the full-resolution data is read once, not once per zoom level).

## Part of the muDM family

- [`mudm`](https://github.com/NovagenResearch/mudm) — the data model
- [`mudm-tools`](https://github.com/NovagenResearch/mudm-tools) — the tiling/conversion engine + viewers
- `mudm-data` — produces & serves the static catalog this client reads

## License

MIT.
