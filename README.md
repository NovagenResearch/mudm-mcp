# mudm-mcp

Ask an AI assistant about the [muDM](https://ngrf.mudm.io) spatial-data catalog: about 100 public datasets of
spatial transcriptomics and proteomics (Xenium, CODEX, CosMx, MERSCOPE, MIBI, Visium), *Drosophila* EM connectomes
and 3D reference anatomy.

`mudm-mcp` is a small [Model Context Protocol](https://modelcontextprotocol.io) server that runs **on your own
machine**. Your assistant (Claude Code, Claude Desktop, Cursor, …) starts it, and through it can find datasets,
describe them, query per-cell and per-neuron attributes, count cells in a region, and download data for local
analysis. There is no account, no API key and no hosted backend: it reads the public static catalog.

## Quickstart

You need [uv](https://docs.astral.sh/uv/getting-started/installation/).

**Claude Code**

```bash
claude mcp add mudm -- uvx mudm-mcp
```

**Claude Desktop** — add this to `claude_desktop_config.json` (Settings → Developer → Edit Config) and restart:

```json
{
  "mcpServers": {
    "mudm": { "command": "uvx", "args": ["mudm-mcp"] }
  }
}
```

If Claude Desktop reports that it cannot find `uvx`, give its full path instead (run `which uvx`; typically
`/Users/you/.local/bin/uvx`).

**Any other MCP host** — register a stdio server with the command `uvx mudm-mcp`.

Then ask, for example:

- *"Find a mouse brain spatial transcriptomics dataset in muDM and describe it."*
- *"How many cells are in the 500 µm square from (2000, 2000) to (2500, 2500) of that section?"*
- *"How many Kenyon cells are in the hemibrain connectome, by type?"*
- *"Which genes have the most transcripts in the Xenium mouse brain coronal section?"*

## Tools

| Tool | What it does | Downloads data? |
|---|---|---|
| `search_datasets` | Find datasets by free text (`"mouse brain"`) or by modality, organism, technique, tissue | no |
| `describe_dataset` | What a dataset contains: assay, license, citation, layers, columns, attributes, units, sizes per zoom | no |
| `query_facets` | SQL over the per-cell / per-neuron attribute table (gene counts, markers, cell types), read in place | no |
| `count_features` | Count cells inside a rectangle given in micrometres, optionally grouped by a tag (2D datasets) | a few MB, once |
| `get_access` | URLs and a decode snippet for reading the data directly, outside this server | no |
| `download` | Fetch one zoom level of the feature Parquet into the local cache | yes (size-gated) |
| `query` | SQL (DuckDB) over the downloaded features | no |
| `sample_patches` | Sample tiles as spatial patches, with per-feature geometry summaries (for ML) | no |
| `get_lod` | Summarise one level of detail of the downloaded data | no |

## Things worth knowing

**Units.** Datasets store their vertices in different units: Xenium in image pixels (0.2125 µm each), the other
2D platforms in micrometres, the connectomes in nanometres, the reference anatomy in metres. `describe_dataset`
reports the stored unit and the factor to micrometres, and every tool argument or result named `*_um` is in
micrometres, already converted.

**Geometry is not GeoParquet.** Feature geometry is muDM binary tile geometry (packed float32 vertices), so SQL
cannot filter on coordinates. Use `count_features` for regions and `sample_patches` for geometry;
`get_access` returns a three-line Python snippet for decoding it yourself.

**Downloads are size-gated.** A download estimated above 500 MB is not started; the tool reports the size and
cheaper options, and proceeds only with `confirm_large=true`. (The hemibrain at full resolution is about 300 GB.)
Most questions need no download at all: `query_facets` reads the remote file in place.

**SQL is sandboxed and bounded.** `query` and `query_facets` accept one read-only `SELECT`. The DuckDB connection
can read only the dataset it was opened for: no other files or URLs, no writes. Results are capped (200 rows by
default) and flagged `truncated` when more exist.

**Cache.** Downloads and catalog metadata live in `~/.cache/mudm-mcp/` (override with `MUDM_CACHE_DIR` or
`--cache-dir`). Catalog metadata refreshes after 24 hours. Delete the directory to reclaim space.

**Another catalog.** `--base-url` (or `MUDM_BASE_URL`) points the server at a different muDM static catalog,
including a local build directory.

## Development

```bash
uv sync
uv run pytest tests/ -x --tb=short      # offline tests
uv run pytest -m live                    # contract tests against the live catalog
uv run python scripts/host_smoke.py      # drive the server over stdio, end to end
```

## Part of the muDM family

- [`mudm`](https://github.com/NovagenResearch/mudm) — the data model
- [`mudm-tools`](https://github.com/NovagenResearch/mudm-tools) — the tiling/conversion engine and viewers
- `mudm-data` — produces and serves the static catalog this client reads

## License

MIT. The datasets carry their own licenses and citations; `describe_dataset` reports them.
