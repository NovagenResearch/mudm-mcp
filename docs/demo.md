# Demo script

A warm-up and two short flows that show an AI assistant discovering and querying muDM data through `mudm-mcp`.
It takes about ten minutes. You type the prompts; the assistant picks the tools. The wording of its answers
varies from run to run, but the numbers below should not: they were measured against the live catalog
(`https://ngrf.mudm.io`) on 2026-11-09 and re-checked on 2026-11-10.

## Before you start

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
   [Claude Code](https://claude.com/claude-code). You need an internet connection; no account or key for muDM.
2. Register the server (once): `claude mcp add mudm -- uvx mudm-mcp`.
   From a source checkout instead: `claude mcp add mudm -- uv run --directory /path/to/mudm-mcp mudm-mcp`.
3. Start a new Claude Code session with `claude` and type `/mcp`. `mudm` should be listed as connected.
   The very first launch downloads the dependencies and can take a few seconds longer.
4. Type the prompts below, in order, in that one session (the Flow A prompts build on each other). The first
   time the assistant uses a tool, Claude Code asks for permission: allow it.

## Warm-up — "I'm new here"

**Prompt:** *I'm new to muDM. What kinds of data does it have, and which human breast cancer dataset has the most cells?*

- Expected calls: `search_datasets()` then `search_datasets(organism="human", tissue="breast", sort="size")`
- Result: 102 datasets (94 2D spatial omics, 8 3D); 75 human, 21 mouse, 6 *Drosophila*. The largest human breast
  dataset is `xenium-breast-cancer-prime5k` with 712,682 cells.

## Flow A — find a dataset, describe it, count cells in a region

**Prompt 1:** *Find a mouse brain spatial transcriptomics dataset in muDM.*

- Expected call: `search_datasets(q="mouse brain")`
- Result: 15 datasets, among them `xenium-mouse-brain-coronal`, `merscope-mouse-brain`, `visium-hd-mouse-brain`.

**Prompt 2:** *Describe the Xenium mouse brain coronal section.*

- Expected call: `describe_dataset("xenium-mouse-brain-coronal")` (about 2 s)
- Points to bring out: 131,889 cells; layers `cells`, `nuclei`, `transcripts`; vertices stored in pixels,
  0.2125 µm each; an `expression` facet table (cell_id, gene, count); sizes per zoom level.

**Prompt 3:** *How many cells are in the 500 µm square from (2000, 2000) to (2500, 2500)?*

- Expected call: `count_features("xenium-mouse-brain-coronal", bbox_um=[2000, 2000, 2500, 2500])`
- Result: **769** of 131,889 cells (under 1 s; the first call downloads about 10 MB).
- What it shows: the box is in micrometres although the file stores pixels; the tool converts.

**Optional prompt 4:** *Which genes have the most transcripts in that section?*

- Expected call: `query_facets("xenium-mouse-brain-coronal", "SELECT gene, sum(count) AS total FROM facets GROUP BY 1 ORDER BY total DESC LIMIT 5")`
- Result: Slc17a7 1,281,040 · Dner 877,859 · Nrn1 818,995 · Rims3 582,081 · Epha4 580,546. No download.
- Note: the expression table also holds control probes (`BLANK_*` and similar: 293 of its 541 names); none is in the top 5.

## Flow B — a connectome question with no download

**Prompt:** *How many Kenyon cells are in the hemibrain connectome, by type?*

- Expected call: `query_facets("hemibrain", "SELECT cell_type, count(*) AS n FROM facets WHERE cell_type LIKE 'KC%' GROUP BY 1 ORDER BY n DESC")` (under 1 s)
- Result: **1,846** neurons in 14 `KC…` types; the largest are KCg-m 590, KCab-m 342, KCab-s 223.
- Caveat to say aloud: this counts cell types whose name starts with `KC`, which is the hemibrain naming
  convention for Kenyon cells, not a curated list.
- What it shows: 25,000 neurons queried in place over HTTP; nothing is downloaded.

## If someone asks "can I get the whole thing?"

**Prompt:** *Download the hemibrain.*

- Expected call: `download("hemibrain")`
- Result: nothing is downloaded. The tool answers that the coarsest level alone is 1,162 files, about 2.6 GB
  (full resolution is about 300 GB), and suggests `query_facets` instead.

## Known rough edges

- Models sometimes add up a result table themselves and slip by one; ask for totals in the SQL (`sum(...)`).
- Connectome synapse counts (`pre`, `post`) are stored as text in hemibrain, malecns, manc-full and
  malecns-courtship: sort or sum them with `CAST(pre AS BIGINT)`, otherwise "999" outranks "17628".
- Searching for a cell type (e.g. "kenyon") finds no dataset: search covers dataset metadata, not cell types.
- `count_features` works on 2D datasets only; for connectomes use `query_facets`.
- Xenium cells carry no `cell_type` tag, so `count_features(group_by="cell_type")` does not apply there.
- The two Visium datasets have no facet table.

## Afterwards

- Remove the server: `claude mcp remove mudm`.
- Reclaim disk space: delete `~/.cache/mudm-mcp/` (this demo leaves under 20 MB there).

## How these numbers were checked

- `uv run python scripts/host_smoke.py` (from a source checkout) replays both flows over the MCP stdio
  protocol against the live catalog and compares the numbers.
- The prompts were also run as real agent sessions (headless Claude Code, Sonnet and Haiku, only the mudm tools
  available) from an empty home directory: each answered correctly in 2 to 4 tool calls and 5 to 13 seconds.
