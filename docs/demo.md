# Demo script

Two short flows that show an agent discovering and querying muDM data through `mudm-mcp`. The expected
tool calls and numbers below were measured against the live catalog (`https://ngrf.mudm.io`) on 2026-11-09;
`uv run python scripts/host_smoke.py` re-checks them.

Setup (once): `claude mcp add mudm -- uvx mudm-mcp`, then start a new Claude Code session.

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

- Searching for a cell type (e.g. "kenyon") finds no dataset: search covers dataset metadata, not cell types.
- `count_features` works on 2D datasets only; for connectomes use `query_facets`.
- Xenium cells carry no `cell_type` tag, so `count_features(group_by="cell_type")` does not apply there.
- The two Visium datasets have no facet table.
