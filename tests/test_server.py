"""Server wiring: tools register on the MCPServer app with metadata, annotations and clean errors."""
import pytest

TOOLS = {"search_datasets", "describe_dataset", "get_access", "query_facets", "count_features", "download", "query", "sample_patches", "get_lod"}


def _tools():
    from mudm_mcp import server
    return {t.name: t for t in server.mcp._tool_manager.list_tools()}


def test_server_is_mcp2_with_version_and_instructions():
    from mcp.server.mcpserver import MCPServer

    import mudm_mcp
    from mudm_mcp import server
    assert isinstance(server.mcp, MCPServer)
    assert mudm_mcp.__version__ == "0.1.0"
    assert "search_datasets" in server.INSTRUCTIONS and len(server.INSTRUCTIONS) < 2000


def test_server_registers_all_tools():
    from mudm_mcp import server
    assert set(_tools()) == TOOLS
    assert callable(server.main)


def test_tool_input_schemas_generated():
    schema = _tools()["describe_dataset"].parameters
    assert "dataset_id" in schema.get("properties", {})


def test_tools_are_annotated_read_only_except_download():
    for name, t in _tools().items():
        assert t.annotations is not None, name
        if name in ("download", "count_features"):  # write to the local cache: never auto-approved
            assert t.annotations.read_only_hint is False
            assert t.annotations.destructive_hint is False and t.annotations.idempotent_hint is True
        else:
            assert t.annotations.read_only_hint is True, name


@pytest.fixture
def local_server(site, tmp_path, monkeypatch):
    from mudm_mcp import server
    monkeypatch.setenv("MUDM_BASE_URL", str(site))
    monkeypatch.setenv("MUDM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(server, "_client", None)
    return server


def test_unknown_dataset_raises_toolerror_with_suggestions(local_server):
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError, match=r"Unknown dataset 'demo2'.*Did you mean: demo2d"):
        local_server.describe_dataset("demo2")


def test_tool_value_errors_become_toolerrors(local_server):
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError, match="download"):  # nothing downloaded yet -> FileNotFoundError inside
        local_server.query("demo2d", "SELECT 1")


def test_search_and_describe_through_the_server(local_server):
    assert local_server.search_datasets(q="mouse brain")["datasets"][0]["id"] == "demo2d"
    assert local_server.describe_dataset("demo2d")["units"]["vertex_unit"] == "pixel"


def test_query_facets_through_the_server(local_server):
    from mcp.server.mcpserver.exceptions import ToolError
    r = local_server.query_facets("demo3d", "SELECT count(*) FROM facets WHERE cell_type LIKE 'KC%'")
    assert r["rows"] == [[3]]
    with pytest.raises(ToolError, match="Available columns"):
        local_server.query_facets("demo3d", "SELECT brain_regions FROM facets")
    with pytest.raises(ToolError, match="Permission|disabled"):
        local_server.query_facets("demo3d", "SELECT * FROM read_csv('/etc/hosts')")


def test_count_features_through_the_server(local_server):
    from mcp.server.mcpserver.exceptions import ToolError
    assert local_server.count_features("demo2d", bbox_um=[0, 0, 50, 50])["count"] == 2
    with pytest.raises(ToolError, match="bbox_um must be"):
        local_server.count_features("demo2d", bbox_um=[50, 0, 0, 50])
    with pytest.raises(ToolError, match="query_facets"):
        local_server.count_features("demo3d")


def test_wire_result_is_one_compact_json_text_block(local_server):
    """Hosts read the text block: it must be compact JSON (no pretty-printing, no duplicate structured
    copy), because the response-size budget is measured on that form."""
    import asyncio
    import json
    res = asyncio.run(local_server.mcp.call_tool("describe_dataset", {"dataset_id": "demo2d"}))
    blocks = res[0] if isinstance(res, tuple) else getattr(res, "content", res)
    structured = res[1] if isinstance(res, tuple) else getattr(res, "structured_content", None)
    assert len(blocks) == 1 and not structured
    text = blocks[0].text
    assert "\n" not in text and json.loads(text) == local_server.describe_dataset("demo2d")
    assert len(text) <= len(json.dumps(local_server.describe_dataset("demo2d")))
