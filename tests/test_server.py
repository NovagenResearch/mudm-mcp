"""Server wiring: all tools register on the FastMCP app + the console entry is callable."""


def test_server_registers_all_seven_tools():
    from mudm_mcp import server
    names = {t.name for t in server.mcp._tool_manager.list_tools()}
    assert names == {"list_datasets", "describe_dataset", "get_access", "download",
                     "query", "sample_patches", "get_lod"}
    assert callable(server.main)


def test_tool_input_schemas_generated():
    # FastMCP builds each tool's inputSchema from the type hints; describe_dataset must require dataset_id.
    from mudm_mcp import server
    by_name = {t.name: t for t in server.mcp._tool_manager.list_tools()}
    schema = by_name["describe_dataset"].parameters
    assert "dataset_id" in schema.get("properties", {})
