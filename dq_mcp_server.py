# dq_mcp_server.py
from mcp.server.fastmcp import FastMCP
import mcp_tools

mcp = FastMCP(name="DQ-MCP-Test")
mcp_tools.register_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")
