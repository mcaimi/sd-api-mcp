#!/usr/bin/env python
import argparse
import os
import sys

# import mcp machinery
from sd_api_mcp.server import mcp


# start the mcp server
def main() -> None:
    parser = argparse.ArgumentParser(description="Stable Diffusion MCP Server")

    # support all transports
    # - stdio for local interactions
    # - SSE for networked interactions (deprecated)
    # - streamable-http for network deployments (preferred if supported)
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
    )
    # host & port configuration for networked transports
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("MCP_PORT", "8080"))
    )
    args = parser.parse_args()

    try:
        # start MCP server
        if args.transport in ("sse", "streamable-http"):
            mcp.run(transport=args.transport, host=args.host, port=args.port)
        else:
            mcp.run()
    except KeyboardInterrupt:
        sys.exit(0)
