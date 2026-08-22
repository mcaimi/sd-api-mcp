# sd-api-mcp

An MCP (Model Context Protocol) server that exposes a Stable Diffusion REST API to AI agents. Supports SD1.5, SDXL, and Illustrious XL pipelines for text-to-image generation, inpainting, model management, and model merging.

It sits on top of the [SD API Backend](https://github.com/mcaimi/sd-api-backend.git) project.

Built with the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) and managed with [uv](https://docs.astral.sh/uv/).

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- A running [Stable Diffusion API](https://github.com/mcaimi/sd-api-backend.git) instance (default: `http://localhost:8000`)

## Installation

```bash
git clone <repo-url>
cd sd-api-mcp
uv sync
```

## Usage

### stdio (default)

For use with MCP clients that manage the server process (e.g., Claude Code, Claude Desktop):

```bash
uv run sd-api-mcp
```

### SSE

```bash
uv run sd-api-mcp --transport sse --host 0.0.0.0 --port 8080
```

### Streamable HTTP (recommended for networked deployments)

```bash
uv run sd-api-mcp --transport streamable-http --host 0.0.0.0 --port 8080
```

The MCP endpoint will be available at `http://<host>:<port>/mcp`.

### Claude Desktop configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "stable-diffusion": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/sd-api-mcp", "sd-api-mcp"],
      "env": {
        "SD_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

### Claude Code configuration

```bash
claude mcp add stable-diffusion -- uv run --project /path/to/sd-api-mcp sd-api-mcp
```

### OpenCode configuration

Add to your `opencode.json`:

```json
{
  "mcp": {
    "stable-diffusion": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--project", "/path/to/sd-api-mcp", "sd-api-mcp"],
      "env": {
        "SD_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

For SSE or Streamable HTTP transports, start the server separately and use a remote URL instead:

```json
{
  "mcp": {
    "stable-diffusion": {
      "type": "sse",
      "url": "http://localhost:8080/sse"
    }
  }
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SD_API_URL` | `http://localhost:8000` | Base URL of the Stable Diffusion API |
| `SD_POLL_INTERVAL` | `2.0` | Seconds between job status polls |
| `SD_POLL_TIMEOUT` | `600.0` | Maximum seconds to wait for job completion |
| `MCP_TRANSPORT` | `stdio` | Transport protocol (`stdio`, `sse`, `streamable-http`) |
| `MCP_HOST` | `127.0.0.1` | Bind address for SSE/HTTP transports |
| `MCP_PORT` | `8080` | Port for SSE/HTTP transports |

## Available Tools

### Image Generation

| Tool | Description |
|---|---|
| `generate_image` | Generate an image and wait for the result |
| `inpaint_image` | Inpaint a masked region and wait for the result |
| `submit_generate` | Submit a generation job, return the job ID immediately |
| `submit_inpaint` | Submit an inpainting job, return the job ID immediately |
| `batch_generate` | Submit up to 10 generation requests at once |
| `compare_models` | Generate with 2-6 models using the same prompt for comparison |

All generation tools accept a `pipeline` parameter: `"sd15"`, `"sdxl"`, or `"illustrious"`.

### Model Management

| Tool | Description |
|---|---|
| `list_models` | List available checkpoints, LoRAs, or VAEs |
| `get_model_metadata` | Read metadata from a model's safetensors header |

### Job Management

| Tool | Description |
|---|---|
| `list_jobs` | List all jobs with status and progress |
| `get_job_status` | Get status and result of a specific job |
| `cancel_job` | Cancel a pending or running job |

### System

| Tool | Description |
|---|---|
| `health_check` | Check if the SD API is reachable |
| `system_info` | Get GPU, cache, and queue statistics |
| `list_schedulers` | List available noise schedulers |
| `get_app_settings` | Get current configuration parameters |

### Model Merging

| Tool | Description |
|---|---|
| `merge_models` | Merge two checkpoints (linear, slerp, additive, subtract) |
| `batch_merge_models` | Merge a base model with multiple targets |
| `recipe_merge` | Execute a multi-step merge recipe |

## Examples

### Generate an image (agent perspective)

An AI agent would call the `generate_image` tool with:

```json
{
  "pipeline": "sdxl",
  "positive_prompt": "a cat sitting on a windowsill, golden hour lighting, photorealistic",
  "negative_prompt": "blurry, low quality",
  "model_checkpoint": "dreamshaperXL_v2.safetensors",
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "cfg_scale": 7.0,
  "seed": -1,
  "scheduler": "DPM++ 2M"
}
```

The tool submits the job to the SD API, polls until completion, and returns the full result including the generated image.

### List available models

```json
{
  "model_type": "sdxl",
  "resource_type": "checkpoints"
}
```

### Merge two models

```json
{
  "model_type": "sd15",
  "base_model": "v1-5-pruned.safetensors",
  "target_model": "dreamshaper_8.safetensors",
  "output_name": "merged_model.safetensors",
  "method": "slerp",
  "alpha": 0.5
}
```

## Project Structure

```
src/sd_api_mcp/
    __init__.py   # CLI entry point with transport selection
    server.py     # MCPServer instance and tool definitions
    client.py     # Async HTTP client for the SD API
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
