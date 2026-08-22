import json

from mcp.server import MCPServer

from sd_api_mcp import client

mcp = MCPServer(
    name="sd-api",
    instructions=(
        "Stable Diffusion image generation API. "
        "Supports SD1.5, SDXL, and Illustrious XL pipelines for text-to-image generation, "
        "inpainting, model comparison, and model merging. "
        "Generation jobs are asynchronous — use generate_image/inpaint_image for automatic "
        "polling, or submit_generate/submit_inpaint + get_job_status for manual control."
    ),
)


def _build_generation_body(
    positive_prompt: str,
    model_checkpoint: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    cfg_scale: float = 7.0,
    seed: int = -1,
    scheduler: str = "DPM++ 2M",
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
) -> dict:
    body: dict = {
        "positive_prompt": positive_prompt,
        "model_checkpoint": model_checkpoint,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "scheduler": scheduler,
    }
    if loras:
        body["loras"] = loras
    if custom_vae:
        body["custom_vae"] = custom_vae
    return body


# ---------------------------------------------------------------------------
# Image Generation (with automatic polling)
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_image(
    pipeline: str,
    positive_prompt: str,
    model_checkpoint: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    cfg_scale: float = 7.0,
    seed: int = -1,
    scheduler: str = "DPM++ 2M",
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
) -> str:
    """Generate an image using Stable Diffusion. Waits for completion and returns the result.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", or "illustrious"
        positive_prompt: Text prompt describing the desired image
        model_checkpoint: Model checkpoint filename to use
        negative_prompt: Negative prompt (things to avoid)
        width: Image width in pixels (256-2048)
        height: Image height in pixels (256-2048)
        steps: Number of inference steps (1-150)
        cfg_scale: Classifier-free guidance scale (1-30)
        seed: Random seed (-1 for random)
        scheduler: Noise scheduler name
        loras: Optional list of LoRA configs, each with "name" (str) and optional "strength" (float 0-1)
        custom_vae: Optional custom VAE checkpoint filename
    """
    body = _build_generation_body(
        positive_prompt, model_checkpoint, negative_prompt,
        width, height, steps, cfg_scale, seed, scheduler, loras, custom_vae,
    )
    job = await client.generate(pipeline, body)
    result = await client.poll_job(job["job_id"])
    return json.dumps(result, indent=2)


@mcp.tool()
async def inpaint_image(
    pipeline: str,
    positive_prompt: str,
    model_checkpoint: str,
    image_base64: str,
    mask_base64: str,
    negative_prompt: str = "",
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = -1,
    scheduler: str = "DPM++ 2M",
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
) -> str:
    """Inpaint an image using Stable Diffusion. Waits for completion and returns the result.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", or "illustrious"
        positive_prompt: Text prompt describing what to paint in the masked region
        model_checkpoint: Model checkpoint filename to use
        image_base64: Base64-encoded input image
        mask_base64: Base64-encoded mask image (white = inpaint region)
        negative_prompt: Negative prompt (things to avoid)
        steps: Number of inference steps (1-150)
        cfg_scale: Classifier-free guidance scale (1-30)
        seed: Random seed (-1 for random)
        scheduler: Noise scheduler name
        loras: Optional list of LoRA configs, each with "name" (str) and optional "strength" (float 0-1)
        custom_vae: Optional custom VAE checkpoint filename
    """
    body = _build_generation_body(
        positive_prompt, model_checkpoint, negative_prompt,
        512, 512, steps, cfg_scale, seed, scheduler, loras, custom_vae,
    )
    body.pop("width")
    body.pop("height")
    body["image_base64"] = image_base64
    body["mask_base64"] = mask_base64
    job = await client.inpaint(pipeline, body)
    result = await client.poll_job(job["job_id"])
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Image Generation (fire-and-forget — returns job ID immediately)
# ---------------------------------------------------------------------------

@mcp.tool()
async def submit_generate(
    pipeline: str,
    positive_prompt: str,
    model_checkpoint: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    cfg_scale: float = 7.0,
    seed: int = -1,
    scheduler: str = "DPM++ 2M",
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
) -> str:
    """Submit an image generation job without waiting. Returns the job ID for manual polling.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", or "illustrious"
        positive_prompt: Text prompt describing the desired image
        model_checkpoint: Model checkpoint filename to use
        negative_prompt: Negative prompt (things to avoid)
        width: Image width in pixels (256-2048)
        height: Image height in pixels (256-2048)
        steps: Number of inference steps (1-150)
        cfg_scale: Classifier-free guidance scale (1-30)
        seed: Random seed (-1 for random)
        scheduler: Noise scheduler name
        loras: Optional list of LoRA configs, each with "name" (str) and optional "strength" (float 0-1)
        custom_vae: Optional custom VAE checkpoint filename
    """
    body = _build_generation_body(
        positive_prompt, model_checkpoint, negative_prompt,
        width, height, steps, cfg_scale, seed, scheduler, loras, custom_vae,
    )
    job = await client.generate(pipeline, body)
    return json.dumps(job, indent=2)


@mcp.tool()
async def submit_inpaint(
    pipeline: str,
    positive_prompt: str,
    model_checkpoint: str,
    image_base64: str,
    mask_base64: str,
    negative_prompt: str = "",
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = -1,
    scheduler: str = "DPM++ 2M",
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
) -> str:
    """Submit an inpainting job without waiting. Returns the job ID for manual polling.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", or "illustrious"
        positive_prompt: Text prompt describing what to paint in the masked region
        model_checkpoint: Model checkpoint filename to use
        image_base64: Base64-encoded input image
        mask_base64: Base64-encoded mask image (white = inpaint region)
        negative_prompt: Negative prompt (things to avoid)
        steps: Number of inference steps (1-150)
        cfg_scale: Classifier-free guidance scale (1-30)
        seed: Random seed (-1 for random)
        scheduler: Noise scheduler name
        loras: Optional list of LoRA configs, each with "name" (str) and optional "strength" (float 0-1)
        custom_vae: Optional custom VAE checkpoint filename
    """
    body = _build_generation_body(
        positive_prompt, model_checkpoint, negative_prompt,
        512, 512, steps, cfg_scale, seed, scheduler, loras, custom_vae,
    )
    body.pop("width")
    body.pop("height")
    body["image_base64"] = image_base64
    body["mask_base64"] = mask_base64
    job = await client.inpaint(pipeline, body)
    return json.dumps(job, indent=2)


# ---------------------------------------------------------------------------
# Batch & Compare
# ---------------------------------------------------------------------------

@mcp.tool()
async def batch_generate(
    pipeline: str,
    requests: list[dict],
) -> str:
    """Submit a batch of image generation requests (up to 10).

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", or "illustrious"
        requests: List of generation request objects, each with at least "positive_prompt" and "model_checkpoint"
    """
    result = await client.batch_generate(pipeline, {"requests": requests})
    return json.dumps(result, indent=2)


@mcp.tool()
async def compare_models(
    pipeline: str,
    positive_prompt: str,
    models: list[str],
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    cfg_scale: float = 7.0,
    seed: int = 42,
    scheduler: str = "DPM++ 2M",
) -> str:
    """Generate images from multiple models using the same prompt for comparison.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", or "illustrious"
        positive_prompt: Text prompt for all models
        models: List of model checkpoint filenames to compare (2-6)
        negative_prompt: Negative prompt (things to avoid)
        width: Image width in pixels (256-2048)
        height: Image height in pixels (256-2048)
        steps: Number of inference steps (1-150)
        cfg_scale: Classifier-free guidance scale (1-30)
        seed: Fixed seed for fair comparison (default 42)
        scheduler: Noise scheduler name
    """
    body = {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "models": models,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "scheduler": scheduler,
    }
    result = await client.compare_models(pipeline, body)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_models(
    model_type: str,
    resource_type: str,
) -> str:
    """List available models.

    Args:
        model_type: Model architecture — "sd15" or "sdxl"
        resource_type: Resource category — "checkpoints", "loras", or "vaes"
    """
    result = await client.list_models(model_type, resource_type)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_model_metadata(
    model_type: str,
    resource_type: str,
    model_name: str,
) -> str:
    """Get metadata for a specific model from its safetensors header.

    Args:
        model_type: Model architecture — "sd15" or "sdxl"
        resource_type: Resource category — "checkpoints", "loras", or "vaes"
        model_name: Model filename
    """
    result = await client.get_model_metadata(model_type, resource_type, model_name)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_jobs() -> str:
    """List all jobs with their current status, progress, and timing information."""
    result = await client.list_jobs()
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_job_status(job_id: str) -> str:
    """Get the current status and result of a specific job.

    Args:
        job_id: The job identifier returned when the job was submitted
    """
    result = await client.get_job_status(job_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def cancel_job(job_id: str) -> str:
    """Cancel a pending or running job.

    Args:
        job_id: The job identifier to cancel
    """
    result = await client.cancel_job(job_id)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# System & Info
# ---------------------------------------------------------------------------

@mcp.tool()
async def health_check() -> str:
    """Check if the Stable Diffusion API is healthy and reachable."""
    result = await client.health_check()
    return json.dumps(result, indent=2)


@mcp.tool()
async def system_info() -> str:
    """Get system information including GPU device, dtype, model cache stats, and job queue stats."""
    result = await client.system_info()
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_schedulers() -> str:
    """List all available noise schedulers that can be used for image generation."""
    result = await client.list_schedulers()
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_app_settings() -> str:
    """Get the current application configuration parameters."""
    result = await client.get_app_settings()
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Model Merging
# ---------------------------------------------------------------------------

@mcp.tool()
async def merge_models(
    model_type: str,
    base_model: str,
    target_model: str,
    output_name: str,
    method: str = "linear",
    alpha: float = 0.5,
    preserve_metadata: bool = True,
) -> str:
    """Merge two model checkpoints. Submits the job and waits for completion.

    Args:
        model_type: Model architecture — "sd15" or "sdxl"
        base_model: Base model checkpoint filename
        target_model: Target model checkpoint filename
        output_name: Output checkpoint filename for the merged model
        method: Merge method — "linear", "slerp", "additive", or "subtract"
        alpha: Merge strength (0.0-1.0)
        preserve_metadata: Whether to preserve model metadata
    """
    body = {
        "model_type": model_type,
        "base_model": base_model,
        "target_model": target_model,
        "output_name": output_name,
        "method": method,
        "alpha": alpha,
        "preserve_metadata": preserve_metadata,
    }
    job = await client.merge_models(body)
    result = await client.poll_job(job["job_id"])
    return json.dumps(result, indent=2)


@mcp.tool()
async def batch_merge_models(
    model_type: str,
    base_model: str,
    target_models: list[str],
    method: str = "linear",
    alpha: float = 0.5,
    output_subdir: str = "batch_merged",
    preserve_metadata: bool = True,
) -> str:
    """Merge a base model with multiple target models in batch. Waits for completion.

    Args:
        model_type: Model architecture — "sd15" or "sdxl"
        base_model: Base model checkpoint filename
        target_models: List of target model checkpoint filenames
        method: Merge method — "linear", "slerp", "additive", or "subtract"
        alpha: Merge strength (0.0-1.0)
        output_subdir: Output subdirectory name for merged models
        preserve_metadata: Whether to preserve model metadata
    """
    body = {
        "model_type": model_type,
        "base_model": base_model,
        "target_models": target_models,
        "method": method,
        "alpha": alpha,
        "output_subdir": output_subdir,
        "preserve_metadata": preserve_metadata,
    }
    job = await client.batch_merge(body)
    result = await client.poll_job(job["job_id"])
    return json.dumps(result, indent=2)


@mcp.tool()
async def recipe_merge(
    model_type: str,
    base_model: str,
    steps: list[dict],
    output_name: str,
) -> str:
    """Execute a multi-step merge recipe. Each step merges the current result with a new target. Waits for completion.

    Args:
        model_type: Model architecture — "sd15" or "sdxl"
        base_model: Base model checkpoint filename
        steps: List of merge steps, each with "target_model" (str), "method" (str), and "alpha" (float 0-1)
        output_name: Output checkpoint filename for the final merged model
    """
    body = {
        "model_type": model_type,
        "base_model": base_model,
        "steps": steps,
        "output_name": output_name,
    }
    job = await client.recipe_merge(body)
    result = await client.poll_job(job["job_id"])
    return json.dumps(result, indent=2)
