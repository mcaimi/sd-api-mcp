import json

from mcp.server import MCPServer
from mcp.server.mcpserver import Image

from sd_api_mcp import client

mcp = MCPServer(
    name="sd-api",
    instructions=(
        "Stable Diffusion image generation API. "
        "Supports SD1.5, SDXL, Illustrious XL, Flux, and Z-Image pipelines for text-to-image generation, "
        "inpainting, model comparison, and model merging. "
        "Generation jobs are asynchronous — use generate_image/inpaint_image for automatic "
        "polling, or submit_generate/submit_inpaint + get_job_status for manual control. "
        "Leave optional sampling parameters unset to get each pipeline's backend defaults. "
        "Persisted results can be browsed with list_outputs and viewed with get_output_image."
    ),
)


def _build_generation_body(
    pipeline: str,
    positive_prompt: str,
    model_checkpoint: str,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int = -1,
    scheduler: str | None = None,
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
    true_cfg_scale: float | None = None,
    variant: str | None = None,
    strength: float | None = None,
) -> dict:
    """Build a request body, leaving out unset fields so the backend applies
    its pipeline-specific defaults. Pipeline-only options are sent only to the
    pipeline that understands them."""
    fields: dict = {
        "positive_prompt": positive_prompt,
        "model_checkpoint": model_checkpoint,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "scheduler": scheduler,
        "loras": loras or None,
        "custom_vae": custom_vae or None,
    }
    if pipeline == "flux":
        fields["true_cfg_scale"] = true_cfg_scale
    if pipeline == "zimage":
        fields["variant"] = variant
    if pipeline in ("flux", "zimage"):
        fields["strength"] = strength
    return {k: v for k, v in fields.items() if v is not None}


def _image(data: bytes, content_type: str) -> Image:
    # content_type is e.g. "image/png"; Image wants just the format
    fmt = content_type.split(";")[0].split("/")[-1] or "png"
    return Image(data=data, format=fmt)


# ---------------------------------------------------------------------------
# Image Generation (with automatic polling)
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_image(
    pipeline: str,
    positive_prompt: str,
    model_checkpoint: str,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int = -1,
    scheduler: str | None = None,
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
    true_cfg_scale: float | None = None,
    variant: str | None = None,
) -> str:
    """Generate an image using Stable Diffusion. Waits for completion and returns the result.

    Leave optional sampling parameters unset to use the pipeline's backend defaults.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", "illustrious", "flux", or "zimage"
        positive_prompt: Text prompt describing the desired image
        model_checkpoint: Model checkpoint filename to use
        negative_prompt: Negative prompt (things to avoid)
        width: Image width in pixels (256-2048). Default 512, or 1024 for "zimage"
        height: Image height in pixels (256-2048). Default 512, or 1024 for "zimage"
        steps: Number of inference steps (1-150). Default 30; "zimage" picks a
            per-variant default
        cfg_scale: Classifier-free guidance scale (0-30). Default 7.0; 3.5 for
            "flux" (distilled guidance); "zimage" picks a per-variant default
        seed: Random seed (-1 for random)
        scheduler: Noise scheduler name. Default "DPM++ 2M"; "flux" and "zimage"
            use flow-matching schedulers (default "FlowMatch Euler"). See
            list_schedulers(model_type=...)
        loras: Optional list of LoRA configs, each with "name" (str) and optional "strength" (float 0-1)
        custom_vae: Optional custom VAE checkpoint filename
        true_cfg_scale: Flux-only. True CFG scale (1-10, default 1.0); set >1.0 to
            enable negative_prompt. Ignored by other pipelines
        variant: Z-Image-only. "turbo" (default) or "base". Ignored by other pipelines
    """
    body = _build_generation_body(
        pipeline, positive_prompt, model_checkpoint, negative_prompt,
        width, height, steps, cfg_scale, seed, scheduler, loras, custom_vae,
        true_cfg_scale, variant,
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
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int = -1,
    scheduler: str | None = None,
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
    true_cfg_scale: float | None = None,
    variant: str | None = None,
    strength: float | None = None,
) -> str:
    """Inpaint an image using Stable Diffusion. Waits for completion and returns the result.

    Leave optional sampling parameters unset to use the pipeline's backend defaults.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", "illustrious", "flux", or "zimage"
        positive_prompt: Text prompt describing what to paint in the masked region
        model_checkpoint: Model checkpoint filename to use
        image_base64: Base64-encoded input image
        mask_base64: Base64-encoded mask image (white = inpaint region)
        negative_prompt: Negative prompt (things to avoid)
        steps: Number of inference steps (1-150). "zimage" picks a per-variant default
        cfg_scale: Classifier-free guidance scale (0-30). Default 7.0; 3.5 for
            "flux" (distilled guidance); "zimage" picks a per-variant default
        seed: Random seed (-1 for random)
        scheduler: Noise scheduler name. Default "DPM++ 2M"; "flux" and "zimage"
            use flow-matching schedulers (default "FlowMatch Euler"). See
            list_schedulers(model_type=...)
        loras: Optional list of LoRA configs, each with "name" (str) and optional "strength" (float 0-1)
        custom_vae: Optional custom VAE checkpoint filename
        true_cfg_scale: Flux-only. True CFG scale (1-10, default 1.0); set >1.0 to
            enable negative_prompt. Ignored by other pipelines
        variant: Z-Image-only. "turbo" (default) or "base". Ignored by other pipelines
        strength: Flux/Z-Image-only. Inpaint re-noising strength (0-1); the backend
            picks a default if omitted. Ignored by other pipelines
    """
    body = _build_generation_body(
        pipeline, positive_prompt, model_checkpoint, negative_prompt,
        None, None, steps, cfg_scale, seed, scheduler, loras, custom_vae,
        true_cfg_scale, variant, strength,
    )
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
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int = -1,
    scheduler: str | None = None,
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
    true_cfg_scale: float | None = None,
    variant: str | None = None,
) -> str:
    """Submit an image generation job without waiting. Returns the job ID for manual polling.

    Leave optional sampling parameters unset to use the pipeline's backend defaults.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", "illustrious", "flux", or "zimage"
        positive_prompt: Text prompt describing the desired image
        model_checkpoint: Model checkpoint filename to use
        negative_prompt: Negative prompt (things to avoid)
        width: Image width in pixels (256-2048). Default 512, or 1024 for "zimage"
        height: Image height in pixels (256-2048). Default 512, or 1024 for "zimage"
        steps: Number of inference steps (1-150). Default 30; "zimage" picks a
            per-variant default
        cfg_scale: Classifier-free guidance scale (0-30). Default 7.0; 3.5 for
            "flux" (distilled guidance); "zimage" picks a per-variant default
        seed: Random seed (-1 for random)
        scheduler: Noise scheduler name. Default "DPM++ 2M"; "flux" and "zimage"
            use flow-matching schedulers (default "FlowMatch Euler"). See
            list_schedulers(model_type=...)
        loras: Optional list of LoRA configs, each with "name" (str) and optional "strength" (float 0-1)
        custom_vae: Optional custom VAE checkpoint filename
        true_cfg_scale: Flux-only. True CFG scale (1-10, default 1.0); set >1.0 to
            enable negative_prompt. Ignored by other pipelines
        variant: Z-Image-only. "turbo" (default) or "base". Ignored by other pipelines
    """
    body = _build_generation_body(
        pipeline, positive_prompt, model_checkpoint, negative_prompt,
        width, height, steps, cfg_scale, seed, scheduler, loras, custom_vae,
        true_cfg_scale, variant,
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
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int = -1,
    scheduler: str | None = None,
    loras: list[dict] | None = None,
    custom_vae: str | None = None,
    true_cfg_scale: float | None = None,
    variant: str | None = None,
    strength: float | None = None,
) -> str:
    """Submit an inpainting job without waiting. Returns the job ID for manual polling.

    Leave optional sampling parameters unset to use the pipeline's backend defaults.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", "illustrious", "flux", or "zimage"
        positive_prompt: Text prompt describing what to paint in the masked region
        model_checkpoint: Model checkpoint filename to use
        image_base64: Base64-encoded input image
        mask_base64: Base64-encoded mask image (white = inpaint region)
        negative_prompt: Negative prompt (things to avoid)
        steps: Number of inference steps (1-150). "zimage" picks a per-variant default
        cfg_scale: Classifier-free guidance scale (0-30). Default 7.0; 3.5 for
            "flux" (distilled guidance); "zimage" picks a per-variant default
        seed: Random seed (-1 for random)
        scheduler: Noise scheduler name. Default "DPM++ 2M"; "flux" and "zimage"
            use flow-matching schedulers (default "FlowMatch Euler"). See
            list_schedulers(model_type=...)
        loras: Optional list of LoRA configs, each with "name" (str) and optional "strength" (float 0-1)
        custom_vae: Optional custom VAE checkpoint filename
        true_cfg_scale: Flux-only. True CFG scale (1-10, default 1.0); set >1.0 to
            enable negative_prompt. Ignored by other pipelines
        variant: Z-Image-only. "turbo" (default) or "base". Ignored by other pipelines
        strength: Flux/Z-Image-only. Inpaint re-noising strength (0-1); the backend
            picks a default if omitted. Ignored by other pipelines
    """
    body = _build_generation_body(
        pipeline, positive_prompt, model_checkpoint, negative_prompt,
        None, None, steps, cfg_scale, seed, scheduler, loras, custom_vae,
        true_cfg_scale, variant, strength,
    )
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
        pipeline: Pipeline to use — "sd15", "sdxl", "illustrious", "flux", or "zimage"
        requests: List of generation request objects, each with at least "positive_prompt" and "model_checkpoint".
            Omitted fields take the pipeline's backend defaults. For "flux", requests may also
            include "true_cfg_scale" (float, 1-10, default 1.0); for "zimage", "variant"
            ("turbo" or "base")
    """
    result = await client.batch_generate(pipeline, {"requests": requests})
    return json.dumps(result, indent=2)


@mcp.tool()
async def compare_models(
    pipeline: str,
    positive_prompt: str,
    models: list[str],
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int = 42,
    scheduler: str | None = None,
    variant: str | None = None,
) -> str:
    """Generate images from multiple models using the same prompt for comparison.

    Leave optional sampling parameters unset to use the pipeline's backend defaults.

    Args:
        pipeline: Pipeline to use — "sd15", "sdxl", "illustrious", "flux", or "zimage"
        positive_prompt: Text prompt for all models
        models: List of model checkpoint filenames to compare (2-6)
        negative_prompt: Negative prompt (things to avoid)
        width: Image width in pixels (256-2048). Default 512, or 1024 for "zimage"
        height: Image height in pixels (256-2048). Default 512, or 1024 for "zimage"
        steps: Number of inference steps (1-150). Default 30; "zimage" picks a
            per-variant default
        cfg_scale: Classifier-free guidance scale (0-30). Default 7.0; 3.5 for
            "flux" (distilled guidance); "zimage" picks a per-variant default
        seed: Fixed seed for fair comparison (default 42)
        scheduler: Noise scheduler name. Default "DPM++ 2M"; "flux" and "zimage"
            use flow-matching schedulers (default "FlowMatch Euler")
        variant: Z-Image-only. "turbo" (default) or "base". Ignored by other pipelines
    """
    fields = {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "models": models,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "scheduler": scheduler,
        "variant": variant if pipeline == "zimage" else None,
    }
    body = {k: v for k, v in fields.items() if v is not None}
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

    Each entry includes its name and path and, when known, Civitai details
    (model/version IDs, base model, trained words), whether a preview image is
    available (see get_model_preview), and for Flux checkpoints which components
    (transformer, CLIP-L, T5-XXL, VAE) the file bundles.

    Args:
        model_type: Model architecture — "sd15", "sdxl", "illustrious", "flux", or "zimage"
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
        model_type: Model architecture — "sd15", "sdxl", "illustrious", "flux", or "zimage"
        resource_type: Resource category — "checkpoints", "loras", or "vaes"
        model_name: Model filename
    """
    result = await client.get_model_metadata(model_type, resource_type, model_name)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_model_preview(
    model_type: str,
    resource_type: str,
    model_name: str,
) -> Image:
    """Get the preview image of a model downloaded from (or identified on) Civitai.

    Only available when list_models reports has_preview=true for the model.

    Args:
        model_type: Model architecture — "sd15", "sdxl", "illustrious", "flux", or "zimage"
        resource_type: Resource category — "checkpoints", "loras", or "vaes"
        model_name: Model filename
    """
    data, content_type = await client.get_model_preview(model_type, resource_type, model_name)
    return _image(data, content_type)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_jobs() -> str:
    """List all jobs (generations, merges, Civitai downloads) with their status, progress, and timing information."""
    result = await client.list_jobs()
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_job_status(job_id: str) -> str:
    """Get the current status and result of a specific job.

    The response includes the job type, the queue lane it runs in, whether it is
    cancellable, progress (0-1), a job-specific detail object (e.g. download
    progress), and the result or error once finished.

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
async def list_schedulers(model_type: str | None = None) -> str:
    """List the noise schedulers available for image generation.

    Args:
        model_type: Optional pipeline. Without it, lists the SD1.5/SDXL/Illustrious
            schedulers; pass "flux" or "zimage" for their flow-matching schedulers
    """
    result = await client.list_schedulers(model_type)
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
    output_dtype: str = "auto",
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
        output_dtype: Precision of the saved checkpoint (e.g. "fp16", "bf16", "fp32");
            "auto" keeps the source precision
    """
    body = {
        "model_type": model_type,
        "base_model": base_model,
        "target_model": target_model,
        "output_name": output_name,
        "method": method,
        "alpha": alpha,
        "preserve_metadata": preserve_metadata,
        "output_dtype": output_dtype,
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
    output_dtype: str = "auto",
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
        output_dtype: Precision of the saved checkpoint (e.g. "fp16", "bf16", "fp32");
            "auto" keeps the source precision
    """
    body = {
        "model_type": model_type,
        "base_model": base_model,
        "target_models": target_models,
        "method": method,
        "alpha": alpha,
        "output_subdir": output_subdir,
        "preserve_metadata": preserve_metadata,
        "output_dtype": output_dtype,
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
    output_dtype: str = "auto",
) -> str:
    """Execute a multi-step merge recipe. Each step merges the current result with a new target. Waits for completion.

    Args:
        model_type: Model architecture — "sd15" or "sdxl"
        base_model: Base model checkpoint filename
        steps: List of merge steps, each with "target_model" (str), "method" (str), and "alpha" (float 0-1)
        output_name: Output checkpoint filename for the final merged model
        output_dtype: Precision of the saved checkpoint (e.g. "fp16", "bf16", "fp32");
            "auto" keeps the source precision
    """
    body = {
        "model_type": model_type,
        "base_model": base_model,
        "steps": steps,
        "output_name": output_name,
        "output_dtype": output_dtype,
    }
    job = await client.recipe_merge(body)
    result = await client.poll_job(job["job_id"])
    return json.dumps(result, indent=2)



# ---------------------------------------------------------------------------
# Outputs (persisted generation results)
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_outputs(limit: int = 24, offset: int = 0) -> str:
    """List persisted generation outputs, newest first.

    Args:
        limit: Maximum number of outputs to return
        offset: Number of outputs to skip, for pagination
    """
    result = await client.list_outputs(limit, offset)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_output_image(output_id: str) -> Image:
    """Get the image of a persisted generation output.

    Args:
        output_id: Output identifier from list_outputs
    """
    data, content_type = await client.get_output_image(output_id)
    return _image(data, content_type)


@mcp.tool()
async def get_output_metadata(output_id: str) -> str:
    """Get the generation parameters (prompt, steps, seed, LoRAs, ...) of a persisted output.

    Args:
        output_id: Output identifier from list_outputs
    """
    result = await client.get_output_metadata(output_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_output(output_id: str) -> str:
    """Delete the persisted image and metadata of a generation output.

    Args:
        output_id: Output identifier from list_outputs
    """
    await client.delete_output(output_id)
    return json.dumps({"deleted": output_id}, indent=2)