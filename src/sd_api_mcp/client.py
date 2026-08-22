import asyncio
import os

import httpx


SD_API_URL = os.environ.get("SD_API_URL", "http://localhost:8000")
POLL_INTERVAL = float(os.environ.get("SD_POLL_INTERVAL", "2.0"))
POLL_TIMEOUT = float(os.environ.get("SD_POLL_TIMEOUT", "600.0"))


def _base_url() -> str:
    return SD_API_URL.rstrip("/")


async def _request(method: str, path: str, **kwargs) -> dict:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=30.0) as client:
        resp = await client.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json()


async def poll_job(job_id: str) -> dict:
    elapsed = 0.0
    while elapsed < POLL_TIMEOUT:
        result = await _request("GET", f"/api/v1/jobs/{job_id}")
        status = result.get("status", "")
        if status == "completed":
            return result
        if status in ("failed", "cancelled"):
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Job {job_id} {status}: {error}")
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    raise TimeoutError(f"Job {job_id} did not complete within {POLL_TIMEOUT}s")


# --- Generation ---

async def generate(pipeline: str, body: dict) -> dict:
    return await _request("POST", f"/api/v1/{pipeline}/generate", json=body)


async def inpaint(pipeline: str, body: dict) -> dict:
    return await _request("POST", f"/api/v1/{pipeline}/inpaint", json=body)


async def batch_generate(pipeline: str, body: dict) -> dict:
    return await _request("POST", f"/api/v1/{pipeline}/generate/batch", json=body)


async def compare_models(pipeline: str, body: dict) -> dict:
    return await _request("POST", f"/api/v1/{pipeline}/generate/compare", json=body)


# --- Models ---

async def list_models(model_type: str, resource_type: str) -> dict:
    return await _request("GET", f"/api/v1/models/{model_type}/{resource_type}")


async def get_model_metadata(model_type: str, resource_type: str, model_name: str) -> dict:
    return await _request("GET", f"/api/v1/models/{model_type}/{resource_type}/{model_name}/metadata")


# --- Jobs ---

async def list_jobs() -> list:
    return await _request("GET", "/api/v1/jobs/")


async def get_job_status(job_id: str) -> dict:
    return await _request("GET", f"/api/v1/jobs/{job_id}")


async def cancel_job(job_id: str) -> dict:
    return await _request("DELETE", f"/api/v1/jobs/{job_id}")


# --- System ---

async def health_check() -> dict:
    return await _request("GET", "/api/v1/system/health")


async def system_info() -> dict:
    return await _request("GET", "/api/v1/system/info")


# --- Informational ---

async def list_schedulers() -> list:
    return await _request("GET", "/api/v1/info/schedulers")


async def get_app_settings() -> dict:
    return await _request("GET", "/api/v1/info/app_settings")


# --- Merge ---

async def merge_models(body: dict) -> dict:
    return await _request("POST", "/api/v1/merge/merge", json=body)


async def batch_merge(body: dict) -> dict:
    return await _request("POST", "/api/v1/merge/batch", json=body)


async def recipe_merge(body: dict) -> dict:
    return await _request("POST", "/api/v1/merge/recipe", json=body)
