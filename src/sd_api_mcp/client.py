import asyncio
import json
import os
from urllib.parse import quote

import httpx


SD_API_URL = os.environ.get("SD_API_URL", "http://localhost:8000")
POLL_INTERVAL = float(os.environ.get("SD_POLL_INTERVAL", "2.0"))
POLL_TIMEOUT = float(os.environ.get("SD_POLL_TIMEOUT", "600.0"))
HTTP_TIMEOUT = float(os.environ.get("SD_HTTP_TIMEOUT", "30.0"))


def _base_url() -> str:
    return SD_API_URL.rstrip("/")


def _check(resp: httpx.Response) -> None:
    """Raise with the backend's error detail instead of a bare HTTP status."""
    if resp.is_success:
        return
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    if not isinstance(detail, str):
        detail = json.dumps(detail)
    raise RuntimeError(f"{resp.request.method} {resp.request.url.path} -> {resp.status_code}: {detail}")


async def _send(method: str, path: str, timeout: float | None = None, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=timeout or HTTP_TIMEOUT) as client:
        resp = await client.request(method, path, **kwargs)
        _check(resp)
        return resp


async def _request(method: str, path: str, timeout: float | None = None, **kwargs) -> dict:
    resp = await _send(method, path, timeout, **kwargs)
    if resp.status_code == 204 or not resp.content:
        return {}
    return resp.json()


async def _request_raw(method: str, path: str, **kwargs) -> tuple[bytes, str]:
    resp = await _send(method, path, **kwargs)
    return resp.content, resp.headers.get("content-type", "application/octet-stream")


async def poll_job(job_id: str, timeout: float | None = None) -> dict:
    timeout = timeout or POLL_TIMEOUT
    elapsed = 0.0
    result: dict = {}
    while elapsed < timeout:
        result = await _request("GET", f"/api/v1/jobs/{job_id}")
        status = result.get("status", "")
        if status == "completed":
            return result
        if status in ("failed", "cancelled"):
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Job {job_id} {status}: {error}")
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    raise TimeoutError(
        f"Job {job_id} did not complete within {timeout}s "
        f"(status={result.get('status')}, progress={result.get('progress')}, detail={result.get('detail')})"
    )


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
    return await _request("GET", f"/api/v1/models/{model_type}/{resource_type}/{quote(model_name)}/metadata")


async def get_model_preview(model_type: str, resource_type: str, model_name: str) -> tuple[bytes, str]:
    return await _request_raw("GET", f"/api/v1/models/{model_type}/{resource_type}/{quote(model_name)}/preview")


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

async def list_schedulers(model_type: str | None = None) -> list:
    params = {"model_type": model_type} if model_type else None
    return await _request("GET", "/api/v1/info/schedulers", params=params)


async def get_app_settings() -> dict:
    return await _request("GET", "/api/v1/info/app_settings")


# --- Merge ---

async def merge_models(body: dict) -> dict:
    return await _request("POST", "/api/v1/merge/merge", json=body)


async def batch_merge(body: dict) -> dict:
    return await _request("POST", "/api/v1/merge/batch", json=body)


async def recipe_merge(body: dict) -> dict:
    return await _request("POST", "/api/v1/merge/recipe", json=body)
