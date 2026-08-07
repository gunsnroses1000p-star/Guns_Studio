"""
utils/runpod.py — RunPod shared job helpers (submit, poll, decode).
"""

import base64
import time
from io import BytesIO

import gradio as gr
import requests
from PIL import Image

from config import RUNPOD_API_KEY, RUNPOD_BASE, RUNPOD_ENDPOINT_ID


# ---------------------------------------------------------------------------
# Headers / authentication
# ---------------------------------------------------------------------------

def get_runpod_headers() -> dict:
    if not RUNPOD_API_KEY:
        raise gr.Error("RUNPOD_API_KEY is missing.")
    auth = "Bearer " + RUNPOD_API_KEY
    return {
        "Authorization": auth,
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_runpod_job(input_payload: dict) -> str:
    """Submit a job to the configured RunPod endpoint and return its ID."""
    if not RUNPOD_ENDPOINT_ID:
        raise gr.Error("RUNPOD_ENDPOINT_ID is missing.")
    run_url = f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}/run"
    response = requests.post(
        run_url,
        headers=get_runpod_headers(),
        json={"input": input_payload},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    job_id = data.get("id")
    if not job_id:
        raise gr.Error(f"RunPod did not return a job ID: {data}")
    return job_id


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------

def wait_for_runpod_job(job_id: str, max_wait_seconds: int = 1200) -> dict:
    """Poll the RunPod status endpoint until the job completes."""
    status_url = f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    deadline = time.time() + max_wait_seconds
    last_result: dict = {}

    while time.time() < deadline:
        response = requests.get(
            status_url, headers=get_runpod_headers(), timeout=60
        )
        response.raise_for_status()
        last_result = response.json()
        status = str(last_result.get("status", "")).upper()

        if status == "COMPLETED":
            return last_result

        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            error = last_result.get("error")
            output = last_result.get("output")
            if not error and isinstance(output, dict):
                error = output.get("error")
            raise gr.Error(
                f"RunPod job {status}: {error or last_result}"
            )

        time.sleep(2)

    raise gr.Error(f"RunPod job timed out. Last result: {last_result}")


def runpod_job(input_payload: dict, max_wait_seconds: int = 1200) -> dict:
    """Submit a job and block until it completes; return the full result."""
    job_id = submit_runpod_job(input_payload)
    return wait_for_runpod_job(job_id, max_wait_seconds=max_wait_seconds)


# ---------------------------------------------------------------------------
# Output decoding
# ---------------------------------------------------------------------------

def decode_runpod_output(output) -> Image.Image:
    """
    Decode a RunPod job output (URL string, base64 string, or dict with
    url/base64 fields) into a PIL Image.
    """
    try:
        if isinstance(output, str):
            if output.startswith("http"):
                resp = requests.get(output, timeout=120)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            # treat as base64
            data = output
            if "," in data:
                data = data.split(",", 1)[1]
            return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")

        if isinstance(output, dict):
            if output.get("error"):
                raise gr.Error(str(output["error"]))

            url = (
                output.get("image_url")
                or output.get("url")
                or output.get("output_url")
            )
            if url:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")

            b64 = output.get("image_base64") or output.get("image")
            if b64:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                return Image.open(
                    BytesIO(base64.b64decode(b64))
                ).convert("RGB")

        raise ValueError(f"Unsupported RunPod output format: {output}")

    except gr.Error:
        raise
    except Exception as err:
        raise gr.Error(f"Could not decode RunPod output: {err}")
