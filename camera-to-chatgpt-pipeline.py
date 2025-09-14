"""Minimal camera-to-LLM pipeline with local model support."""

import base64
import json
import logging
import os
import threading
import time
import sys
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import pytesseract
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

CAPTURE_INTERVAL_SEC = int(os.getenv("CAPTURE_INTERVAL_SEC", "10"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_RTSP_URL = os.getenv("CAMERA_RTSP_URL", "")

# Image transform controls
MIRROR_HORIZONTAL = os.getenv("MIRROR_HORIZONTAL", "true").lower() in ("1", "true", "yes", "on")
ROTATE_DEG = float(os.getenv("ROTATE_DEG", "0"))
# For perspective keystone correction (camera low, tilted up => top appears narrower)
# Value in [0, 0.4]; 0 disables.
KEYSTONE_TOP_INSET_PCT = float(os.getenv("KEYSTONE_TOP_INSET_PCT", "0.10"))
KEYSTONE_BOTTOM_INSET_PCT = float(os.getenv("KEYSTONE_BOTTOM_INSET_PCT", "0.0"))

ANSWER_MODE = os.getenv("ANSWER_MODE", "api")  # api | local | browser
LOCAL_LLM_KIND = os.getenv("LOCAL_LLM_KIND", "ollama")  # ollama | openai_compat
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")

VISION_MODE = os.getenv("VISION_MODE", "auto")  # auto | ocr | ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:7b")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_QUEUE_FILE = os.getenv("WEBHOOK_QUEUE_FILE", "/data/webhook_queue.jsonl")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

AUTO_CAPTURE = False
last_problem: str = ""
last_answer: str = ""
webhook_url = WEBHOOK_URL
last_image_original_b64: str = ""
last_image_transformed_b64: str = ""

# Prometheus metrics
ANALYZER_LATENCY = Histogram("analyzer_latency_seconds", "Time spent analyzing frames")
SOLVER_LATENCY = Histogram("solver_latency_seconds", "Time spent solving problems")
WEBHOOK_LATENCY = Histogram("webhook_latency_seconds", "Time to post webhook payloads")
SOLVER_ERRORS = Counter("solver_errors_total", "Solver errors")
WEBHOOK_ERRORS = Counter("webhook_errors_total", "Webhook post errors")
ANALYZER_ERRORS = Counter("analyzer_errors_total", "Analyzer errors")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()


class Status(BaseModel):
    interval_sec: int
    auto_capture: bool
    last_problem: str
    last_answer: str
    webhook_url: str


@app.get("/status", response_model=Status)
async def get_status() -> Status:
    """Return basic status including capture interval."""
    return Status(
        interval_sec=CAPTURE_INTERVAL_SEC,
        auto_capture=AUTO_CAPTURE,
        last_problem=last_problem,
        last_answer=last_answer,
        webhook_url=webhook_url,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Expose Prometheus metrics."""
    return generate_latest().decode("utf-8")


# ---------------------------------------------------------------------------
# Vision extraction helpers
# ---------------------------------------------------------------------------


def ocr_extract_text(img_bgr: np.ndarray) -> str:
    """Extract text from an image using Tesseract OCR."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()


def transform_frame(img_bgr: np.ndarray) -> np.ndarray:
    """Apply mirror, rotation, and keystone correction to the frame.

    - Mirror horizontally by default to undo typical webcam mirroring.
    - Optional small rotation to correct tilt.
    - Keystone correction to compensate for camera placed low and tilted up.
    """
    out = img_bgr

    # Mirror horizontally
    if MIRROR_HORIZONTAL:
        out = cv2.flip(out, 1)

    # Rotate if configured
    if abs(ROTATE_DEG) > 0.01:
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ROTATE_DEG, 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Keystone correction (top narrower than bottom)
    h, w = out.shape[:2]
    top_inset_px = int(max(0.0, min(KEYSTONE_TOP_INSET_PCT, 0.4)) * w)
    bottom_inset_px = int(max(0.0, min(KEYSTONE_BOTTOM_INSET_PCT, 0.4)) * w)
    if top_inset_px > 0 and bottom_inset_px == 0:
        src = np.float32([
            [top_inset_px, 0],
            [w - 1 - top_inset_px, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])
        dst = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])
        P = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    elif bottom_inset_px > 0 and top_inset_px == 0:
        src = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1 - bottom_inset_px, h - 1],
            [bottom_inset_px, h - 1],
        ])
        dst = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])
        P = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return out


def ollama_vision_summarize(jpeg_bytes: bytes) -> str:
    """Summarize an image using an Ollama vision model."""
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Describe the problem in the image succinctly.",
        "images": [b64],
        "stream": False,
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except Exception:
        return ""


def extract_problem_statement(jpeg: bytes, img_bgr: np.ndarray) -> str:
    """Derive a problem statement from an image using vision/OCR."""
    summary: Optional[str] = None
    if VISION_MODE in ("auto", "ollama"):
        summary = ollama_vision_summarize(jpeg)
    if VISION_MODE in ("auto", "ocr") and (
        not summary or summary.upper() == "NO PROBLEM FOUND"
    ):
        txt = ocr_extract_text(img_bgr)
        summary = txt if txt else "NO PROBLEM FOUND"
    return summary or "NO PROBLEM FOUND"


# ---------------------------------------------------------------------------
# Solver backends
# ---------------------------------------------------------------------------


def send_to_openai_api(problem_statement: str) -> str:
    """Send the problem to OpenAI's API and return the answer with retries."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Answer concisely; show steps for math."},
                    {"role": "user", "content": problem_statement},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def send_to_local_llm(problem_statement: str) -> str:
    """Send the problem to a local LLM via Ollama or OpenAI-compatible API with retries."""
    if LOCAL_LLM_KIND == "ollama":
        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer concisely; show steps for math.",
                },
                {"role": "user", "content": f"Solve this problem:\n\n{problem_statement}"},
            ],
            "stream": False,
        }
        url = f"{LOCAL_LLM_BASE_URL}/api/chat"
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "").strip()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    else:  # openai_compat
        url = f"{LOCAL_LLM_BASE_URL}/v1/chat/completions"
        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer concisely; show steps for math.",
                },
                {"role": "user", "content": f"Solve this problem:\n\n{problem_statement}"},
            ],
            "temperature": 0.2,
        }
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


def send_to_chatgpt_browser(problem_statement: str) -> str:
    """Placeholder for browser-based ChatGPT interaction."""
    return "Browser mode not implemented"


def send_problem_to_chat(problem_statement: str) -> str:
    """Route problem statement to the selected backend, tracking latency/errors."""
    start = time.time()
    try:
        if ANSWER_MODE == "browser":
            backend = "browser"
            answer = send_to_chatgpt_browser(problem_statement)
        elif ANSWER_MODE == "local":
            backend = "local"
            answer = send_to_local_llm(problem_statement)
        else:
            backend = "api"
            answer = send_to_openai_api(problem_statement)
        latency = time.time() - start
        SOLVER_LATENCY.observe(latency)
        logging.info("Solver (%s) took %.2fs", backend, latency)
        return answer
    except Exception as e:
        SOLVER_ERRORS.inc()
        logging.exception("Solver failed")
        return f"Solver error: {e}".strip()


# ---------------------------------------------------------------------------
# Capture and automation helpers
# ---------------------------------------------------------------------------


def capture_images() -> Optional[tuple[bytes, bytes, np.ndarray]]:
    """Capture one frame and return (orig_jpeg, transformed_jpeg, transformed_frame)."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened() and CAMERA_RTSP_URL:
        cap.release()
        cap = cv2.VideoCapture(CAMERA_RTSP_URL)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    orig = frame
    transformed = transform_frame(frame)
    ok1, buf1 = cv2.imencode(".jpg", orig)
    ok2, buf2 = cv2.imencode(".jpg", transformed)
    if not (ok1 and ok2):
        return None
    return buf1.tobytes(), buf2.tobytes(), transformed


def analyze_current_frame() -> str:
    """Capture a frame, extract a problem statement, and log latency."""
    start = time.time()
    try:
        data = capture_images()
        if not data:
            ANALYZER_ERRORS.inc()
            logging.info("Analyzer found no frame in %.2fs", time.time() - start)
            return "NO PROBLEM FOUND"
        orig_jpeg, transformed_jpeg, frame = data
        # Persist images for display on index page
        global last_image_original_b64, last_image_transformed_b64
        last_image_original_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_jpeg).decode("utf-8")
        last_image_transformed_b64 = "data:image/jpeg;base64," + base64.b64encode(transformed_jpeg).decode("utf-8")
        summary = extract_problem_statement(transformed_jpeg, frame)
        latency = time.time() - start
        ANALYZER_LATENCY.observe(latency)
        logging.info("Analyzer took %.2fs", latency)
        return summary
    except Exception:
        ANALYZER_ERRORS.inc()
        logging.exception("Analyzer failed")
        return "NO PROBLEM FOUND"


def enqueue_webhook_payload(payload: dict) -> None:
    """Persist a failed webhook payload for later retry."""
    try:
        os.makedirs(os.path.dirname(WEBHOOK_QUEUE_FILE), exist_ok=True)
        with open(WEBHOOK_QUEUE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        logging.exception("Failed to enqueue webhook payload")


def flush_webhook_queue() -> None:
    """Attempt to resend any queued webhook payloads."""
    if not os.path.exists(WEBHOOK_QUEUE_FILE) or not webhook_url:
        return
    remaining = []
    sent = 0
    with open(WEBHOOK_QUEUE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                requests.post(webhook_url, json=json.loads(line), timeout=10)
                sent += 1
            except Exception:
                WEBHOOK_ERRORS.inc()
                remaining.append(line)
    if remaining:
        with open(WEBHOOK_QUEUE_FILE, "w", encoding="utf-8") as f:
            f.writelines(remaining)
    else:
        os.remove(WEBHOOK_QUEUE_FILE)
    if sent:
        logging.info("Flushed %d queued webhook payloads", sent)


def post_webhook(problem: str, answer: str) -> None:
    """Send the result to the configured webhook with retry queue."""
    if not webhook_url:
        return
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "problem_statement": problem,
        "answer": answer,
        "source": "camera-to-chatgpt-pipeline",
    }
    flush_webhook_queue()
    start = time.time()
    try:
        requests.post(webhook_url, json=payload, timeout=10)
        latency = time.time() - start
        WEBHOOK_LATENCY.observe(latency)
        logging.info("Webhook posted in %.2fs", latency)
    except Exception:
        WEBHOOK_ERRORS.inc()
        enqueue_webhook_payload(payload)
        logging.warning("Webhook failed; payload queued")


def capture_and_send() -> None:
    """Capture, analyze, send to solver, and notify webhook."""
    global last_problem, last_answer
    last_problem = analyze_current_frame()
    last_answer = send_problem_to_chat(last_problem)
    post_webhook(last_problem, last_answer)


def auto_loop() -> None:
    """Background loop for automatic capture and solving."""
    while True:
        if AUTO_CAPTURE:
            capture_and_send()
        time.sleep(CAPTURE_INTERVAL_SEC)


def list_ollama_models(base_url: str) -> set:
    """Return a set of model names available from an Ollama server."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        return {m.get("name", "") for m in data.get("models", [])}
    except Exception:
        return set()


def wait_for_ollama_model(base_url: str, model: str, timeout_sec: int = 180) -> bool:
    """Wait until a specific Ollama model is available via /api/tags."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        models = list_ollama_models(base_url)
        if model in models:
            return True
        time.sleep(2)
    return False


def can_capture_once() -> bool:
    """Try to open the configured camera/RTSP and grab a frame once."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened() and CAMERA_RTSP_URL:
        cap.release()
        cap = cv2.VideoCapture(CAMERA_RTSP_URL)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return bool(ret)


def wait_for_camera_feed(timeout_sec: int = 45) -> bool:
    """Wait until the camera/RTSP feed yields a frame."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if can_capture_once():
            return True
        time.sleep(2)
    return False


def run_startup_checks() -> None:
    """Validate external dependencies before starting the server.

    Fails fast if required LLM models are missing or the camera feed is unavailable.
    """
    # If using Ollama-backed vision, ensure the vision model exists
    if VISION_MODE in ("auto", "ollama") and OLLAMA_BASE_URL and OLLAMA_MODEL:
        logging.info("Waiting for Ollama vision model '%s' at %s", OLLAMA_MODEL, OLLAMA_BASE_URL)
        if not wait_for_ollama_model(OLLAMA_BASE_URL, OLLAMA_MODEL):
            logging.error("Required vision model '%s' not available at %s", OLLAMA_MODEL, OLLAMA_BASE_URL)
            sys.exit(1)

    # If answering locally via Ollama, ensure the chat model exists
    if ANSWER_MODE == "local" and LOCAL_LLM_KIND == "ollama" and LOCAL_LLM_BASE_URL and LOCAL_LLM_MODEL:
        logging.info("Waiting for Ollama chat model '%s' at %s", LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL)
        if not wait_for_ollama_model(LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL):
            logging.error("Required chat model '%s' not available at %s", LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL)
            sys.exit(1)

    # If using OpenAI API, ensure key exists (basic sanity)
    if ANSWER_MODE == "api" and not OPENAI_API_KEY:
        logging.error("ANSWER_MODE=api but OPENAI_API_KEY is not set")
        sys.exit(1)

    # Ensure a frame can be captured (index or RTSP)
    logging.info("Checking camera/RTSP feed availability...")
    if not wait_for_camera_feed():
        logging.error("Camera/RTSP feed is not available. Set CAMERA_RTSP_URL or adjust CAMERA_INDEX.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# API endpoint to solve provided problems
# ---------------------------------------------------------------------------


class SolveRequest(BaseModel):
    problem_statement: str


class SolveResponse(BaseModel):
    problem_statement: str
    answer: str


@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest) -> SolveResponse:
    answer = send_problem_to_chat(req.problem_statement)
    return SolveResponse(problem_statement=req.problem_statement, answer=answer)


# ---------------------------------------------------------------------------
# Web UI and control endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Simple HTML control panel."""
    return f"""
    <html>
    <body>
    <h1>Camera LLM</h1>
    <p>Auto capture: {AUTO_CAPTURE}</p>
    <p>Last problem: {last_problem}</p>
    <p>Last answer: {last_answer}</p>
    <form action='/toggle_auto' method='post'><button type='submit'>Toggle Auto</button></form>
    <form action='/capture_now' method='post'><button type='submit'>Capture Now</button></form>
    <form action='/send' method='post'><button type='submit'>Send</button></form>
    <div style="margin-top:16px; border-top:1px solid #ccc; padding-top:12px;">
      <h3>Last Capture</h3>
      <div style="display:{'block' if last_image_original_b64 else 'none'}; margin-bottom:8px;">
        <div><strong>Original</strong></div>
        <img src="{last_image_original_b64}" alt="original" style="max-width: 48vw; border:1px solid #ddd;" />
      </div>
      <div style="display:{'block' if last_image_transformed_b64 else 'none'};">
        <div><strong>Transformed</strong></div>
        <img src="{last_image_transformed_b64}" alt="transformed" style="max-width: 48vw; border:1px solid #ddd;" />
      </div>
    </div>
    </body>
    </html>
    """


@app.post("/toggle_auto")
async def toggle_auto() -> dict:
    """Toggle automatic capture."""
    global AUTO_CAPTURE
    AUTO_CAPTURE = not AUTO_CAPTURE
    return {"auto_capture": AUTO_CAPTURE}


@app.post("/capture_now")
async def capture_now() -> RedirectResponse:
    """Manually capture and analyze a frame, then return to index to display images."""
    global last_problem
    last_problem = analyze_current_frame()
    return RedirectResponse(url="/", status_code=303)


@app.post("/send")
async def send() -> RedirectResponse:
    """Send the last captured problem to the solver and return to index page."""
    global last_answer
    if last_problem:
        last_answer = send_problem_to_chat(last_problem)
        post_webhook(last_problem, last_answer)
    else:
        # If no problem captured yet, keep last_answer as-is
        pass
    # Redirect back to the index so the page shows updated results
    return RedirectResponse(url="/", status_code=303)


class WebhookRequest(BaseModel):
    url: str


@app.post("/set_webhook")
async def set_webhook(req: WebhookRequest) -> dict:
    """Update the webhook URL."""
    global webhook_url
    webhook_url = req.url
    return {"webhook_url": webhook_url}


if __name__ == "__main__":
    run_startup_checks()
    threading.Thread(target=auto_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
