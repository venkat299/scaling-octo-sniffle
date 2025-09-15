"""Minimal camera-to-LLM pipeline with local model support."""

import base64
import json
import logging
import os
import threading
import subprocess
import tempfile
import shutil
import time
import sys
from datetime import datetime
from typing import Optional, List

import cv2
import numpy as np
import pytesseract
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from pydantic import BaseModel
from openai import OpenAI
from lmstudio_client import send_to_lmstudio, lmstudio_vision_extract
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

CAPTURE_INTERVAL_SEC = int(os.getenv("CAPTURE_INTERVAL_SEC", "10"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_RTSP_URL = os.getenv("CAMERA_RTSP_URL", "")
MACOS_CAPTURE_TOOL = os.getenv("MACOS_CAPTURE_TOOL", "auto")  # auto | imagesnap | ffmpeg | off
IMAGESNAP_DEVICE = os.getenv("IMAGESNAP_DEVICE", "")  # optional device name for imagesnap
IMAGESNAP_WARMUP = os.getenv("IMAGESNAP_WARMUP", "1")  # seconds to warm up imagesnap
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
AVFOUNDATION_DEVICE = os.getenv("AVFOUNDATION_DEVICE", "0:")  # e.g. "0:" video only
AVFOUNDATION_FRAMERATE = os.getenv("AVFOUNDATION_FRAMERATE", "30")
AVFOUNDATION_SIZE = os.getenv("AVFOUNDATION_SIZE", "")  # e.g. "1280x720" (optional)

# Image transform controls
# Default to false since many cameras already provide an unmirrored feed.
MIRROR_HORIZONTAL = os.getenv("MIRROR_HORIZONTAL", "false").lower() in ("1", "true", "yes", "on")
ROTATE_DEG = float(os.getenv("ROTATE_DEG", "0"))
# For perspective keystone correction (camera low, tilted up => top appears narrower)
# Value in [0, 0.4]; 0 disables and is the default.
KEYSTONE_TOP_INSET_PCT = float(os.getenv("KEYSTONE_TOP_INSET_PCT", "0.0"))
KEYSTONE_BOTTOM_INSET_PCT = float(os.getenv("KEYSTONE_BOTTOM_INSET_PCT", "0.0"))

ANSWER_MODE = os.getenv("ANSWER_MODE", "api")  # api | local | browser
LOCAL_LLM_KIND = os.getenv("LOCAL_LLM_KIND", "ollama")  # ollama | openai_compat
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")

VISION_MODE = os.getenv("VISION_MODE", "auto")  # auto | ocr | ollama | lmstudio
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
AUTO_SEND = False
MULTI_CAPTURE = False
last_problem_parts: List[str] = []
last_problem: str = ""
last_answer: str = ""
webhook_url = WEBHOOK_URL
last_image_original_b64: str = ""
last_image_transformed_b64: str = ""
# Keep raw JPEG bytes for direct download/debug viewing
last_image_original_bytes: Optional[bytes] = None
last_image_transformed_bytes: Optional[bytes] = None
processing_state: str = "idle"
last_camera_source: str = ""

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
    auto_send: bool
    multi_capture: bool
    last_problem: str
    last_answer: str
    webhook_url: str
    processing_state: str
    answer_mode: str
    answer_model: str
    vision_mode: str
    vision_model: str
    camera_source: str
    has_original: bool
    has_transformed: bool


@app.get("/status", response_model=Status)
async def get_status() -> Status:
    """Return basic status including capture interval."""
    # Resolve model labels based on configured modes
    if ANSWER_MODE == "lmstudio":
        answer_model = os.getenv("LMSTUDIO_MODEL", "") or ""
    elif ANSWER_MODE == "local":
        answer_model = LOCAL_LLM_MODEL
    elif ANSWER_MODE == "api":
        answer_model = OPENAI_MODEL
    else:
        answer_model = ""

    if VISION_MODE == "lmstudio":
        vision_model = os.getenv("LMSTUDIO_VISION_MODEL", "") or os.getenv("LMSTUDIO_MODEL", "") or ""
    elif VISION_MODE == "ollama":
        vision_model = OLLAMA_MODEL
    elif VISION_MODE == "ocr":
        vision_model = "tesseract"
    else:
        vision_model = "auto"

    return Status(
        interval_sec=CAPTURE_INTERVAL_SEC,
        auto_capture=AUTO_CAPTURE,
        auto_send=AUTO_SEND,
        multi_capture=MULTI_CAPTURE,
        last_problem=last_problem,
        last_answer=last_answer,
        webhook_url=webhook_url,
        processing_state=processing_state,
        answer_mode=ANSWER_MODE,
        answer_model=answer_model,
        vision_mode=VISION_MODE,
        vision_model=vision_model,
        camera_source=last_camera_source,
        has_original=last_image_original_bytes is not None,
        has_transformed=last_image_transformed_bytes is not None,
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
    """Apply optional mirror, rotation, and keystone correction to the frame.

    - Mirror horizontally if configured.
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


def capture_macos_jpeg() -> Optional[bytes]:
    """Capture one JPEG using macOS CLI tools (imagesnap or ffmpeg/avfoundation).

    Returns JPEG bytes or None if unavailable. This can trigger macOS camera
    permission prompts which sometimes do not appear for OpenCV processes.
    """
    if sys.platform != "darwin" or MACOS_CAPTURE_TOOL == "off":
        return None

    # Try imagesnap first unless explicitly set to ffmpeg
    if MACOS_CAPTURE_TOOL in ("auto", "imagesnap"):
        imagesnap = shutil.which("imagesnap")
        if imagesnap:
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    tmp_path = tf.name
                try:
                    cmd = [imagesnap, "-q", "-w", IMAGESNAP_WARMUP]
                    if IMAGESNAP_DEVICE:
                        cmd += ["-d", IMAGESNAP_DEVICE]
                    cmd += [tmp_path]
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=15,
                    )
                    with open(tmp_path, "rb") as f:
                        data = f.read()
                    return data if data else None
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            except Exception:
                logging.exception("imagesnap capture failed")

    # Fallback: ffmpeg avfoundation
    if MACOS_CAPTURE_TOOL in ("auto", "ffmpeg"):
        ffmpeg = shutil.which(FFMPEG_BIN)
        if ffmpeg:
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    tmp_path = tf.name
                try:
                    cmd = [
                        ffmpeg,
                        "-y",
                        "-f",
                        "avfoundation",
                        "-framerate",
                        AVFOUNDATION_FRAMERATE,
                    ]
                    if AVFOUNDATION_SIZE:
                        cmd += ["-video_size", AVFOUNDATION_SIZE]
                    cmd += [
                        "-i",
                        AVFOUNDATION_DEVICE,
                        "-frames:v",
                        "1",
                        "-vcodec",
                        "mjpeg",
                        tmp_path,
                    ]
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=20,
                    )
                    with open(tmp_path, "rb") as f:
                        data = f.read()
                    return data if data else None
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            except Exception:
                logging.exception("ffmpeg avfoundation capture failed")

    return None


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
    """Derive a problem statement from an image using LM Studio vision or OCR.

    Order of attempts:
    - If VISION_MODE=lmstudio or auto: try LM Studio vision first.
    - If VISION_MODE=ollama or auto (and not found): try Ollama vision.
    - If VISION_MODE=ocr (and not found in previous): use Tesseract OCR.
    """
    summary: Optional[str] = None

    # LM Studio multimodal (preferred when enabled)
    if VISION_MODE in ("auto", "lmstudio"):
        try:
            summary = lmstudio_vision_extract(jpeg)
        except Exception:
            logging.exception("LM Studio vision extraction errored")

    # Ollama vision fallback
    if (VISION_MODE in ("auto", "ollama")) and (not summary or not summary.strip()):
        summary = ollama_vision_summarize(jpeg)

    # OCR fallback only if explicitly requested or nothing found in auto
    if (VISION_MODE in ("auto", "ocr")) and (not summary or not summary.strip()):
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
        elif ANSWER_MODE == "lmstudio":
            backend = "lmstudio"
            answer = send_to_lmstudio(problem_statement)
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
    """Capture one frame and return (orig_jpeg, transformed_jpeg, transformed_frame).

    Attempts OpenCV first; on failure (common on macOS due to privacy
    permissions), falls back to macOS CLI tools if available.
    """
    # Try OpenCV device or RTSP first
    global last_camera_source
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened() and CAMERA_RTSP_URL:
            cap.release()
            cap = cv2.VideoCapture(CAMERA_RTSP_URL)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                orig = frame.copy()
                transformed = transform_frame(orig.copy())
                ok1, buf1 = cv2.imencode(".jpg", orig)
                ok2, buf2 = cv2.imencode(".jpg", transformed)
                if ok1 and ok2:
                    # Determine source label
                    if CAMERA_RTSP_URL and CAMERA_RTSP_URL.strip():
                        last_camera_source = f"opencv:rtsp {CAMERA_RTSP_URL}"
                    else:
                        last_camera_source = f"opencv:index {CAMERA_INDEX}"
                    return buf1.tobytes(), buf2.tobytes(), transformed
    except Exception:
        try:
            cap.release()
        except Exception:
            pass

    # Fallback to macOS CLI capture
    mac_jpeg = capture_macos_jpeg()
    if mac_jpeg:
        arr = np.frombuffer(mac_jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        orig = frame.copy()
        transformed = transform_frame(orig.copy())
        ok2, buf2 = cv2.imencode(".jpg", transformed)
        if ok2:
            # Record macOS source info
            if MACOS_CAPTURE_TOOL in ("auto", "imagesnap") and shutil.which("imagesnap"):
                dev = IMAGESNAP_DEVICE or "default"
                last_camera_source = f"macos:imagesnap {dev}"
            elif MACOS_CAPTURE_TOOL in ("auto", "ffmpeg") and shutil.which(FFMPEG_BIN):
                last_camera_source = f"macos:ffmpeg avfoundation {AVFOUNDATION_DEVICE}"
            else:
                last_camera_source = "macos:unknown"
            return mac_jpeg, buf2.tobytes(), transformed

    return None


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
        # Persist images for display on index page and for direct download
        global last_image_original_b64, last_image_transformed_b64, last_image_original_bytes, last_image_transformed_bytes
        last_image_original_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_jpeg).decode("utf-8")
        last_image_transformed_b64 = "data:image/jpeg;base64," + base64.b64encode(transformed_jpeg).decode("utf-8")
        last_image_original_bytes = orig_jpeg
        last_image_transformed_bytes = transformed_jpeg
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


def capture_once() -> None:
    """Capture and analyze a frame and optionally send to solver."""
    global last_problem, last_answer, processing_state, last_problem_parts
    processing_state = "analyzing"
    try:
        new_text = analyze_current_frame()
        if MULTI_CAPTURE:
            last_problem_parts.append(new_text)
            last_problem = "\n".join(last_problem_parts)
        else:
            last_problem = new_text
        if AUTO_SEND and not MULTI_CAPTURE:
            processing_state = "solving"
            last_answer = send_problem_to_chat(last_problem)
            post_webhook(last_problem, last_answer)
    finally:
        processing_state = "idle"


def auto_loop() -> None:
    """Background loop for automatic capture and solving."""
    while True:
        if AUTO_CAPTURE and not MULTI_CAPTURE:
            capture_once()
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
    """Try to open the configured camera/RTSP and grab a frame once.

    Includes macOS CLI fallback so startup checks pass when OpenCV cannot
    trigger camera permissions.
    """
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened() and CAMERA_RTSP_URL:
            cap.release()
            cap = cv2.VideoCapture(CAMERA_RTSP_URL)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return True
    except Exception:
        try:
            cap.release()
        except Exception:
            pass

    # macOS fallback
    if sys.platform == "darwin" and capture_macos_jpeg():
        return True
    return False


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
    """Responsive UI using Tailwind CSS."""
    return f"""
    <html>
    <head>
      <meta name='viewport' content='width=device-width, initial-scale=1.0'>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="h-screen flex text-sm">
      <div id="content" class="flex-1 overflow-auto p-2">
        <div id="problem" class="tab-pane"><pre id='last_problem' class="whitespace-pre-wrap break-words"></pre></div>
        <div id="solution" class="tab-pane"><pre id='last_answer' class="whitespace-pre-wrap break-words"></pre></div>
        <div id="status" class="tab-pane space-y-1">
          <div>Processing: <span id='processing_state'></span></div>
          <div>Camera: <span id='camera_source'></span></div>
          <div>Answer mode: <span id='answer_mode'></span> <small id='answer_model'></small></div>
          <div>Vision mode: <span id='vision_mode'></span> <small id='vision_model'></small></div>
          <div>Auto capture: <span id='auto_capture'></span></div>
          <div>Auto send: <span id='auto_send'></span></div>
          <div>Multi capture: <span id='multi_capture'></span></div>
        </div>
        <div id="images" class="tab-pane space-y-2">
          <div id="orig_block" class="hidden">
            <div class="text-xs font-bold">Original - <a href='/original.jpg' target='_blank' class="text-blue-600 underline">open</a></div>
            <img id="img_original" class="w-full rounded" alt="original"/>
          </div>
          <div id="trans_block" class="hidden">
            <div class="text-xs font-bold">Transformed - <a href='/transformed.jpg' target='_blank' class="text-blue-600 underline">open</a></div>
            <img id="img_transformed" class="w-full rounded" alt="transformed"/>
          </div>
          <div class="text-[10px] font-mono text-gray-500">
            MIRROR_HORIZONTAL={str(MIRROR_HORIZONTAL).lower()} | ROTATE_DEG={ROTATE_DEG} | KEYSTONE_TOP_INSET_PCT={KEYSTONE_TOP_INSET_PCT} | KEYSTONE_BOTTOM_INSET_PCT={KEYSTONE_BOTTOM_INSET_PCT}
          </div>
        </div>
      </div>
      <div id="sidebar" class="w-28 flex flex-col border-l border-gray-300 p-2 justify-between">
        <div id="tabs" class="flex flex-col space-y-1">
          <button data-tab="problem" class="text-xs py-1 px-2 border rounded">Problem</button>
          <button data-tab="solution" class="text-xs py-1 px-2 border rounded">Solution</button>
          <button data-tab="status" class="text-xs py-1 px-2 border rounded">System</button>
          <button data-tab="images" class="text-xs py-1 px-2 border rounded">Images</button>
        </div>
        <div id="controls" class="mt-4 flex flex-col space-y-1">
          <button id='capture_btn' class="text-xs py-1 px-2 bg-blue-600 text-white rounded">Capture</button>
          <button id='send_btn' class="text-xs py-1 px-2 bg-green-600 text-white rounded">Send</button>
          <button id='toggle_auto_capture_btn' class="text-xs py-1 px-2 border rounded">Auto Capture</button>
          <button id='toggle_auto_send_btn' class="text-xs py-1 px-2 border rounded">Auto Send</button>
          <button id='toggle_multi_btn' class="text-xs py-1 px-2 border rounded">Multi</button>
        </div>
      </div>
    <script>
    function activateTab(id) {{
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
      document.getElementById(id).classList.remove('hidden');
    }}
    document.querySelectorAll('#tabs button').forEach(btn => {{
      btn.addEventListener('click', () => activateTab(btn.dataset.tab));
    }});
    activateTab('problem');

    async function refreshStatus() {{
      try {{
        const r = await fetch('/status');
        const s = await r.json();
        document.getElementById('last_problem').innerText = s.last_problem;
        document.getElementById('last_answer').innerText = s.last_answer;
        document.getElementById('processing_state').innerText = s.processing_state;
        document.getElementById('camera_source').innerText = s.camera_source || '';
        document.getElementById('answer_mode').innerText = s.answer_mode;
        document.getElementById('answer_model').innerText = s.answer_model ? '(' + s.answer_model + ')' : '';
        document.getElementById('vision_mode').innerText = s.vision_mode;
        document.getElementById('vision_model').innerText = s.vision_model ? '(' + s.vision_model + ')' : '';
        document.getElementById('auto_capture').innerText = s.auto_capture;
        document.getElementById('auto_send').innerText = s.auto_send;
        document.getElementById('multi_capture').innerText = s.multi_capture;
        if (s.has_original) {{
          document.getElementById('orig_block').classList.remove('hidden');
          document.getElementById('img_original').src = '/original.jpg?ts=' + Date.now();
        }}
        if (s.has_transformed) {{
          document.getElementById('trans_block').classList.remove('hidden');
          document.getElementById('img_transformed').src = '/transformed.jpg?ts=' + Date.now();
        }}
      }} catch (e) {{
        console.error('status update failed', e);
      }}
    }}

    async function doCapture() {{
      const r = await fetch('/capture_now', {{method:'POST'}});
      const d = await r.json();
      await refreshStatus();
      activateTab(d.tab || 'problem');
    }}

    async function doSend() {{
      const r = await fetch('/send', {{method:'POST'}});
      const d = await r.json();
      await refreshStatus();
      activateTab(d.tab || 'solution');
    }}

    async function toggleAutoCapture() {{
      await fetch('/toggle_auto_capture', {{method:'POST'}});
      await refreshStatus();
    }}

    async function toggleAutoSend() {{
      await fetch('/toggle_auto_send', {{method:'POST'}});
      await refreshStatus();
    }}

    async function toggleMulti() {{
      await fetch('/toggle_multi', {{method:'POST'}});
      await refreshStatus();
    }}

    document.getElementById('capture_btn').onclick = doCapture;
    document.getElementById('send_btn').onclick = doSend;
    document.getElementById('toggle_auto_capture_btn').onclick = toggleAutoCapture;
    document.getElementById('toggle_auto_send_btn').onclick = toggleAutoSend;
    document.getElementById('toggle_multi_btn').onclick = toggleMulti;

    setInterval(refreshStatus, 1000);
    refreshStatus();
    </script>
    </body>
    </html>
    """


@app.get("/original.jpg")
async def get_original_jpeg() -> Response:
    """Return the last raw-captured JPEG."""
    if last_image_original_bytes:
        return Response(content=last_image_original_bytes, media_type="image/jpeg")
    return PlainTextResponse("No image captured yet", status_code=404)


@app.get("/transformed.jpg")
async def get_transformed_jpeg() -> Response:
    """Return the last transformed JPEG."""
    if last_image_transformed_bytes:
        return Response(content=last_image_transformed_bytes, media_type="image/jpeg")
    return PlainTextResponse("No image captured yet", status_code=404)


@app.post("/toggle_auto_capture")
async def toggle_auto_capture() -> dict:
    """Toggle automatic capture."""
    global AUTO_CAPTURE
    AUTO_CAPTURE = not AUTO_CAPTURE
    return {"auto_capture": AUTO_CAPTURE}


@app.post("/toggle_auto_send")
async def toggle_auto_send() -> dict:
    """Toggle automatic sending after capture."""
    global AUTO_SEND
    AUTO_SEND = not AUTO_SEND
    return {"auto_send": AUTO_SEND}


@app.post("/toggle_multi")
async def toggle_multi() -> dict:
    """Toggle multi-capture mode (disables auto modes)."""
    global MULTI_CAPTURE, AUTO_CAPTURE, AUTO_SEND, last_problem_parts
    MULTI_CAPTURE = not MULTI_CAPTURE
    if MULTI_CAPTURE:
        AUTO_CAPTURE = False
        AUTO_SEND = False
        last_problem_parts = []
    return {
        "multi_capture": MULTI_CAPTURE,
        "auto_capture": AUTO_CAPTURE,
        "auto_send": AUTO_SEND,
    }


@app.post("/capture_now")
async def capture_now() -> dict:
    """Manually capture and analyze a frame."""
    capture_once()
    tab = "solution" if (AUTO_SEND and not MULTI_CAPTURE) else "problem"
    return {"tab": tab}


@app.post("/send")
async def send() -> dict:
    """Send the last captured problem to the solver."""
    global last_answer, processing_state, last_problem_parts
    if last_problem:
        processing_state = "solving"
        try:
            last_answer = send_problem_to_chat(last_problem)
            post_webhook(last_problem, last_answer)
        finally:
            processing_state = "idle"
        last_problem_parts = []
    return {"tab": "solution"}


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
