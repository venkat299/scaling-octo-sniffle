"""Minimal camera-to-LLM pipeline with local model support."""

import base64
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import pytesseract
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
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


def capture_frame() -> Optional[tuple[bytes, np.ndarray]]:
    """Capture a frame from the configured camera."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened() and CAMERA_RTSP_URL:
        cap.release()
        cap = cv2.VideoCapture(CAMERA_RTSP_URL)
    if not cap.isOpened():
        cap.release()
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    success, buf = cv2.imencode(".jpg", frame)
    if not success:
        return None
    return buf.tobytes(), frame


def analyze_current_frame() -> str:
    """Capture a frame, extract a problem statement, and log latency."""
    start = time.time()
    try:
        data = capture_frame()
        if not data:
            ANALYZER_ERRORS.inc()
            logging.info("Analyzer found no frame in %.2fs", time.time() - start)
            return "NO PROBLEM FOUND"
        jpeg, frame = data
        summary = extract_problem_statement(jpeg, frame)
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


# Start background thread
threading.Thread(target=auto_loop, daemon=True).start()


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
async def capture_now() -> dict:
    """Manually capture and analyze a frame."""
    global last_problem
    last_problem = analyze_current_frame()
    return {"problem_statement": last_problem}


@app.post("/send", response_model=SolveResponse)
async def send() -> SolveResponse:
    """Send the last captured problem to the solver."""
    global last_answer
    if not last_problem:
        return SolveResponse(problem_statement="", answer="No problem captured")
    last_answer = send_problem_to_chat(last_problem)
    post_webhook(last_problem, last_answer)
    return SolveResponse(problem_statement=last_problem, answer=last_answer)


class WebhookRequest(BaseModel):
    url: str


@app.post("/set_webhook")
async def set_webhook(req: WebhookRequest) -> dict:
    """Update the webhook URL."""
    global webhook_url
    webhook_url = req.url
    return {"webhook_url": webhook_url}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
