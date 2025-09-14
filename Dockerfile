FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for Tesseract, OpenCV, and camera access
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev libleptonica-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application code
COPY camera-to-chatgpt-pipeline.py /app/app.py

# Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose web UI
EXPOSE 8000

# Default environment variables
ENV CAPTURE_INTERVAL_SEC=10 \
    CAMERA_INDEX=0 \
    CAMERA_RTSP_URL= \
    ANSWER_MODE=local \
    LOCAL_LLM_KIND=ollama \
    LOCAL_LLM_BASE_URL=http://ollama:11434 \
    LOCAL_LLM_MODEL=qwen2.5:7b \
    VISION_MODE=auto \
    OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=llava:7b \
    WEBHOOK_URL=http://webhook:9000/inbox \
    WEBHOOK_QUEUE_FILE=/data/webhook_queue.jsonl \
    BROWSER_MODE=false

# Healthcheck for /status
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
  python - <<'PY' || exit 1
import urllib.request, json
try:
  with urllib.request.urlopen('http://127.0.0.1:8000/status', timeout=3) as r:
    data=json.load(r)
    assert 'interval_sec' in data
except Exception:
  raise SystemExit(1)
PY

CMD ["python", "/app/app.py"]
