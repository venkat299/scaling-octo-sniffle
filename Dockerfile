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
COPY lmstudio_client.py /app/lmstudio_client.py

# Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose web UI
EXPOSE 8000

# Default environment variables
ENV CAPTURE_INTERVAL_SEC=10 \
    CAMERA_INDEX=0 \
    CAMERA_RTSP_URL= \
    MIRROR_HORIZONTAL=false \
    ROTATE_DEG=0 \
    KEYSTONE_TOP_INSET_PCT=0.0 \
    KEYSTONE_BOTTOM_INSET_PCT=0.0 \
    # local | lmstudio | api | browser
    ANSWER_MODE=lmstudio \
    LMSTUDIO_MODEL="openai/gpt-oss-20b" \
    # "http://host.docker.internal:1234"
    LMSTUDIO_BASE_URL="http://localhost:1234" \
    LMSTUDIO_API_KEY= \
    LMSTUDIO_MAX_TOKENS=512 \
    LMSTUDIO_TEMPERATURE=0.2 \
    LMSTUDIO_VISION_MODEL="google/gemma-3-4b" \
    LMSTUDIO_VISION_MAX_TOKENS=512 \
    LMSTUDIO_VISION_IMAGE_SIZE=896 \
    LMSTUDIO_VISION_PAD_COLOR="#000000" \
    LOCAL_LLM_KIND=ollama \
    LOCAL_LLM_BASE_URL=http://ollama:11434 \
    LOCAL_LLM_MODEL=qwen2.5:7b \
    VISION_MODE=lmstudio \
    OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=llava:7b \
    WEBHOOK_URL=http://webhook:80/inbox \
    WEBHOOK_QUEUE_FILE=/data/webhook_queue.jsonl \
    BROWSER_MODE=false \
    IMAGESNAP_DEVICE="Angetube Live Camera" \
    MACOS_CAPTURE_TOOL=imagesnap

# Healthcheck for /status (single-line exec form)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python", "-c", "import urllib.request, json, sys\ntry:\n    with urllib.request.urlopen('http://127.0.0.1:8000/status', timeout=3) as r:\n        data = json.load(r)\n        sys.exit(0 if 'interval_sec' in data else 1)\nexcept Exception:\n    sys.exit(1)"]

CMD ["python", "/app/app.py"]
