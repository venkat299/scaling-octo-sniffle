"""LM Studio client for OpenAI-compatible chat completions and vision.

Provides:
- `send_to_lmstudio(problem_statement: str) -> str` — solve text problems.
- `lmstudio_vision_extract(jpeg_bytes: bytes) -> str` — extract visible text and
  identify the problem from an image using LM Studio (multimodal model).

Prompt style for solving:
- Short, numbered steps; terse wording.
- For coding problems, return minimal code with brief end-of-line comments.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional, Tuple

import requests
from PIL import Image
import io


# Configuration via environment
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "")  # optional; LM Studio often doesn't require
LMSTUDIO_TIMEOUT_SEC = int(os.getenv("LMSTUDIO_TIMEOUT_SEC", "120"))
LMSTUDIO_VISION_MODEL = os.getenv("LMSTUDIO_VISION_MODEL", LMSTUDIO_MODEL)
LMSTUDIO_MAX_TOKENS = int(os.getenv("LMSTUDIO_MAX_TOKENS", "512"))
LMSTUDIO_TEMPERATURE = float(os.getenv("LMSTUDIO_TEMPERATURE", "0.2"))
LMSTUDIO_VISION_MAX_TOKENS = int(os.getenv("LMSTUDIO_VISION_MAX_TOKENS", "512"))
LMSTUDIO_VISION_IMAGE_SIZE = int(os.getenv("LMSTUDIO_VISION_IMAGE_SIZE", "896"))
LMSTUDIO_VISION_PAD_COLOR = os.getenv("LMSTUDIO_VISION_PAD_COLOR", "#000000")


def _make_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if LMSTUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"
    return headers


def _system_prompt() -> str:
    return (
        "You are a precise assistant.\n"
        "- If it is not a leetcode stype problem, gieve the final anwer first and Answer with short, numbered steps.\n"
        "- Be terse; avoid verbose explanations.\n"
        "- If it is a coding problem, include a minimal code block with terse end-of-line comments.\n"
        "- Do not add extra commentary beyond the solution."
    )


def send_to_lmstudio(problem_statement: str) -> str:
    """Send `problem_statement` to a local LM Studio server and return the answer.

    Uses the OpenAI-compatible Chat Completions API exposed by LM Studio.
    Controlled by env vars: LMSTUDIO_BASE_URL, LMSTUDIO_MODEL, LMSTUDIO_API_KEY.
    """
    if not LMSTUDIO_MODEL:
        return "Solver error: LMSTUDIO_MODEL is not set"

    url = f"{LMSTUDIO_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": (
                    "Identify the problem from the input and solve it.\n"
                    "Follow the style guide above. Be concise.\n\n"
                    f"Input:\n{problem_statement}"
                ),
            },
        ],
        "temperature": LMSTUDIO_TEMPERATURE,
        "max_tokens": LMSTUDIO_MAX_TOKENS,
    }

    headers = _make_headers()
    last_err: Optional[str] = None
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=LMSTUDIO_TIMEOUT_SEC)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(2 ** attempt)
    return f"Solver error: {last_err or 'unknown error'}"


def _vision_system_prompt() -> str:
    return (
        "Extract the visible text which is/are problems to solve.\n"
        "Output the exact text related to the problem\n"
        "questions will be of type 1. leetcode type, technical questions, multichoice or aptitude/math\n"
        "sometime there can be multiple questions\n"
        "ignore noisy text which are not relevant\n"
    )


def lmstudio_vision_extract(jpeg_bytes: bytes) -> str:
    """Send one image to LM Studio and get a terse extraction/identification.

    Requires a multimodal model served by LM Studio. If the configured model is
    not multimodal, the server may return an error or ignore the image.
    """
    model = LMSTUDIO_VISION_MODEL or LMSTUDIO_MODEL
    if not model:
        return ""

    import base64 as _b64

    prepped = _prepare_image_for_gemma(jpeg_bytes)
    b64 = "data:image/jpeg;base64," + _b64.b64encode(prepped).decode("utf-8")
    url = f"{LMSTUDIO_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _vision_system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image."},
                    {"type": "image_url", "image_url": {"url": b64}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": LMSTUDIO_VISION_MAX_TOKENS,
    }
    headers = _make_headers()
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=LMSTUDIO_TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        logging.exception("LM Studio vision extraction failed: %s", e)
        return ""

def _parse_pad_color(s: str) -> Tuple[int, int, int]:
    try:
        if s.startswith("#") and len(s) == 7:
            return tuple(int(s[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    except Exception:
        pass
    return (0, 0, 0)


def _prepare_image_for_gemma(jpeg_bytes: bytes) -> bytes:
    """Resize + letterbox image to LMStudio/Gemma-3 preferred 896x896.

    Preserves aspect ratio, pads with LMSTUDIO_VISION_PAD_COLOR, returns JPEG bytes.
    """
    size = max(64, min(4096, LMSTUDIO_VISION_IMAGE_SIZE))
    pad_color = _parse_pad_color(LMSTUDIO_VISION_PAD_COLOR)
    im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    w, h = im.size
    scale = min(size / max(1, w), size / max(1, h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), pad_color)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas.paste(im_resized, (x, y))
    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=90)
    return out.getvalue()
