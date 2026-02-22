"""
Cats vs Dogs — Inference API
────────────────────────────
Endpoints
  GET  /health      liveness check
  GET  /metrics     Prometheus scrape endpoint
  GET  /stats       JSON summary of in-process counters
  POST /predict     image → {label, confidence, probabilities}
  POST /feedback    record ground-truth label for a previous prediction
                    (used by the performance tracker for post-deployment accuracy)
"""

import io
import json
import logging
import os
import time
import threading
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY,
)

from src.model import SimpleCNN

# ── Logging ────────────────────────────────────────────────────────────────────
# Emit structured JSON lines so log aggregators (Loki, CloudWatch, etc.) can
# parse fields without regex.  Sensitive data (file contents) is never logged.
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base)

_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger("catsdogs.api")

# ── Config ─────────────────────────────────────────────────────────────────────
APP_NAME   = "cats-dogs-inference"
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pt")
DEVICE     = os.getenv("DEVICE", "cpu")
CLASS_NAMES = ["cats", "dogs"]

# ── Prometheus metrics ─────────────────────────────────────────────────────────
PREDICT_REQUESTS = Counter(
    "predict_requests_total",
    "Total number of /predict requests",
    ["status"],           # labels: success | error
)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "End-to-end latency of /predict requests",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICT_CONFIDENCE = Histogram(
    "predict_confidence",
    "Model confidence score for each prediction",
    ["label"],            # label: cats | dogs
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
MODEL_ERRORS = Counter(
    "model_errors_total",
    "Inference errors (bad image, model not loaded, etc.)",
)
FEEDBACK_CORRECT = Counter(
    "feedback_correct_total",
    "Feedback events where prediction matched true label",
)
FEEDBACK_TOTAL = Counter(
    "feedback_total",
    "Total feedback events received",
)
MODEL_ACCURACY_GAUGE = Gauge(
    "model_post_deploy_accuracy",
    "Rolling post-deployment accuracy from /feedback calls",
)

# Thread-safe in-process counters (mirrors Prometheus for the /stats endpoint)
_lock = threading.Lock()
_stats: Dict = {
    "request_count": 0,
    "error_count": 0,
    "total_latency_sec": 0.0,
    "label_counts": defaultdict(int),
    "feedback_total": 0,
    "feedback_correct": 0,
}

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL: Optional[SimpleCNN] = None

def preprocess_pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))          # HWC → CHW
    return torch.tensor(arr).unsqueeze(0)        # 1×3×224×224

def load_model() -> SimpleCNN:
    model = SimpleCNN(num_classes=2)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first and place weights there."
        )
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd)
    model.to(DEVICE)
    model.eval()
    return model

# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    MODEL = load_model()
    logger.info({"event": "model_loaded", "path": MODEL_PATH, "device": DEVICE})
    yield
    logger.info({"event": "shutdown"})

app = FastAPI(title=APP_NAME, lifespan=lifespan)

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": APP_NAME}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus scrape endpoint — returns metrics in text/plain exposition format."""
    return PlainTextResponse(
        generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/stats")
def stats() -> Dict:
    """Human-readable JSON summary of in-process counters (no auth required)."""
    with _lock:
        rc  = _stats["request_count"]
        err = _stats["error_count"]
        lat = _stats["total_latency_sec"]
        fb_total   = _stats["feedback_total"]
        fb_correct = _stats["feedback_correct"]
    return {
        "request_count":    rc,
        "error_count":      err,
        "avg_latency_ms":   round((lat / max(rc, 1)) * 1000, 2),
        "label_counts":     dict(_stats["label_counts"]),
        "feedback_total":   fb_total,
        "feedback_correct": fb_correct,
        "post_deploy_accuracy": round(fb_correct / max(fb_total, 1), 4),
    }


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    request_count: int
    avg_latency_ms: float


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        MODEL_ERRORS.inc()
        raise HTTPException(status_code=500, detail="Model not loaded")

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        MODEL_ERRORS.inc()
        raise HTTPException(
            status_code=400, detail=f"Unsupported content type: {file.content_type}"
        )

    t0 = time.time()
    content = await file.read()

    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        MODEL_ERRORS.inc()
        PREDICT_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Could not decode image file")

    x = preprocess_pil_to_tensor(img).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx   = int(np.argmax(probs))
        label      = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

    latency = time.time() - t0

    # ── Prometheus ──
    PREDICT_REQUESTS.labels(status="success").inc()
    PREDICT_LATENCY.observe(latency)
    PREDICT_CONFIDENCE.labels(label=label).observe(confidence)

    # ── In-process counters ──
    with _lock:
        _stats["request_count"]      += 1
        _stats["total_latency_sec"]  += latency
        _stats["label_counts"][label] += 1
        rc  = _stats["request_count"]
        lat = _stats["total_latency_sec"]

    prob_map      = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    avg_latency_ms = (lat / max(rc, 1)) * 1000.0

    # Structured log — no file content, no raw bytes
    logger.info(json.dumps({
        "event":       "predict",
        "request_id":  rc,
        "filename":    file.filename,
        "label":       label,
        "confidence":  round(confidence, 4),
        "latency_ms":  round(latency * 1000, 2),
        "content_type": file.content_type,
    }))

    return PredictResponse(
        label=label,
        confidence=confidence,
        probabilities=prob_map,
        request_count=rc,
        avg_latency_ms=round(avg_latency_ms, 2),
    )


class FeedbackRequest(BaseModel):
    filename: str
    predicted_label: str
    true_label: str


@app.post("/feedback", status_code=200)
def feedback(body: FeedbackRequest):
    """
    Record the true label for a previous prediction.
    Used by the performance tracker to compute post-deployment accuracy.
    Accepts: filename, predicted_label, true_label.
    Does NOT log the image content or any sensitive data.
    """
    if body.true_label not in CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"true_label must be one of {CLASS_NAMES}",
        )

    correct = int(body.predicted_label == body.true_label)

    FEEDBACK_TOTAL.inc()
    if correct:
        FEEDBACK_CORRECT.inc()

    with _lock:
        _stats["feedback_total"]   += 1
        _stats["feedback_correct"] += correct
        accuracy = _stats["feedback_correct"] / max(_stats["feedback_total"], 1)

    MODEL_ACCURACY_GAUGE.set(accuracy)

    logger.info(json.dumps({
        "event":           "feedback",
        "filename":        body.filename,
        "predicted_label": body.predicted_label,
        "true_label":      body.true_label,
        "correct":         bool(correct),
        "running_accuracy": round(accuracy, 4),
    }))

    return {
        "accepted": True,
        "correct":  bool(correct),
        "running_accuracy": round(accuracy, 4),
    }
