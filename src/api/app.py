import os
import time
import logging
import io
from typing import Dict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.model import SimpleCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

APP_NAME = "cats-dogs-inference"
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pt")
DEVICE = os.getenv("DEVICE", "cpu")

CLASS_NAMES = ["cats", "dogs"]  # must match ImageFolder class names used in preprocessing

app = FastAPI(title=APP_NAME)

# simple in-app counters (basic monitoring)
REQUEST_COUNT = 0
TOTAL_LATENCY_SEC = 0.0

def preprocess_pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.tensor(arr).unsqueeze(0)  # 1x3x224x224
    return x

def load_model() -> SimpleCNN:
    model = SimpleCNN(num_classes=2)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first and place weights there.")
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd)
    model.to(DEVICE)
    model.eval()
    return model

MODEL = None

@app.on_event("startup")
def startup_event():
    global MODEL
    MODEL = load_model()
    logging.info("Model loaded from %s on %s", MODEL_PATH, DEVICE)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": APP_NAME}

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    request_count: int
    avg_latency_ms: float

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    global REQUEST_COUNT, TOTAL_LATENCY_SEC
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    t0 = time.time()
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    x = preprocess_pil_to_tensor(img).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

    latency = time.time() - t0
    REQUEST_COUNT += 1
    TOTAL_LATENCY_SEC += latency

    prob_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    avg_latency_ms = (TOTAL_LATENCY_SEC / max(REQUEST_COUNT, 1)) * 1000.0

    logging.info("predict request=%d label=%s confidence=%.4f latency_ms=%.2f",
                 REQUEST_COUNT, label, confidence, latency * 1000.0)

    return PredictResponse(
        label=label,
        confidence=confidence,
        probabilities=prob_map,
        request_count=REQUEST_COUNT,
        avg_latency_ms=float(avg_latency_ms),
    )
