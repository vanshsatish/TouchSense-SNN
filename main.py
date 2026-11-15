from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .features import extract_features, normalize_feature_vector
from .schemas import AnalyzeRequest, AnalyzeResponse
from .snn_model import load_checkpoint

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "touch_snn.pt"

app = FastAPI(
    title="TouchSense-SNN API",
    version="0.1.0",
    description="Research API for touchscreen gesture feature extraction and SNN experiments."
)

# Restrict origins in production. This permissive setting is for local development only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    _, status = load_checkpoint(MODEL_PATH)
    return {"status": "ok", "model_status": status}

@app.get("/model")
def model_info():
    _, status = load_checkpoint(MODEL_PATH)
    return {
        "architecture": "Leaky Integrate-and-Fire SNN",
        "model_status": status,
        "checkpoint_present": MODEL_PATH.exists(),
        "classes": ["pattern_a", "pattern_b", "pattern_c"],
        "warning": "Class names are placeholders until a validated study defines labels."
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        points = [p.model_dump() for p in request.points]
        features = extract_features(points, request.tap_interval_ms)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    model, status = load_checkpoint(MODEL_PATH)
    prediction = None

    if model is not None:
        vector = normalize_feature_vector(features)
        # Turn continuous features into a short temporal event sequence.
        # This is a simple rate-code baseline; replace/compare with latency coding.
        steps = 20
        seq = np.repeat(vector[None, :], steps, axis=0)
        seq = (seq >= np.linspace(0.15, 0.85, steps)[:, None]).astype(np.float32)
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            spikes, membrane = model(tensor)
            scores = membrane[0].cpu().numpy()
            probs = torch.softmax(torch.tensor(scores), dim=0).numpy()
        prediction = {
            "class_index": int(np.argmax(probs)),
            "scores": [float(v) for v in probs],
            "label": ["pattern_a", "pattern_b", "pattern_c"][int(np.argmax(probs))],
            "interpretation": "Experimental model output only; not a diagnosis."
        }

    return AnalyzeResponse(
        features=features,
        model_status=status,
        prediction=prediction,
        notice=(
            "Research measurement only. Not validated for diagnosis or screening. "
            "Pressure may be unavailable or not comparable across devices."
        )
    )
