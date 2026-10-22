"""FastAPI service for VisDrone object detection.

Run locally (from the repo root, with the package installed):
    uvicorn visdrone.serve.app:app --reload

Environment variables:
    VISDRONE_MODEL_PATH  trained weights file   (default: models/best.pt)
    VISDRONE_IMGSZ       inference image size   (default: 960; match training)
    MAX_UPLOAD_MB        max upload size in MB  (default: 20)

Endpoints:
    GET  /health   liveness check
    POST /predict  multipart image upload -> list of detections
    GET  /docs     interactive API docs
"""

import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel

MODEL_PATH = Path(os.getenv("VISDRONE_MODEL_PATH", "models/best.pt"))
IMGSZ = int(os.getenv("VISDRONE_IMGSZ", "960"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024

state: dict = {}
# A single YOLO model instance is not guaranteed to be thread-safe, and sync
# endpoints run in FastAPI's threadpool, so serialise inference calls.
predict_lock = threading.Lock()


# --------------------------------------------------------------------------
# Startup: load the model once. Fail fast if it is missing so a bad deploy
# crashes visibly instead of serving a half-working container.
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model weights not found at {MODEL_PATH.resolve()}")
    state["model"] = YOLO(str(MODEL_PATH))
    yield
    state.clear()


app = FastAPI(title="VisDrone Detection API", version="0.1.0", lifespan=lifespan)


class Detection(BaseModel):
    label: str
    confidence: float
    box_xyxy: list[float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


# --------------------------------------------------------------------------
# Inference. Deliberately a plain `def`: model.predict is blocking CPU work,
# and an `async def` would freeze the event loop (including /health) while
# it runs. FastAPI runs sync endpoints in a threadpool instead.
# --------------------------------------------------------------------------
@app.post("/predict", response_model=list[Detection])
def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    with predict_lock:
        result = state["model"].predict(image, conf=conf, imgsz=IMGSZ, verbose=False)[0]

    return [
        Detection(
            label=result.names[int(box.cls)],
            confidence=round(float(box.conf), 4),
            box_xyxy=[round(v, 1) for v in box.xyxy[0].tolist()],
        )
        for box in result.boxes
    ]
