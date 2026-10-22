"""
FastAPI app deployment

Package requirements:
    fastapi 
    "uvicorn[standard]" 
    python-multipart 
    pillow 
    ultralytics

Run:
    uvicorn main:app --reload

Endpoint:
http://127.0.0.1:8000

Docs
http://127.0.0.1:8000/docs for the interactive API docs.
"""

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Startup: load the model once, not on every request
# --------------------------------------------------------------------------
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from ultralytics import YOLO

        state["model"] = YOLO("yolo11n.pt")  # swap for your own best.pt
    except Exception as exc:  # keep the CRUD endpoints usable without a model
        print(f"Model not loaded: {exc}")
        state["model"] = None
    yield
    state.clear()


app = FastAPI(title="Example API", version="0.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class ItemIn(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    price: float = Field(gt=0)
    in_stock: bool = True


class Item(ItemIn):
    id: int


class Detection(BaseModel):
    label: str
    confidence: float
    box_xyxy: list[float]


# --------------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.get("model") is not None}


# --------------------------------------------------------------------------
# Simple in-memory CRUD
# --------------------------------------------------------------------------
items: dict[int, Item] = {}
next_id = 1


@app.post("/items", response_model=Item, status_code=201)
def create_item(payload: ItemIn):
    global next_id
    item = Item(id=next_id, **payload.model_dump())
    items[next_id] = item
    next_id += 1
    return item


@app.get("/items", response_model=list[Item])
def list_items(in_stock: bool | None = None, limit: int = 100):
    results = list(items.values())
    if in_stock is not None:
        results = [i for i in results if i.in_stock == in_stock]
    return results[:limit]


@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]


@app.put("/items/{item_id}", response_model=Item)
def update_item(item_id: int, payload: ItemIn):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    items[item_id] = Item(id=item_id, **payload.model_dump())
    return items[item_id]


@app.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int):
    if items.pop(item_id, None) is None:
        raise HTTPException(status_code=404, detail="Item not found")


# --------------------------------------------------------------------------
# Model inference: upload an image, get detections back
# --------------------------------------------------------------------------
@app.post("/predict", response_model=list[Detection])
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    result = model.predict(image, conf=conf, verbose=False)[0]
    return [
        Detection(
            label=result.names[int(box.cls)],
            confidence=round(float(box.conf), 4),
            box_xyxy=[round(v, 1) for v in box.xyxy[0].tolist()],
        )
        for box in result.boxes
    ]