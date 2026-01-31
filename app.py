"""
FastAPI service for LeafLens
- Lazy-loads model AFTER server startup (Cloud Run safe)
"""

from fastapi import FastAPI, UploadFile, File
import os
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import inference
import uvicorn

app = FastAPI(title="Banana Disease Detector API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Globals ----------------
MODEL = None  # 👈 IMPORTANT
UNKNOWN_CONFIDENCE_THRESHOLD = float(os.environ.get("UNKNOWN_THRESHOLD", "0.40"))

# ---------------- Startup ----------------
@app.on_event("startup")
def load_model_on_startup():
    global MODEL
    print("🚀 Server started, loading model...")
    MODEL = inference.load_model()
    print("✅ Model loaded successfully")

# ---------------- Health ----------------
@app.get("/")
def read_root():
    return {
        "status": "Banana Disease Detector API is running",
        "model_loaded": MODEL is not None
    }

# ---------------- Prediction ----------------
@app.post("/predict/")
async def predict_image_endpoint(file: UploadFile = File(...)):
    if MODEL is None:
        return {"error": "Model not loaded yet"}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    confidences = inference.predict_image(MODEL, image)

    sorted_items = sorted(confidences.items(), key=lambda kv: kv[1], reverse=True)
    topK = [{"label": k, "probability": float(v)} for k, v in sorted_items]
    top1 = topK[0]

    if top1["probability"] < UNKNOWN_CONFIDENCE_THRESHOLD:
        top1 = {"label": "No disease found", "probability": top1["probability"]}
        topK = [top1]

    return {
        "top1": top1,
        "topK": topK,
        "threshold": UNKNOWN_CONFIDENCE_THRESHOLD
    }

# ---------------- Local run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
