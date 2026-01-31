"""
FastAPI service for LeafLens
- Accepts an uploaded image at /predict/ and returns top predictions
    with a simple threshold to mark unknown/no-disease cases.
"""

from fastapi import FastAPI, UploadFile, File
import os  # Needed for UNKNOWN_CONFIDENCE_THRESHOLD env lookup
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
import io
import inference

# Initialize the FastAPI application
app = FastAPI(title="Banana Disease Detector API")

# --- CORS Middleware Setup ---
# Crucial for allowing your Expo app (on a different host/port) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: Set to specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model ONCE when the API starts
MODEL = inference.load_model()

# Minimum confidence required to trust a disease classification. Any top1 below
# this will be relabeled to "Unknown" to reduce false positives on out-of-domain images.
UNKNOWN_CONFIDENCE_THRESHOLD = float(os.environ.get("UNKNOWN_THRESHOLD", "0.40"))

@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {"status": "Banana Disease Detector API is running", "model_status": "Loaded"}

# --- Static files and simple UI ---
# Serve files under /static (for CSS/JS/assets) and expose a minimal upload UI at /ui
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    # If the directory doesn't exist, skip mounting; UI route can still fallback
    pass

@app.get("/ui", response_class=HTMLResponse)
def upload_ui():
    """Serve a simple HTML upload UI to test predictions."""
    try:
        return FileResponse("static/index.html")
    except Exception:
        # Fallback minimal HTML if file missing
        return """
        <!doctype html>
        <meta charset='utf-8'/>
        <title>LeafLens UI</title>
        <p>UI file missing. POST an image to <code>/predict/</code> using curl or Postman.</p>
        <pre>
        curl -F "file=@/path/to/image.jpg" http://localhost:8000/predict/
        </pre>
        """

@app.post("/predict/")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """API endpoint to receive an image and return predictions."""
    try:
        # Read file contents and open as PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run model inference -> returns dict: {class_name: probability}
        confidences = inference.predict_image(MODEL, image)

        # Build sorted topK list and top1 object
        sorted_items = sorted(confidences.items(), key=lambda kv: kv[1], reverse=True)
        topK = [{"label": k, "probability": float(v)} for k, v in sorted_items]
        top1 = topK[0] if topK else {"label": "Unknown", "probability": 0.0}

        # Apply rejection threshold: if confidence is low, treat as a clean "no disease found".
        if top1["probability"] < UNKNOWN_CONFIDENCE_THRESHOLD:
            top1 = {"label": "No disease found", "probability": top1["probability"]}
            # Truncate topK to avoid showing misleading disease list
            topK = [top1]

        # Return a consistent shape the app expects
        return {"top1": top1, "topK": topK, "predictions": topK, "threshold": UNKNOWN_CONFIDENCE_THRESHOLD}

    except Exception as e:
        # Log the error and return a 500 status message
        print(f"Inference Error: {e}")

        return {"error": "Prediction failed due to server error."}
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
