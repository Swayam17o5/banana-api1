"""
Model loading and inference helpers
- Downloads model if missing
- Loads HybridCNNViT weights
- Performs image inference
"""

import os
import urllib.request
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import HybridCNNViT

# ---------------- Configuration ----------------
DEVICE = torch.device("cpu")
NUM_CLASSES = 9

CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "best_model.pth")
CHECKPOINT_URL = os.environ.get("MODEL_URL")

CLASS_NAMES = [
    'Anthracnose',
    'Banana Fruit-Scarring Beetle',
    'Banana Skipper Damage',
    'Banana Split Peel',
    'Black and Yellow Sigatoka',
    'Chewing insect damage on banana leaf',
    'Healthy Banana',
    'Healthy Banana leaf',
    'Panama Wilt Disease'
]

# ---------------- Transforms ----------------
base_inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---------------- Utilities ----------------
def _ensure_checkpoint_available():
    """
    Ensure the model checkpoint exists locally.
    If not, download it from MODEL_URL.
    """
    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH

    if not CHECKPOINT_URL:
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH} and MODEL_URL is not set."
        )

    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".", exist_ok=True)
    print(f"Downloading model from {CHECKPOINT_URL}...")

    try:
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as resp, open(CHECKPOINT_PATH, "wb") as out:
            chunk_size = 1024 * 1024  # 1 MB
            downloaded = 0
            total = int(resp.headers.get("Content-Length", 0)) or None

            while True:
                data = resp.read(chunk_size)
                if not data:
                    break
                out.write(data)
                downloaded += len(data)

                if total:
                    pct = (downloaded / total) * 100
                    print(f"Downloaded {downloaded / 1_048_576:.1f} MB ({pct:.1f}%)", end="\r")

        print("\nModel download complete.")
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

    return CHECKPOINT_PATH

# ---------------- Model Loading ----------------
def load_model():
    """
    Instantiate the model and load weights.
    """
    ckpt_path = _ensure_checkpoint_available()
    print(f"Loading model from {ckpt_path} on {DEVICE}...")

    model = HybridCNNViT(NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    # Support both checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully.")
    return model

# ---------------- Inference ----------------
def predict_image(model, pil_image: Image.Image, top_k: int = 3):
    """
    Perform inference on a PIL image.
    Returns top-1 and top-k predictions.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    input_tensor = base_inference_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits[0], dim=0)

    # Sort predictions
    sorted_indices = torch.argsort(probs, descending=True)

    top1 = {
        "label": CLASS_NAMES[sorted_indices[0]],
        "probability": probs[sorted_indices[0]].item()
    }

    topk = [
        {
            "label": CLASS_NAMES[i],
            "probability": probs[i].item()
        }
        for i in sorted_indices[:top_k]
    ]

    return {
        "top1": top1,
        "topK": topk
    }
