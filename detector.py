"""
detector.py
-----------
ML-based building condition classifier using real uploaded images.

Usage (from your Flask route):
    from detector import predict_condition

    result = predict_condition("path/to/uploaded/image.jpg")
    # result = {
    #     "condition":  "Good" | "Warning" | "Critical",
    #     "confidence": 0.87,
    #     "color":      "green" | "yellow" | "red",
    # }
"""

import pickle
import os
import numpy as np
import cv2

MODEL_PATH   = os.path.join("models", "building_model.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
IMG_SIZE     = (64, 64)

_model, _label_encoder = None, None


def _load_model():
    global _model, _label_encoder
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Run train_model.py first."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            _label_encoder = pickle.load(f)


def _extract_features(image_path: str) -> np.ndarray:
    """Same feature extraction as in train_model.py — must stay identical."""
    with open(image_path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_bgr  = cv2.resize(img_bgr, IMG_SIZE)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)

    channel_stats = []
    for c in range(3):
        channel_stats.append(img_rgb[:, :, c].mean() / 255.0)
        channel_stats.append(img_rgb[:, :, c].std()  / 255.0)
    channel_stats = np.array(channel_stats)

    edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
    edge_density = edges.mean() / 255.0

    return np.concatenate([hist, channel_stats, [edge_density]])


_COLOR_MAP = {
    "good":     "green",
    "warning":  "yellow",
    "critical": "red",
}


def predict_condition(image_path: str) -> dict:
    """
    Predict building condition from an uploaded image.

    Returns:
        condition   — "Good" / "Warning" / "Critical"
        confidence  — 0.0 to 1.0
        color       — "green" / "yellow" / "red"
    """
    _load_model()

    features = _extract_features(image_path).reshape(1, -1)

    pred_encoded = _model.predict(features)[0]
    proba        = _model.predict_proba(features)[0]
    confidence   = float(proba[pred_encoded])

    label = _label_encoder.inverse_transform([pred_encoded])[0]  # "good"/"warning"/"critical"
    color = _COLOR_MAP.get(label, "grey")

    return {
        "condition":  label.title(),    # "Good" / "Warning" / "Critical"
        "confidence": round(confidence, 4),
        "color":      color,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detector.py <path_to_image>")
        sys.exit(1)
    result = predict_condition(sys.argv[1])
    print(f"\nImage     : {sys.argv[1]}")
    print(f"Condition : {result['condition']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Color     : {result['color']}\n")