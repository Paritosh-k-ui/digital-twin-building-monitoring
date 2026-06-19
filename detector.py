"""
detector.py
-----------
ML-based building condition classifier + image-derived structural metrics.

Usage (from your Flask route):
    from detector import predict_condition, extract_image_metrics

    result  = predict_condition("path/to/image.jpg")
    metrics = extract_image_metrics("path/to/image.jpg")
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


def _load_image(image_path: str):
    """Read image robustly (handles Unicode paths on Windows)."""
    with open(image_path, "rb") as f:
        file_bytes = np.frombuffer(f.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img_bgr


def _extract_features(image_path: str) -> np.ndarray:
    """Same feature extraction as in train_model.py — must stay identical."""
    img_bgr  = cv2.resize(_load_image(image_path), IMG_SIZE)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)

    channel_stats = []
    for c in range(3):
        channel_stats.append(img_rgb[:, :, c].mean() / 255.0)
        channel_stats.append(img_rgb[:, :, c].std()  / 255.0)
    channel_stats = np.array(channel_stats)

    edges        = cv2.Canny(img_gray, threshold1=50, threshold2=150)
    edge_density = edges.mean() / 255.0

    return np.concatenate([hist, channel_stats, [edge_density]])


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE-DERIVED STRUCTURAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def extract_image_metrics(image_path: str, ml_condition: str = None, ml_confidence: float = 1.0) -> dict:
    """
    Compute five structural/visual metrics from the uploaded image using OpenCV.
    Uses ML condition prediction to modulate values, preventing false positives
    on healthy structures (e.g., window grids, brick lines).

    Returns a dict with keys:
        crack_density       — 0-100  (higher = more cracks)
        discolouration      — 0-100  (higher = more staining / colour deviation)
        tilt                — 0-100  (higher = more structural tilt detected)
        vegetation          — 0-100  (higher = more plant growth present)
        surface_roughness   — 0-100  (higher = rougher / more degraded surface)
    Each value is a rounded integer.
    """
    # Auto-resolve ML condition if not provided
    if ml_condition is None:
        try:
            pred = predict_condition(image_path)
            ml_condition = pred["condition"]
            ml_confidence = pred["confidence"]
        except Exception:
            ml_condition = "Good"
            ml_confidence = 1.0

    img_bgr  = _load_image(image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w     = img_gray.shape

    # ── 1. Crack Density ──────────────────────────────────────────────────
    # Cracks appear as thin dark elongated edges.
    # Use Canny on a blurred image to get edge map.
    blurred   = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges     = cv2.Canny(blurred, 30, 100)
    
    # Calculate density of edge pixels directly
    edge_density = float(edges.mean()) / 255.0
    
    # Scale: typical crack density range 0.001 - 0.03
    crack_score = min(100, round(edge_density * 3000))

    # ── 2. Discolouration ─────────────────────────────────────────────────
    # Staining / damp patches deviate from a neutral grey tone.
    # Measure saturation (HSV S channel) — grey walls have low S; stains raise it.
    # Also penalise very dark regions (damp patches are darker than surroundings).
    s_channel       = img_hsv[:, :, 1].astype(float)   # 0-255
    v_channel       = img_hsv[:, :, 2].astype(float)

    mean_sat        = s_channel.mean()
    dark_fraction   = float((v_channel < 80).mean())    # fraction of very dark px
    discolour_raw   = (mean_sat / 255.0) * 0.65 + dark_fraction * 0.35
    discolour_score = min(100, round(discolour_raw * 150))

    # ── 3. Structural Tilt ────────────────────────────────────────────────
    # Detect dominant line angles via HoughLinesP.
    # Focus on roughly vertical lines and calculate their average deviation from 90°.
    edges_tilt = cv2.Canny(blurred, 50, 150)
    lines      = cv2.HoughLinesP(
        edges_tilt, rho=1, theta=np.pi/180,
        threshold=40, minLineLength=h//8, maxLineGap=10
    )

    tilt_score = 0
    if lines is not None:
        vertical_devs = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle)
            dev_from_vertical = abs(angle - 90)
            if dev_from_vertical < 30: # only consider roughly vertical lines
                vertical_devs.append(dev_from_vertical)
        
        if vertical_devs:
            mean_dev = np.mean(vertical_devs)
            # Scale deviation: 0 degrees dev -> 0 score, 15 degrees dev -> 100 score
            tilt_score = min(100, round((mean_dev / 15.0) * 100))

    # ── 4. Vegetation ─────────────────────────────────────────────────────
    # Plants/moss are green-dominated pixels.
    # Use HSV hue range for green (35–85°) with reasonable saturation.
    lower_green = np.array([35,  40,  40])
    upper_green = np.array([85, 255, 255])
    green_mask  = cv2.inRange(img_hsv, lower_green, upper_green)
    veg_frac    = float(green_mask.mean()) / 255.0
    veg_score   = min(100, round(veg_frac * 400))   # 25 % coverage → 100

    # ── 5. Surface Roughness ─────────────────────────────────────────────
    # Laplacian variance measures texture sharpness — degraded/pitted surfaces
    # have higher local variance than smooth painted walls.
    lap_var       = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    # Typical range 0–2000; clamp and scale
    roughness_score = min(100, round((lap_var / 800.0) * 100))

    # ── Prior-Guided Heuristic Modulation ────────────────────────────────
    if ml_condition == "Good":
        crack_score     = min(15, round(crack_score * (1.0 - ml_confidence * 0.9)))
        discolour_score = min(20, round(discolour_score * (1.0 - ml_confidence * 0.8)))
        tilt_score      = min(20, round(tilt_score * (1.0 - ml_confidence * 0.8)))
        veg_score       = min(15, round(veg_score * (1.0 - ml_confidence * 0.9)))
        roughness_score = min(25, round(roughness_score * (1.0 - ml_confidence * 0.75)))
    elif ml_condition == "Warning":
        crack_score     = max(15, min(50, round(crack_score * 0.7 + 10)))
        discolour_score = max(15, min(50, round(discolour_score * 0.7 + 10)))
        tilt_score      = max(10, min(50, round(tilt_score * 0.7 + 10)))
        veg_score       = max(5,  min(40, round(veg_score * 0.7 + 5)))
        roughness_score = max(20, min(60, round(roughness_score * 0.7 + 15)))
    else: # Critical
        crack_score     = max(50, min(100, round(crack_score * 0.8 + 30)))
        discolour_score = max(40, min(100, round(discolour_score * 0.8 + 25)))
        tilt_score      = max(40, min(100, round(tilt_score * 0.8 + 25)))
        veg_score       = max(25, min(100, round(veg_score * 0.8 + 15)))
        roughness_score = max(50, min(100, round(roughness_score * 0.8 + 30)))

    return {
        "crack_density":     int(crack_score),
        "discolouration":    int(discolour_score),
        "tilt":              int(tilt_score),
        "vegetation":        int(veg_score),
        "surface_roughness": int(roughness_score),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED WEIGHTED RISK SCORE
# ─────────────────────────────────────────────────────────────────────────────

# Weights for final combined score (environmental metrics removed)
_ML_WEIGHT = 0.60       # ML condition prediction
_CV_WEIGHT = 0.40       # image-derived CV metrics (average)

# Per-metric weights within the CV block
_CV_METRIC_WEIGHTS = {
    "crack_density":     0.35,
    "discolouration":    0.20,
    "tilt":              0.25,
    "vegetation":        0.10,
    "surface_roughness": 0.10,
}


def compute_combined_score(
    ml_condition: str,
    ml_confidence: float,
    image_metrics: dict,
) -> dict:
    """
    Combine ML prediction and image-derived CV metrics
    into a single weighted risk score (0-100) and final condition label.

    Args:
        ml_condition   : "Good" / "Warning" / "Critical"
        ml_confidence  : 0.0 – 1.0
        image_metrics  : dict from extract_image_metrics()

    Returns:
        combined_score  : int 0-100
        final_condition : "Good" / "Warning" / "Critical"
        ml_score        : int  (ML contribution as 0-100)
        cv_score        : int  (CV metrics weighted average)
        env_score       : int  (0 for compatibility)
    """
    # ML score: base risk per label, modulated by confidence
    base = {"Good": 10, "Warning": 50, "Critical": 90}.get(ml_condition, 50)
    ml_score = round(base * ml_confidence + (100 - base) * (1 - ml_confidence) * 0.3)

    # CV score: weighted average of image metrics
    cv_score = round(sum(
        image_metrics.get(k, 0) * w
        for k, w in _CV_METRIC_WEIGHTS.items()
    ))

    # Combined (60% ML + 40% CV)
    combined = (
        ml_score  * _ML_WEIGHT +
        cv_score  * _CV_WEIGHT
    )
    combined_score = min(100, round(combined))

    # Derive condition from combined score
    if combined_score >= 65:
        final_condition = "Critical"
    elif combined_score >= 35:
        final_condition = "Warning"
    else:
        final_condition = "Good"

    return {
        "combined_score":    combined_score,
        "final_condition":   final_condition,
        "ml_score":          ml_score,
        "cv_score":          cv_score,
        "env_score":         0,
    }


# ─────────────────────────────────────────────────────────────────────────────
_COLOR_MAP = {"good": "green", "warning": "yellow", "critical": "red"}


def predict_condition(image_path: str) -> dict:
    """
    Predict building condition from an uploaded image.

    Returns:
        condition   — "Good" / "Warning" / "Critical"
        confidence  — 0.0 to 1.0
        color       — "green" / "yellow" / "red"
    """
    _load_model()

    features     = _extract_features(image_path).reshape(1, -1)
    pred_encoded = _model.predict(features)[0]
    proba        = _model.predict_proba(features)[0]
    confidence   = float(proba[pred_encoded])
    label        = _label_encoder.inverse_transform([pred_encoded])[0]
    color        = _COLOR_MAP.get(label, "grey")

    return {
        "condition":  label.title(),
        "confidence": round(confidence, 4),
        "color":      color,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python detector.py <path_to_image>")
        sys.exit(1)

    path    = sys.argv[1]
    result  = predict_condition(path)
    metrics = extract_image_metrics(path)
    scores  = compute_combined_score(result["condition"], result["confidence"], metrics)

    print(f"\nImage     : {path}")
    print(f"Condition : {result['condition']}  (confidence {result['confidence']*100:.1f}%)")
    print(f"\nImage Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}/100")
    print(f"\nCombined Score : {scores['combined_score']}/100  → {scores['final_condition']}\n")