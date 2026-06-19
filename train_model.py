"""
train_model.py
--------------
Trains a Random Forest classifier on building condition images.

Dataset folder structure required:
    dataset/
    ├── good/        (images of healthy, well-maintained buildings)
    ├── warning/      (images of buildings with visible minor damage/cracks)
    └── critical/     (images of severely damaged/structurally unsafe buildings)

Run:
    python train_model.py

Output:
    models/building_model.pkl
    models/label_encoder.pkl
"""

import os
import pickle
import numpy as np
from pathlib import Path

import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
MODEL_DIR    = "models"
IMG_SIZE     = (64, 64)
N_ESTIMATORS = 100
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────────────────────────────


def extract_features(image_path: str):
    """
    Read an image and return a flat feature vector (263 dims).
    Same pipeline as the road damage project:
        1. Grayscale histogram          - 256 values
        2. RGB channel mean + std       - 6 values
        3. Canny edge density           - 1 value
    """
    with open(image_path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        print(f"  [WARN] Could not read: {image_path}")
        return None

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


def load_dataset(dataset_dir: str):
    X, y = [], []
    dataset_path = Path(dataset_dir)

    class_names = sorted([
        d.name for d in dataset_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not class_names:
        raise FileNotFoundError(
            f"No class subfolders found in '{dataset_dir}'. "
            "Make sure you have: dataset/good/, dataset/warning/, dataset/critical/"
        )

    print(f"\nClasses found: {class_names}\n")

    for class_name in class_names:
        class_path = dataset_path / class_name
        image_files = [
            f for f in class_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
        ]
        print(f"  [{class_name}] - {len(image_files)} images")

        for img_file in image_files:
            features = extract_features(str(img_file))
            if features is not None:
                X.append(features)
                y.append(class_name)

    return np.array(X), np.array(y), class_names


def train(dataset_dir: str = DATASET_DIR, model_dir: str = MODEL_DIR):
    print("=" * 55)
    print("  Building Condition ML Classifier - Training")
    print("=" * 55)

    print("\n[1/4] Loading dataset ...")
    X, y, class_names = load_dataset(dataset_dir)
    print(f"\n  Total samples: {len(X)}")

    if len(X) < 10:
        raise ValueError("Too few images (found < 10). Add more images to your dataset folders.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\n[2/4] Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"\n[3/4] Split -> Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"\n[4/4] Training Random Forest (n_estimators={N_ESTIMATORS}) ...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(f"  Accuracy: {acc * 100:.1f}%")
    print("=" * 55)
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(model_dir, exist_ok=True)
    model_path   = os.path.join(model_dir, "building_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    print(f"\n[OK] Model saved   ->  {model_path}")
    print(f"[OK] Encoder saved ->  {encoder_path}")
    print("\nTraining complete. You can now run your Flask app.\n")

    return clf, le


if __name__ == "__main__":
    train()