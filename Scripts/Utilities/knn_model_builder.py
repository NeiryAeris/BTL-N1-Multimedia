import os
import hashlib
import numpy as np
import joblib

from Scripts.config import DB_PATH, CACHE_DIR
from Scripts.Utilities.DB_utils import load_database_features
from sklearn.neighbors import NearestNeighbors

HASH_FILE = os.path.join(CACHE_DIR, "db_hash.txt")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_previous_hash():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return f.read().strip()
    return ""

def save_hash(hash_value):
    with open(HASH_FILE, "w") as f:
        f.write(hash_value)

def build_knn_model():
    print("📦 Loading features from DB to build model...")
    image_paths, color, hog, rgb, shape, texture = load_database_features()
    vectors = np.concatenate([color, hog, rgb, shape, texture], axis=1)

    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(vectors)

    joblib.dump(knn, f"{CACHE_DIR}/knn_model.pkl")
    np.save(f"{CACHE_DIR}/vectors.npy", vectors)
    joblib.dump(image_paths, f"{CACHE_DIR}/image_paths.pkl")

    print(f"✅ Model built with {len(image_paths)} entries.")

def knn_model_controller():
    print("🔍 Checking if database has changed...")
    current_hash = get_file_hash(DB_PATH)
    previous_hash = load_previous_hash()

    if current_hash != previous_hash:
        print("🧠 Change detected in database. Rebuilding KNN model...")
        build_knn_model()
        save_hash(current_hash)
    else:
        print("✅ No changes detected. KNN model is up to date.")