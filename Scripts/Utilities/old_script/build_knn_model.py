# build_knn_model.py
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import joblib
import numpy as np
from DB_utils import load_database_features
from sklearn.neighbors import NearestNeighbors
from Scripts.config import CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)

def build_and_save_knn():
    image_paths, color, hog, rgb, shape, texture = load_database_features()
    vectors = np.concatenate([color, hog, rgb, shape, texture], axis=1)

    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(vectors)

    joblib.dump(knn, f"{CACHE_DIR}/knn_model.pkl")
    np.save(f"{CACHE_DIR}/vectors.npy", vectors)
    joblib.dump(image_paths, f"{CACHE_DIR}/image_paths.pkl")

    print(f"✅ KNN model rebuilt with {len(image_paths)} images.")

if __name__ == "__main__":
    build_and_save_knn()
