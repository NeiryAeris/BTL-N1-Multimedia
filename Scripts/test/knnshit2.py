import cv2
import os
import json
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from Ultilities.DB_utils import load_database_features
from config import IMG_SIZE
from Scripts.Vectors.extraction import (
    extract_color_histogram,
    extract_rgb,
    extract_shape_descriptor,
    extract_texture_descriptor,
    extract_hog
)


def build_knn_index(db_features, n_neighbors=4):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(db_features)
    return knn


def resize_image_to_256(img_path, output_path=None):
    image = cv2.imread(img_path)
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    if output_path is None:
        output_path = img_path
    cv2.imwrite(output_path, resized)
    return output_path


def preprocess_and_resize(img_path):
    image = cv2.imread(img_path)
    return cv2.resize(image, IMG_SIZE)


def extract_all_features_parallel(img_path):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(func, img_path)
            for func in [
                extract_color_histogram,
                extract_hog,
                extract_rgb,
                extract_shape_descriptor,
                extract_texture_descriptor,
            ]
        ]
        results = [f.result() for f in futures]
    return results


def search_similar(input_image_path, top_k=4):
    image_paths, color_db, hog_db, rgb_db, shape_db, texture_db = load_database_features()

    color_q, hog_q, rgb_q, shape_q, texture_q = extract_all_features_parallel(input_image_path)
    query_vec = np.concatenate((color_q, hog_q, rgb_q, shape_q, texture_q)).reshape(1, -1)

    db_vectors = np.concatenate(
        [color_db, hog_db, rgb_db, shape_db, texture_db], axis=1
    )

    knn = build_knn_index(db_vectors, n_neighbors=len(image_paths))
    distances, indices = knn.kneighbors(query_vec)

    seen = set()
    unique_results = []

    for dist, idx in zip(distances[0], indices[0]):
        img_path = image_paths[idx]
        img_name = os.path.basename(img_path)

        if img_name not in seen:
            seen.add(img_name)
            unique_results.append((
                img_path,
                dist,
                np.linalg.norm(color_q - color_db[idx]),
                np.linalg.norm(hog_q - hog_db[idx]),
                np.linalg.norm(rgb_q - rgb_db[idx]),
                np.linalg.norm(shape_q - shape_db[idx]),
                np.linalg.norm(texture_q - texture_db[idx]),
            ))

        if len(unique_results) == top_k:
            break

    return unique_results


def visualize_results(input_image_path, results):
    plt.figure(figsize=(20, 6))
    input_image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
    plt.subplot(1, len(results) + 1, 1)
    plt.imshow(input_image)
    plt.title("Input")
    plt.axis("off")

    for idx, (path, total, c, h, r, s, t) in enumerate(results):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        sim_str = (
            f"Total:{1/(1+total):.2f}\n"
            f"Color:{1/(1+c):.2f} HOG:{1/(1+h):.2f} RGB:{1/(1+r):.2f}\n"
            f"Shape:{1/(1+s):.2f} Texture:{1/(1+t):.2f}"
        )
        plt.subplot(1, len(results) + 1, idx + 2)
        plt.imshow(img)
        plt.title(sim_str)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(current_dir, "sample_query.jpg")
    results = search_similar(test_image, top_k=4)
    visualize_results(test_image, results)
