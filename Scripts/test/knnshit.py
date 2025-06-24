import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor
from Ultilities.DB_utils import load_database_features
from config import IMG_SIZE
from Scripts.Vectors.extraction import (
    extract_color_histogram,
    extract_hog,
    extract_rgb,
    extract_shape_descriptor,
    extract_texture_descriptor,
)


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def build_knn_index(db_features, n_neighbors=4):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
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
    # Sequential with progress bar
    feature_funcs = [
        extract_color_histogram,
        extract_hog,
        extract_rgb,
        extract_shape_descriptor,
        extract_texture_descriptor,
    ]

    results = []
    for func in tqdm(feature_funcs, desc="Extracting features"):
        results.append(func(img_path))
    return results


def plot_knn_distances(distances):
    plt.figure(figsize=(8, 4))
    plt.plot(distances[0], marker="o")
    plt.title("KNN Distance to Query Image")
    plt.xlabel("Neighbor Index")
    plt.ylabel("Euclidean Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_similarity_breakdown(results):
    labels = ["Color", "HOG", "RGB", "Shape", "Texture"]
    plt.figure(figsize=(12, 5))
    for idx, (_, _, c, h, r, s, t) in enumerate(results):
        sims = [1 / (1 + c), 1 / (1 + h), 1 / (1 + r), 1 / (1 + s), 1 / (1 + t)]
        plt.bar(
            [i + idx * 0.15 for i in range(len(sims))],
            sims,
            width=0.15,
            label=f"Top {idx + 1}",
        )
    plt.xticks(range(len(labels)), labels)
    plt.title("Similarity Breakdown per Feature")
    plt.ylabel("Similarity (1 / (1 + distance))")
    plt.legend()
    plt.tight_layout()
    plt.show()


def search_similar(input_image_path, top_k=4):
    image_paths, color_db, hog_db, rgb_db, shape_db, texture_db = (
        load_database_features()
    )
    color_q, hog_q, rgb_q, shape_q, texture_q = extract_all_features_parallel(
        input_image_path
    )
    query_vec = np.concatenate((color_q, hog_q, rgb_q, shape_q, texture_q)).reshape(
        1, -1
    )

    db_vectors = np.concatenate(
        [color_db, hog_db, rgb_db, shape_db, texture_db], axis=1
    )

    knn = build_knn_index(db_vectors, n_neighbors=len(image_paths))
    distances, indices = knn.kneighbors(query_vec)

    print("\n--- Feature Vector Lengths ---")
    print(
        f"Color: {len(color_q)} | HOG: {len(hog_q)} | RGB: {len(rgb_q)} | Shape: {len(shape_q)} | Texture: {len(texture_q)}"
    )
    print(f"Combined: {query_vec.shape}")

    plot_knn_distances(distances)

    seen = set()
    unique_results = []

    for dist, idx in zip(distances[0], indices[0]):
        img_path = image_paths[idx]
        img_name = os.path.basename(img_path)

        if img_name not in seen:
            seen.add(img_name)
            unique_results.append(
                (
                    img_path,
                    dist,
                    np.linalg.norm(color_q - color_db[idx]),
                    np.linalg.norm(hog_q - hog_db[idx]),
                    np.linalg.norm(rgb_q - rgb_db[idx]),
                    np.linalg.norm(shape_q - shape_db[idx]),
                    np.linalg.norm(texture_q - texture_db[idx]),
                )
            )

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

    # Also show breakdown
    plot_similarity_breakdown(results)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(current_dir, "sample_query2.jpg")
    results = search_similar(test_image, top_k=4)
    visualize_results(test_image, results)
