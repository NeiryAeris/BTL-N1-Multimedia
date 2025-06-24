import os
import json
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from config import DB_PATH

def load_features_from_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT image_path, color_histogram, hog_vector, rgb_vector, shape_descriptor, texture_descriptor FROM image_features"
    )
    rows = cursor.fetchall()
    conn.close()

    image_paths = []
    feature_vectors = []

    for row in rows:
        image_paths.append(row[0])
        features = (
            json.loads(row[1]) + json.loads(row[2]) +
            json.loads(row[3]) + json.loads(row[4]) +
            json.loads(row[5])
        )
        feature_vectors.append(features)

    return image_paths, np.array(feature_vectors, dtype="float32")

def evaluate_clustering_with_pca(k_range=range(2, 15), pca_components=100):
    _, features = load_features_from_db()
    pca = PCA(n_components=pca_components)
    reduced = pca.fit_transform(features)

    scores = []
    models = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(reduced)
        score = silhouette_score(reduced, labels)
        scores.append(score)
        models.append((k, labels))
        print(f"k={k}: silhouette_score={score:.4f}")
    return k_range, scores, reduced, models

def plot_scores(k_values, scores):
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, scores, marker='o')
    plt.title("Silhouette Score for Different k (PCA)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_clusters_2D(reduced, labels, title="2D PCA Cluster Visualization"):
    pca_2d = PCA(n_components=2)
    reduced_2d = pca_2d.fit_transform(reduced)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, ticks=range(len(set(labels))))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    k_values, silhouette_scores, reduced_data, model_outputs = evaluate_clustering_with_pca()

    plot_scores(k_values, silhouette_scores)

    # Plot the best k
    best_k_index = int(np.argmax(silhouette_scores))
    best_k, best_labels = model_outputs[best_k_index]
    plot_clusters_2D(reduced_data, best_labels, title=f"Best k={best_k} Clusters (PCA-Reduced)")
