import os
import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
import joblib

from Scripts.config import CACHE_DIR
from Scripts.Vectors.extraction import (
    extract_color_histogram,
    extract_hog,
    extract_rgb,
    extract_shape_descriptor,
    extract_texture_descriptor,
)

# Load cached KNN model & feature vectors
knn = joblib.load(f"{CACHE_DIR}/knn_model.pkl")
vectors = np.load(f"{CACHE_DIR}/vectors.npy")
image_paths = joblib.load(f"{CACHE_DIR}/image_paths.pkl")


def extract_all_features(img_path, update_callback=None):
    feature_funcs = [
        extract_color_histogram,
        extract_hog,
        extract_rgb,
        extract_shape_descriptor,
        extract_texture_descriptor,
    ]
    features = [None] * len(feature_funcs)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(func, img_path): i for i, func in enumerate(feature_funcs)
        }
        for completed, f in enumerate(as_completed(futures), 1):
            idx = futures[f]
            features[idx] = f.result()
            if update_callback:
                update_callback(completed, len(feature_funcs))

    return features


def search_similar(input_image_path, update_callback=None, top_k=3):
    features = extract_all_features(input_image_path, update_callback)
    color_q, hog_q, rgb_q, shape_q, texture_q = features
    query_vec = np.concatenate(features).reshape(1, -1)
    distances, indices = knn.kneighbors(query_vec)

    seen = set()
    unique_results = []

    # Get lengths for slicing
    len_color = len(color_q)
    len_hog = len(hog_q)
    len_rgb = len(rgb_q)
    len_shape = len(shape_q)

    for dist, idx in zip(distances[0], indices[0]):
        img_path = image_paths[idx]
        img_name = os.path.basename(img_path)
        if img_name in seen:
            continue
        seen.add(img_name)

        vec = vectors[idx]
        color_db = vec[:len_color]
        hog_db = vec[len_color : len_color + len_hog]
        rgb_db = vec[len_color + len_hog : len_color + len_hog + len_rgb]
        shape_db = vec[
            len_color + len_hog + len_rgb : len_color + len_hog + len_rgb + len_shape
        ]
        texture_db = vec[len_color + len_hog + len_rgb + len_shape :]

        d_color = np.linalg.norm(color_q - color_db)
        d_hog = np.linalg.norm(hog_q - hog_db)
        d_rgb = np.linalg.norm(rgb_q - rgb_db)
        d_shape = np.linalg.norm(shape_q - shape_db)
        d_texture = np.linalg.norm(texture_q - texture_db)

        unique_results.append(
            (img_path, dist, d_color, d_hog, d_rgb, d_shape, d_texture)
        )

        if len(unique_results) == top_k:
            break

    return unique_results, distances


def plot_knn_distances(distances):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(distances[0], marker="o")
    ax.set_title("KNN Distance to Query Image")
    ax.set_xlabel("Neighbor Index")
    ax.set_ylabel("Euclidean Distance")
    ax.grid(True)
    return fig


def plot_similarity_breakdown(results):
    labels = ["Color", "HOG", "RGB", "Shape", "Texture"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (_, _, c, h, r, s, t) in enumerate(results):
        sims = [1 / (1 + c), 1 / (1 + h), 1 / (1 + r), 1 / (1 + s), 1 / (1 + t)]
        ax.bar(
            [i + idx * 0.15 for i in range(len(sims))],
            sims,
            width=0.15,
            label=f"Top {idx + 1}",
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Similarity")
    ax.set_title("Similarity Breakdown")
    ax.legend()
    return fig


def plot_visual_results(input_image_path, results):
    fig, axs = plt.subplots(1, len(results) + 1, figsize=(20, 6))
    input_img = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
    axs[0].imshow(input_img)
    axs[0].set_title("Input")
    axs[0].axis("off")
    for i, (path, total, c, h, r, s, t) in enumerate(results):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        sim_str = f"Total:{1/(1+total):.2f}\nHSV:{1/(1+c):.2f}\nHOG:{1/(1+h):.2f}\nRGB:{1/(1+r):.2f}\nShape:{1/(1+s):.2f}\nTexture:{1/(1+t):.2f}"
        axs[i + 1].imshow(img)
        axs[i + 1].set_title(sim_str)
        axs[i + 1].axis("off")
    return fig


class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN Image Search Visualizer")
        self.root.geometry("1280x720")
        self.input_image_path = None
        self.results = None
        self.distances = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # UI
        ttk.Button(root, text="Choose Image", command=self.choose_image).pack(pady=10)
        ttk.Button(
            root, text="Visual Match Result", command=self.show_match_images
        ).pack(pady=10)
        ttk.Button(root, text="Plot Distance Graph", command=self.show_knn_plot).pack(
            pady=10
        )
        ttk.Button(
            root, text="Plot Feature Similarities", command=self.show_breakdown_plot
        ).pack(pady=10)

        self.progress_label = ttk.Label(root, text="Idle", anchor="center")
        self.progress_label.pack(pady=5)

        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.input_image_path = path
            thread = Thread(target=self.process_image)
            thread.start()

    def update_feature_progress(self, completed, total):
        self.root.after(
            0,
            lambda: self.progress_label.config(
                text=f"Extracted {completed}/{total} features"
            ),
        )

    def process_image(self):
        self.root.after(0, lambda: self.progress_label.config(text="Processing..."))
        self.results, self.distances = search_similar(
            self.input_image_path, update_callback=self.update_feature_progress
        )
        self.root.after(0, lambda: self.progress_label.config(text="Done"))

    def clear_canvas(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

    def show_figure(self, fig):
        self.clear_canvas()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_knn_plot(self):
        if self.distances is not None:
            fig = plot_knn_distances(self.distances)
            self.show_figure(fig)

    def show_breakdown_plot(self):
        if self.results is not None:
            fig = plot_similarity_breakdown(self.results)
            self.show_figure(fig)

    def show_match_images(self):
        if self.results is not None:
            fig = plot_visual_results(self.input_image_path, self.results)
            self.show_figure(fig)

    def on_close(self):
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()
