import os
import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from threading import Thread

from config import IMG_SIZE
from Ultilities.DB_utils import load_database_features
from Scripts.Vectors.extraction import (
    extract_color_histogram,
    extract_hog,
    extract_rgb,
    extract_shape_descriptor,
    extract_texture_descriptor,
)

def extract_all_features(img_path):
    feature_funcs = [
        extract_color_histogram,
        extract_hog,
        extract_rgb,
        extract_shape_descriptor,
        extract_texture_descriptor,
    ]
    return [func(img_path) for func in tqdm(feature_funcs, desc="Extracting features")]

def search_similar(input_image_path, top_k=4):
    image_paths, color_db, hog_db, rgb_db, shape_db, texture_db = (
        load_database_features()
    )
    color_q, hog_q, rgb_q, shape_q, texture_q = extract_all_features(input_image_path)
    query_vec = np.concatenate((color_q, hog_q, rgb_q, shape_q, texture_q)).reshape(1, -1)
    db_vectors = np.concatenate([color_db, hog_db, rgb_db, shape_db, texture_db], axis=1)

    knn = NearestNeighbors(n_neighbors=len(image_paths), metric='euclidean')
    knn.fit(db_vectors)
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
    return unique_results, distances

def plot_knn_distances(distances):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(distances[0], marker='o')
    ax.set_title("KNN Distance to Query Image")
    ax.set_xlabel("Neighbor Index")
    ax.set_ylabel("Euclidean Distance")
    ax.grid(True)
    return fig

def plot_similarity_breakdown(results):
    labels = ['Color', 'HOG', 'RGB', 'Shape', 'Texture']
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (_, _, c, h, r, s, t) in enumerate(results):
        sims = [1 / (1 + c), 1 / (1 + h), 1 / (1 + r), 1 / (1 + s), 1 / (1 + t)]
        ax.bar([i + idx * 0.15 for i in range(len(sims))], sims, width=0.15, label=f"Top {idx + 1}")
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
        sim_str = f"Total:{1/(1+total):.2f}\nColor:{1/(1+c):.2f}\n Hog:{1/(1+h):.2f} RGB:{1/(1+r):.2f}\nShape:{1/(1+s):.2f}\n Texture:{1/(1+t):.2f}"
        axs[i+1].imshow(img)
        axs[i+1].set_title(sim_str)
        axs[i+1].axis("off")
    return fig

class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN Image Search Visualizer")
        self.root.geometry("1280x720")
        self.input_image_path = None
        self.results = None
        self.distances = None

        # UI Elements
        ttk.Button(root, text="Choose Image", command=self.choose_image).pack(pady=10)
        ttk.Button(root, text="Visual Match Result", command=self.show_match_images).pack(pady=10)
        ttk.Button(root, text="Plot Distance Graph", command=self.show_knn_plot).pack(pady=10)
        ttk.Button(root, text="Plot Feature Similarities", command=self.show_breakdown_plot).pack(pady=10)

        self.progress_label = ttk.Label(root, text="Idle", anchor="center")
        self.progress_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(root, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=20)

        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.input_image_path = path
            thread = Thread(target=self.process_image)
            thread.start()

    def process_image(self):
        self.update_progress("Processing...", start=True)
        self.results, self.distances = search_similar(self.input_image_path, top_k=4)
        self.update_progress("Done", stop=True)

    def update_progress(self, text, start=False, stop=False):
        self.progress_label.config(text=text)
        if start:
            self.progress_bar.start(10)
        if stop:
            self.progress_bar.stop()

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

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()
