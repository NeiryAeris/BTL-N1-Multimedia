import gc
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import numpy as np
from tqdm import tqdm
import concurrent.futures
from Scripts.Utilities.DB_utils import init_db, insert_features, save_sample
from Scripts.config import ROOT_DIR, BATCH_SIZE
# from extractions import (
#     extract_color_histogram,
#     extract_hog,
#     extract_rgb,
#     extract_shape_descriptor,
#     extract_hu,
#     extract_texture_descriptor,
# )

from Scripts.Vectors.extraction import (
    extract_color_histogram,
    extract_rgb,
    extract_shape_descriptor,
    extract_texture_descriptor,
    extract_hog
)
    


def extract_all_features(img_path):
    return {
        "image_path": img_path,
        "color_histogram": extract_color_histogram(img_path),
        "hog_vector": extract_hog(img_path),
        "rgb_vector": extract_rgb(img_path),
        "shape_descriptor": extract_shape_descriptor(img_path),
        "texture_descriptor": extract_texture_descriptor(img_path),
    }


def process_and_store_batch(image_batch, label_batch, cursor):
    insert_data = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(extract_all_features, img_path) for img_path in image_batch
        ]
        # results = [f.result() for f in concurrent.futures.as_completed(futures)]
        results = []
        for f in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Extracting features",
        ):
            results.append(f.result())

    for idx, features in enumerate(results):
        full_feature_vector = np.array(
            features["color_histogram"]
            + features["shape_descriptor"]
            + features["texture_descriptor"]
        )
        normalized_vector = full_feature_vector / (
            np.linalg.norm(full_feature_vector) + 1e-8
        )

        insert_data.append(
            (
                features["image_path"],
                label_batch[idx],
                json.dumps(features["color_histogram"]),
                json.dumps(features["hog_vector"]),
                json.dumps(features["rgb_vector"]),
                json.dumps(features["shape_descriptor"]),
                json.dumps(features["texture_descriptor"]),
            )
        )

    insert_features(cursor, insert_data)
    gc.collect()

def extract_features():
    conn, c = init_db()

    image_batch, label_batch = [], []

    class_dirs = [
        d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))
    ]

    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join(ROOT_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue

        for image_file in tqdm(
            os.listdir(class_path), desc=f"{class_dir}", leave=False
        ):
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(class_path, image_file)
            image_batch.append(image_path)
            label_batch.append(class_dir)

            if len(image_batch) == BATCH_SIZE:
                process_and_store_batch(image_batch, label_batch, c)
                image_batch, label_batch = [], []

    if image_batch:
        process_and_store_batch(image_batch, label_batch, c)

    save_sample(c)
    conn.commit()
    conn.close()
