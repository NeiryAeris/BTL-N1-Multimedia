import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import gc
import json
import sqlite3
import numpy as np
from Scripts.config import DB_PATH, JSON_OUTPUT

# ----------Database Initialization and Loading Functions----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS image_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            label TEXT,
            color_histogram TEXT,
            hog_vector TEXT,
            rgb_vector TEXT,
            shape_descriptor TEXT,
            texture_descriptor TEXT
        )
    """
    )
    return conn, c

def insert_features(cursor, data):
    cursor.executemany(
        """
        INSERT INTO image_features (
            image_path, 
            label,
            color_histogram,
            hog_vector,
            rgb_vector,
            shape_descriptor,
            texture_descriptor
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        data,
    )

def save_sample(cursor):
    cursor.execute("SELECT * FROM image_features LIMIT 10;")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    with open(JSON_OUTPUT, "w") as f:
        json.dump([dict(zip(columns, row)) for row in rows], f, indent=4)

def load_database_features():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT image_path, color_histogram, hog_vector, rgb_vector, shape_descriptor, texture_descriptor FROM image_features"
    )
    rows = cursor.fetchall()
    conn.close()

    image_paths = []
    color_features = []
    hog_vector = []
    rgb_vector = []
    shape_features = []
    texture_features = []

    for row in rows:
        image_paths.append(row[0])
        color_features.append(json.loads(row[1]))
        hog_vector.append(json.loads(row[2]))
        rgb_vector.append(json.loads(row[3]))
        shape_features.append(json.loads(row[4]))
        texture_features.append(json.loads(row[5]))

    return (
        image_paths,
        np.array(color_features),
        np.array(hog_vector),
        np.array(rgb_vector),
        np.array(shape_features),
        np.array(texture_features),
    )

# ----------<>----------