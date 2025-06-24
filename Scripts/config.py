import os

ROOT_DIR = "./images"
DB_PATH = "./features.db"
BATCH_SIZE = 32
IMG_SIZE = (256, 256)
JSON_OUTPUT = "./image_features_sample.json"
CACHE_DIR = "cache"
HASH_FILE = "cache/db_hash.txt"
POLL_INTERVAL = 5  # seconds