# watch_db_and_rebuild.py
import os
import time
import hashlib
from config import DB_PATH, HASH_FILE, POLL_INTERVAL
from build_knn_model import build_and_save_knn

def get_file_hash(filepath):
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
    os.makedirs(os.path.dirname(HASH_FILE), exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(hash_value)

def main():
    last_hash = load_previous_hash()
    print("🔄 Watching database for changes...")

    while True:
        try:
            current_hash = get_file_hash(DB_PATH)
            if current_hash != last_hash:
                print("📦 Change detected in database. Rebuilding KNN model...")
                build_and_save_knn()
                save_hash(current_hash)
                last_hash = current_hash
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\n👋 Watcher stopped.")
            break

if __name__ == "__main__":
    main()
