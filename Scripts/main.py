from Vectors.extraction_endpoint import extract_features
from Scripts.Utilities.knn_model_builder import knn_model_controller

def main():
    print("🚀 Starting feature extraction...")
    extract_features()
    knn_model_controller()

if __name__ == "__main__":
    main()
