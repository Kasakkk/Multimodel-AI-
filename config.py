import os
import torch

# Best-available default: start with higher quality Moondream.
# If local compatibility issues occur, multimodal_model.py auto-falls back to BLIP.
MODEL_ID = "vikhyatk/moondream2"

# Automatically choose CUDA if a compatible GPU is found, otherwise fallback to CPU.
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

IMAGE_SIZE = (448, 448)

# Output paths
IMAGE_SAVE_PATH = "current_view.jpg"
LOG_FILE_PATH = "logs.json"

# Dataset paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
DATASET_TRAIN_CSV = os.path.join(DATA_DIR, "data_train.csv")
DATASET_EVAL_CSV = os.path.join(DATA_DIR, "data_eval.csv")
DATASET_IMAGES_DIR = os.path.join(DATA_DIR, "images")

if __name__ == "__main__":
    print(f"Config loaded. Using model: {MODEL_ID} on device: {DEVICE}")
