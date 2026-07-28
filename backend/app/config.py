"""
Central configuration for the PixelTruth backend.

IMPORTANT: PixelTruth uses a CUSTOM-TRAINED convolutional neural network
(trained from scratch by the project author on the CIFAKE dataset). It does
NOT call any third-party AI-detection API. All numbers in TRAINING_METRICS
below are the real, reported evaluation numbers for that model and must not
be altered or fabricated.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_ROOT = BACKEND_DIR.parent                              # repo root
MODEL_PATH = BACKEND_DIR / "models" / "pixeltruth_model.keras"

FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"                       # built by `npm run build`
MODEL_INFO_ASSETS_DIR = FRONTEND_DIR / "public" / "model-info"  # confusion_matrix.png / training_curves.png

# ---------------------------------------------------------------------------
# Model input / label mapping
# ---------------------------------------------------------------------------
# The model's Input layer is (None, 64, 64, 3) -- see frontend/public/model/model.json
INPUT_SIZE = (64, 64)

# Label mapping used during training: 0 = FAKE (AI-generated), 1 = REAL.
# The model's final layer is Dense(1, activation="sigmoid"), so raw_output is
# P(class == REAL). We invert it to get an "AI probability":
#     ai_probability = 1 - raw_output
FAKE_LABEL = 0
REAL_LABEL = 1

# The model has an internal Rescaling(1./255) layer, so the backend must feed
# raw 0-255 pixel values into the model. Do NOT divide by 255 in preprocessing.

# ---------------------------------------------------------------------------
# Signal weights
# ---------------------------------------------------------------------------
# NOTE: The FFT frequency-domain signal and EXIF metadata signal have been
# removed from scoring. The final AI-probability score is now based solely
# on our custom-trained CNN model's output.
WEIGHT_CNN = 1.00   # our trained CNN model is the sole signal

# ---------------------------------------------------------------------------
# Verdict thresholds (on the final combined 0-100 "AI probability" score)
# ---------------------------------------------------------------------------
THRESHOLD_AI = 65.0          # score >= 65  -> "Likely AI-Generated"
THRESHOLD_AUTHENTIC = 35.0   # score <= 35  -> "Likely Authentic"
# anything in between        -> "Uncertain"

# ---------------------------------------------------------------------------
# REAL training / evaluation metrics for pixeltruth_model.keras.
# These are the actual reported numbers -- DO NOT CHANGE OR FABRICATE THESE.
# ---------------------------------------------------------------------------
TRAINING_METRICS = {
    "dataset": {
        "name": "CIFAKE",
        "description": (
            "Real vs. AI-generated (Stable Diffusion) image classification dataset."
        ),
        "total_images": 100000,
        "real_images": 50000,
        "fake_images": 50000,
        "train_val_split": "80/20",
    },
    "test_set": {
        "total_images": 20000,
        "real_images": 10000,
        "fake_images": 10000,
    },
    "overall_accuracy": 0.91,
    "per_class": {
        "FAKE": {"precision": 0.86, "recall": 0.97, "f1_score": 0.91},
        "REAL": {"precision": 0.97, "recall": 0.84, "f1_score": 0.90},
    },
    "architecture_summary": [
        "Input (64x64x3 RGB image)",
        "Data augmentation: RandomFlip (horizontal), RandomRotation (+/-10%), RandomZoom (10%) [training only]",
        "Rescaling(1/255) -- normalizes raw 0-255 pixel input internally",
        "Conv2D(32, 3x3, ReLU) -> BatchNormalization -> MaxPooling2D(2x2)",
        "Conv2D(64, 3x3, ReLU) -> BatchNormalization -> MaxPooling2D(2x2)",
        "Conv2D(128, 3x3, ReLU) -> BatchNormalization -> MaxPooling2D(2x2)",
        "Flatten",
        "Dense(128, ReLU)",
        "Dropout(0.5)",
        "Dense(1, Sigmoid) -- outputs P(REAL); AI probability = 1 - output",
    ],
    "training_details": {
        "optimizer": "Adam (lr=1e-3)",
        "loss": "Binary Crossentropy",
        "label_mapping": "0 = FAKE (AI-generated), 1 = REAL",
    },
}
