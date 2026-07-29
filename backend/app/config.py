"""
Central configuration for the PixelTruth backend.

IMPORTANT: PixelTruth uses a CUSTOM-TRAINED convolutional neural network
(trained from scratch by the project author on the "140k Real and Fake
Faces" dataset). It does NOT call any third-party AI-detection API. All
numbers in TRAINING_METRICS below are the real, reported evaluation
numbers for that model and must not be altered or fabricated.

All inference happens server-side in this backend via tf.keras -- there is
no browser-based TensorFlow.js inference anywhere in this app.
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
# The model's Input layer is (None, 128, 128, 3) -- confirmed from the
# .keras file's config.json. This model was trained on 128x128 face crops
# (up from the earlier 64x64 CIFAKE-based model) to preserve more of the
# fine detail needed to catch modern generator artifacts.
INPUT_SIZE = (128, 128)

# Label mapping used during training: 0 = FAKE (AI-generated), 1 = REAL.
# The model's final layer is Dense(1, activation="sigmoid"), so raw_output is
# P(class == REAL). We invert it to get an "AI probability":
#     ai_probability = 1 - raw_output
FAKE_LABEL = 0
REAL_LABEL = 1

# The model has an internal Rescaling(1./255) layer (confirmed scale=1/255
# in the .keras file's config.json), so the backend must feed raw 0-255
# pixel values into the model. Do NOT divide by 255 in preprocessing.

# ---------------------------------------------------------------------------
# Signal weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_CNN = 0.60   # our trained CNN model
WEIGHT_FFT = 0.20   # frequency-domain / FFT periodic-artifact analysis
WEIGHT_EXIF = 0.20  # metadata / EXIF presence check

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
        "name": "140k Real and Fake Faces",
        "description": (
            "Real human face photographs (from the FFHQ / Flickr-Faces "
            "dataset) vs. AI-generated fake faces produced by StyleGAN."
        ),
        "total_images": 140000,
        "train_images": 100000,
        "valid_images": 20000,
        "test_images": 20000,
        "split_description": "100,000 train / 20,000 valid / 20,000 test (perfectly balanced)",
        "train_val_test_split": "100k / 20k / 20k",
    },
    "test_set": {
        "total_images": 20000,
        "real_images": 10000,
        "fake_images": 10000,
    },
    "overall_accuracy": 0.94,
    "per_class": {
        "FAKE": {"precision": 0.96, "recall": 0.91, "f1_score": 0.93},
        "REAL": {"precision": 0.91, "recall": 0.96, "f1_score": 0.94},
    },
    "architecture_summary": [
        "Input (128x128x3 RGB image)",
        "Data augmentation: RandomFlip (horizontal), RandomRotation (+/-10%), RandomZoom (10%), RandomBrightness (+/-10%) [training only]",
        "Rescaling(1/255) -- normalizes raw 0-255 pixel input internally",
        "Conv2D(32, 3x3, ReLU) -> BatchNormalization -> MaxPooling2D(2x2)",
        "Conv2D(64, 3x3, ReLU) -> BatchNormalization -> MaxPooling2D(2x2)",
        "Conv2D(128, 3x3, ReLU) -> BatchNormalization -> MaxPooling2D(2x2)",
        "Conv2D(256, 3x3, ReLU) -> BatchNormalization -> MaxPooling2D(2x2)",
        "Flatten",
        "Dense(128, ReLU)",
        "Dropout(0.5)",
        "Dense(1, Sigmoid) -- outputs P(REAL); AI probability = 1 - output",
    ],
    "training_details": {
        "optimizer": "Adam",
        "loss": "Binary Crossentropy",
        "label_mapping": "0 = FAKE (AI-generated), 1 = REAL",
        "input_resolution": "128x128 (increased from an earlier 64x64 CIFAKE-based model to preserve fine detail)",
        "notes": (
            "Trained on real photographed faces (FFHQ/Flickr) vs. StyleGAN- "
            "generated fake faces. Because this dataset is faces-only, "
            "detection accuracy is highest on photos that contain a clear "
            "human face; the model was not trained on generic non-face "
            "AI-generated imagery (e.g. landscapes, objects, illustrations)."
        ),
    },
}
