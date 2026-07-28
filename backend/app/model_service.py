"""
Loads and serves PixelTruth's CUSTOM-TRAINED CNN model
(backend/models/pixeltruth_model.keras).

This is a model trained from scratch on the CIFAKE dataset (see
app/config.py::TRAINING_METRICS for the real, reported evaluation numbers).
There is no call to any external AI-detection API anywhere in this module --
every prediction and every saliency gradient below comes from weights baked
into pixeltruth_model.keras.

The model architecture (see frontend/public/model/model.json) is:
    Input(64,64,3)
      -> [augmentation layers -- inference no-ops]
      -> Rescaling(1/255)          <-- model normalizes internally!
      -> Conv2D(32) -> BatchNorm -> MaxPool
      -> Conv2D(64) -> BatchNorm -> MaxPool
      -> Conv2D(128) -> BatchNorm -> MaxPool
      -> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(1, sigmoid)

Label mapping used during training: 0 = FAKE (AI-generated), 1 = REAL.
The sigmoid output is therefore P(REAL); we invert it to report an
"AI probability" = 1 - P(REAL).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image

from app.config import INPUT_SIZE, MODEL_PATH

logger = logging.getLogger("pixeltruth.model")

_model = None
_model_load_error: Optional[str] = None


def load_model() -> None:
    """Load the .keras model once at process startup. Call this from the
    FastAPI startup event -- NEVER load the model per-request."""
    global _model, _model_load_error
    try:
        logger.info("Loading custom-trained PixelTruth CNN from %s ...", MODEL_PATH)
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        # Warm up / build the graph once so the first real request isn't slow.
        dummy = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.float32)
        _model.predict(dummy, verbose=0)
        logger.info(
            "[PixelTruth] Custom-trained CNN model loaded successfully from %s "
            "(no external AI API used).",
            MODEL_PATH,
        )
    except Exception as exc:  # noqa: BLE001 -- we want to capture and report any failure
        _model = None
        _model_load_error = str(exc)
        logger.exception("[PixelTruth] FAILED to load custom CNN model: %s", exc)


def is_loaded() -> bool:
    return _model is not None


def load_error() -> Optional[str]:
    return _model_load_error


def _preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize to the model's expected 64x64 input and return RAW 0-255 pixel
    values (float32) with shape (1, 64, 64, 3). Do NOT divide by 255 here --
    the model has its own internal Rescaling(1/255) layer that does this.
    """
    resized = image.convert("RGB").resize(INPUT_SIZE, resample=Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)  # (64, 64, 3), values 0-255
    return np.expand_dims(arr, axis=0)


def predict_with_saliency(image: Image.Image) -> dict:
    """
    Run the custom CNN on `image` and compute a genuine input-gradient
    saliency map (gradient of the model's raw output w.r.t. input pixels).
    Because this is our own model, we have full access to its gradients --
    this is a real gradient-based explanation, not a third-party approximation.

    Returns a dict with:
        raw_output      float  -- sigmoid output, i.e. P(REAL) in [0, 1]
        ai_probability  float  -- 1 - raw_output, in [0, 1]
        predicted_label str    -- "REAL" or "FAKE" (whichever the model favors)
        saliency        np.ndarray (64, 64) -- normalized per-pixel importance
    """
    if _model is None:
        raise RuntimeError("PixelTruth CNN model is not loaded")

    input_batch = _preprocess(image)  # (1, 64, 64, 3), raw pixel values
    input_tensor = tf.convert_to_tensor(input_batch)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = _model(input_tensor, training=False)  # shape (1, 1), P(REAL)
        scalar_output = output[0, 0]

    grads = tape.gradient(scalar_output, input_tensor)  # (1, 64, 64, 3)
    if grads is None:
        # Extremely defensive fallback -- should not happen for this architecture.
        saliency = np.zeros(INPUT_SIZE, dtype=np.float32)
    else:
        # Max across color channels -> classic saliency-map reduction.
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()

    raw_output = float(scalar_output.numpy())
    ai_probability = 1.0 - raw_output
    predicted_label = "REAL" if raw_output >= 0.5 else "FAKE"

    return {
        "raw_output": raw_output,
        "ai_probability": ai_probability,
        "predicted_label": predicted_label,
        "saliency": saliency,
    }
