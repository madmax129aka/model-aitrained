"""
Loads and serves PixelTruth's CUSTOM-TRAINED CNN model
(backend/models/pixeltruth_model.keras).

This is a model trained from scratch on the "140k Real and Fake Faces"
dataset (see app/config.py::TRAINING_METRICS for the real, reported
evaluation numbers). There is no call to any external AI-detection API
anywhere in this module -- every prediction and every saliency gradient
below comes from weights baked into pixeltruth_model.keras, and all
inference runs server-side in this process (no browser-based TensorFlow.js).

The model architecture (see backend/app/config.py::INPUT_SIZE for the
current input resolution, and TRAINING_METRICS["architecture_summary"] for
the full layer-by-layer breakdown) is:
    Input(INPUT_SIZE, INPUT_SIZE, 3)
      -> [augmentation layers -- inference no-ops]
      -> Rescaling(1/255)          <-- model normalizes internally!
      -> Conv2D(32) -> BatchNorm -> MaxPool
      -> Conv2D(64) -> BatchNorm -> MaxPool
      -> Conv2D(128) -> BatchNorm -> MaxPool
      -> Conv2D(256) -> BatchNorm -> MaxPool
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

# A second "headless" model sharing the same weights as _model, but with the
# final Dense layer's sigmoid activation stripped off (outputs the raw
# pre-activation logit instead of the squashed [0,1] probability). Used ONLY
# for computing saliency gradients -- see _build_logits_model() below for why.
_logits_model = None


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
        _build_logits_model()
        logger.info(
            "[PixelTruth] Custom-trained CNN model loaded successfully from %s "
            "(no external AI API used).",
            MODEL_PATH,
        )
    except Exception as exc:  # noqa: BLE001 -- we want to capture and report any failure
        _model = None
        _model_load_error = str(exc)
        logger.exception("[PixelTruth] FAILED to load custom CNN model: %s", exc)


def _build_logits_model() -> None:
    """
    Build a sub-model that outputs the PRE-sigmoid logit instead of the final
    [0,1] probability, sharing the exact same trained weights as `_model`.

    Why this matters: computing saliency as the gradient of the *sigmoid
    output* w.r.t. the input pixels suffers from vanishing gradients whenever
    the model is confident (output near 0 or near 1) -- the sigmoid's own
    derivative approaches zero at saturation, so the entire gradient chain
    collapses to ~0 everywhere, producing a flat/empty heatmap even though
    the model's decision was influenced by specific pixels. This is a
    well-documented issue with sigmoid/softmax-output saliency maps.

    Differentiating the pre-activation logit instead avoids this: the
    logit's gradient does not vanish at saturation, so the saliency map
    stays meaningful even for very confident predictions (which is exactly
    when users most want to see *why* the model is so sure).
    """
    global _logits_model
    try:
        last_layer = _model.layers[-1]
        penultimate_output = _model.layers[-2].output

        # Dense layers cannot accept pretrained weights via the constructor
        # (Keras requires a layer to be built -- i.e. called on an input --
        # before its weights can be set). So build a fresh, unactivated Dense
        # layer by calling it on the penultimate layer's output, THEN copy
        # over the original layer's trained kernel/bias with set_weights().
        logits_head = tf.keras.layers.Dense(
            units=last_layer.units, activation=None, name="saliency_logits_head"
        )
        logits = logits_head(penultimate_output)
        logits_head.set_weights(last_layer.get_weights())

        _logits_model = tf.keras.Model(inputs=_model.input, outputs=logits)
        logger.info("Built logits sub-model for saturation-safe saliency computation.")
    except Exception as exc:  # noqa: BLE001
        # Defensive fallback: if the architecture ever changes such that this
        # doesn't apply cleanly, saliency computation below falls back to
        # differentiating the sigmoid output directly (the old behavior).
        _logits_model = None
        logger.warning(
            "Could not build logits sub-model for saliency (falling back to "
            "sigmoid-output gradients, which may vanish at saturation): %s",
            exc,
        )


def is_loaded() -> bool:
    return _model is not None


def load_error() -> Optional[str]:
    return _model_load_error


def _preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize to the model's expected input size (app/config.py::INPUT_SIZE)
    and return RAW 0-255 pixel values (float32) with shape
    (1, INPUT_SIZE[0], INPUT_SIZE[1], 3). Do NOT divide by 255 here -- the
    model has its own internal Rescaling(1/255) layer that does this.

    NOTE: an earlier CIFAKE-based version of this model (64x64 input) was
    found to key off image sharpness/blur as a shortcut signal rather than
    genuine AI-generation artifacts, because CIFAKE's source images are
    natively only 32x32 pixels. The current model was retrained from
    scratch on the "140k Real and Fake Faces" dataset at 128x128 input to
    address this -- see backend/training/README.md for the full history.
    """
    resized = image.convert("RGB").resize(INPUT_SIZE, resample=Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)  # (H, W, 3), values 0-255
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
        saliency        np.ndarray (INPUT_SIZE) -- normalized per-pixel importance
    """
    if _model is None:
        raise RuntimeError("PixelTruth CNN model is not loaded")

    input_batch = _preprocess(image)  # (1, H, W, 3), raw pixel values, H/W = INPUT_SIZE
    input_tensor = tf.convert_to_tensor(input_batch)

    # The actual prediction/verdict always comes from the real sigmoid model
    # (P(REAL) must stay a calibrated probability in [0,1] for the verdict
    # thresholds in config.py to mean anything).
    output = _model(input_tensor, training=False)  # shape (1, 1), P(REAL)
    raw_output = float(output[0, 0].numpy())

    # Saliency gradients, however, are computed against the LOGITS model
    # when available (see _build_logits_model's docstring for why) --
    # differentiating the raw sigmoid output vanishes to ~0 everywhere for
    # confident predictions, producing an empty/flat heatmap identical to
    # the original image, exactly when the model is most confident.
    gradient_target_model = _logits_model if _logits_model is not None else _model

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        grad_output = gradient_target_model(input_tensor, training=False)
        scalar_grad_output = grad_output[0, 0]

    grads = tape.gradient(scalar_grad_output, input_tensor)  # (1, H, W, 3), H/W = INPUT_SIZE
    if grads is None:
        # Extremely defensive fallback -- should not happen for this architecture.
        saliency = np.zeros(INPUT_SIZE, dtype=np.float32)
    else:
        # Max across color channels -> classic saliency-map reduction.
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()

    ai_probability = 1.0 - raw_output
    predicted_label = "REAL" if raw_output >= 0.5 else "FAKE"

    return {
        "raw_output": raw_output,
        "ai_probability": ai_probability,
        "predicted_label": predicted_label,
        "saliency": saliency,
    }
