"""
Standalone diagnostic script for pixeltruth_model.keras.

WHY THIS EXISTS
------------------
Reported symptom: every uploaded image (through the deployed backend)
produces an identical raw_sigmoid_output of exactly 0.0, regardless of
image content. A careful line-by-line review of the backend pipeline
(model_service.py, image_utils.py, main.py) found no bug that would explain
an output that never changes with input -- the backend genuinely re-decodes,
resizes, and re-runs the model on each request.

This script bypasses the ENTIRE backend/FastAPI app and talks to the
.keras file directly, so we can determine definitively whether:
  (a) the model itself always outputs ~0.0 for any input (a training/
      weights issue -- e.g. the model collapsed to always predicting FAKE,
      or something went wrong saving/exporting the weights), or
  (b) the model actually DOES vary its output here, which would mean the
      problem is somewhere in the deployed environment (wrong TensorFlow/
      Keras version loading the file differently, a stale process that
      never picked up the new model file, etc.) rather than the model
      weights themselves.

USAGE
------
Run this on the SAME machine/environment where you have (or can get)
pixeltruth_model.keras, e.g. in Colab right after training, or locally
next to backend/models/pixeltruth_model.keras:

    python diagnose_model.py --model ../models/pixeltruth_model.keras

By default (no --images given), it generates a few synthetic test inputs
(solid colors, random noise) just to confirm the model produces DIFFERENT
outputs for different inputs at all -- if even synthetic random-noise
images and solid-color images all produce exactly the same output, that
is strong evidence the model's weights themselves are the problem (e.g.
a dead/collapsed network), not anything about real photos.

You can also pass real image files to test:

    python diagnose_model.py --model ../models/pixeltruth_model.keras \
      --images photo1.jpg photo2.jpg ai_generated.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose pixeltruth_model.keras directly.")
    parser.add_argument("--model", type=str, required=True, help="Path to pixeltruth_model.keras")
    parser.add_argument(
        "--images", type=str, nargs="*", default=[], help="Optional real image files to test"
    )
    parser.add_argument(
        "--input-size", type=int, default=128, help="Model's expected square input size (default 128)"
    )
    return parser.parse_args()


def make_synthetic_inputs(size: int) -> dict:
    """A handful of very different synthetic images -- if the model gives
    the exact same output for all of these, the weights themselves are the
    problem, not anything about real photos."""
    inputs = {}

    inputs["all_black"] = np.zeros((size, size, 3), dtype=np.float32)
    inputs["all_white"] = np.full((size, size, 3), 255.0, dtype=np.float32)
    inputs["all_gray_128"] = np.full((size, size, 3), 128.0, dtype=np.float32)

    rng = np.random.default_rng(42)
    inputs["random_noise_1"] = rng.uniform(0, 255, size=(size, size, 3)).astype(np.float32)
    rng2 = np.random.default_rng(999)
    inputs["random_noise_2"] = rng2.uniform(0, 255, size=(size, size, 3)).astype(np.float32)

    # A simple gradient -- structured but not photographic.
    grad = np.linspace(0, 255, size, dtype=np.float32)
    gradient_img = np.tile(grad, (size, 1))
    inputs["horizontal_gradient"] = np.stack([gradient_img] * 3, axis=-1)

    return inputs


def load_real_image(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"ERROR: model file not found at {model_path}")

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))
    print("Model loaded. Summary:")
    model.summary()

    print("\n" + "=" * 70)
    print("Testing model output on synthetic inputs (bypassing all app code)")
    print("=" * 70)
    synthetic = make_synthetic_inputs(args.input_size)
    outputs = {}
    for name, arr in synthetic.items():
        batch = np.expand_dims(arr, axis=0)
        pred = model.predict(batch, verbose=0)
        raw = float(pred[0, 0])
        outputs[name] = raw
        print(f"  {name:22s} -> raw_output (P(REAL)) = {raw:.6f}")

    unique_outputs = set(round(v, 6) for v in outputs.values())
    print(f"\nNumber of DISTINCT output values across {len(outputs)} very different synthetic inputs: {len(unique_outputs)}")
    if len(unique_outputs) == 1:
        print(
            "\n*** DIAGNOSIS: the model produces the EXACT SAME output for completely "
            "different synthetic inputs (solid colors, random noise, gradients). This "
            "strongly indicates a problem with the model's WEIGHTS ITSELF -- e.g. the "
            "network collapsed during training (a common failure mode is the final "
            "sigmoid saturating to 0 or 1 for every input, often caused by a learning "
            "rate that was too high, a labeling bug during training, or a bad "
            "BatchNormalization state saved into the file). This is NOT a backend/"
            "deployment bug -- retraining is very likely required.\n"
        )
    else:
        print(
            "\n*** DIAGNOSIS: the model DOES produce different outputs for different "
            "synthetic inputs. This means the model weights are NOT broken/collapsed. "
            "If the deployed backend still always shows 0.0 for every real photo, the "
            "problem is likely in the deployed ENVIRONMENT instead -- e.g. a different "
            "TensorFlow/Keras version loading the file differently than here, or a "
            "backend process that's stuck on an old/corrupted copy of the model file "
            "and never actually restarted after the new file was placed. Double-check "
            "the backend startup log's TensorFlow version against the one printed "
            "above, and confirm the running process was actually restarted (not just "
            "the code files updated) after the new model was uploaded.\n"
        )

    if args.images:
        print("=" * 70)
        print("Testing model output on your real image files")
        print("=" * 70)
        real_outputs = {}
        for path in args.images:
            try:
                arr = load_real_image(path, args.input_size)
                batch = np.expand_dims(arr, axis=0)
                pred = model.predict(batch, verbose=0)
                raw = float(pred[0, 0])
                real_outputs[path] = raw
                label = "REAL" if raw >= 0.5 else "FAKE"
                ai_prob = (1.0 - raw) * 100
                print(f"  {path:40s} -> raw_output={raw:.6f}  predicted={label}  ai_probability={ai_prob:.1f}%")
            except Exception as exc:
                print(f"  {path:40s} -> ERROR loading/predicting: {exc}")

        if real_outputs:
            unique_real = set(round(v, 6) for v in real_outputs.values())
            print(f"\nNumber of DISTINCT output values across {len(real_outputs)} real images: {len(unique_real)}")
            if len(unique_real) == 1 and len(unique_outputs) > 1:
                print(
                    "\n*** This is an interesting split result: the model responds "
                    "differently to synthetic inputs, but identically to your real "
                    "images. This could mean your real test images are unusually "
                    "similar to each other after resizing to "
                    f"{args.input_size}x{args.input_size} (e.g. all very dark/bright, "
                    "or all cropped/framed the same way), or that something about how "
                    "these specific files are being decoded differs from the synthetic "
                    "test arrays. Try images with very different lighting/framing/"
                    "content to narrow this down further.\n"
                )


if __name__ == "__main__":
    main()
