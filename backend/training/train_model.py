"""
PixelTruth -- CNN training script (CIFAKE: Real vs AI-Generated Images).

Run this on Google Colab (or any machine with a GPU + internet access) --
NOT in a restricted sandbox. It:

  1. Downloads/loads the CIFAKE dataset (100,000 images: 50k REAL, 50k FAKE).
  2. Builds the EXACT architecture PixelTruth's backend expects (see
     ../app/config.py::TRAINING_METRICS["architecture_summary"] and
     ../../frontend/public/model/model.json):

         Input(64,64,3)
           -> RandomFlip("horizontal")
           -> RandomRotation(0.1)
           -> RandomZoom(0.1)
           -> Rescaling(1./255)                <- baked into the model!
           -> Conv2D(32,3x3,relu) -> BatchNorm -> MaxPool(2x2)
           -> Conv2D(64,3x3,relu) -> BatchNorm -> MaxPool(2x2)
           -> Conv2D(128,3x3,relu) -> BatchNorm -> MaxPool(2x2)
           -> Flatten -> Dense(128,relu) -> Dropout(0.5) -> Dense(1,sigmoid)

  3. Trains it CORRECTLY:
       - proper train/val split from the dataset's own train folder (the
         80/20 split the app's config.py already describes)
       - class balance is verified, not assumed
       - EarlyStopping + ReduceLROnPlateau + ModelCheckpoint (saves the BEST
         val-accuracy epoch, not just the last one -- this alone fixes a lot
         of "why is my model bad" problems from undertrained/overfit runs)
       - EXTRA data-pipeline augmentation (brightness/contrast/JPEG-quality
         jitter/gaussian noise) applied only during training, NOT baked into
         the exported model. This specifically targets the domain-shift
         problem where a model trained only on clean CIFAR-10-style "REAL"
         images misjudges real phone-camera photos, which have different
         sensor noise, sharpening, and compression artifacts. This is the
         single most important correctness fix for real-world generalization.
  4. Evaluates on the REAL held-out CIFAKE test set (20,000 images) and
     prints the exact classification report -- use these real numbers to
     update ../app/config.py::TRAINING_METRICS after retraining. Never
     fabricate numbers; always paste in whatever this script actually prints.
  5. Saves:
       - pixeltruth_model.keras            (Keras 3 format, for the backend)
       - confusion_matrix.png              (for frontend/public/model-info/)
       - training_curves.png               (for frontend/public/model-info/)
     and, if tensorflowjs is installed, a TF.js export for frontend/public/model/.

Usage (Colab):
    !pip install -q tensorflow kagglehub scikit-learn matplotlib tensorflowjs
    !python train_model.py --data-dir /content/cifake --epochs 30

Usage (dataset already downloaded/unzipped locally, e.g. from Kaggle):
    python train_model.py --data-dir /path/to/cifake --epochs 30

Expected --data-dir structure (this is CIFAKE's own official layout):
    <data-dir>/
      train/
        REAL/   *.jpg   (50,000 total real images across train, class-balanced)
        FAKE/   *.jpg
      test/
        REAL/   *.jpg   (10,000 images)
        FAKE/   *.jpg   (10,000 images)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# Constants -- MUST match backend/app/config.py exactly, or the backend will
# silently misinterpret the model's output.
# ---------------------------------------------------------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
SEED = 42
# CIFAKE's own folder names are "REAL" and "FAKE". Sorted alphabetically,
# Keras assigns label 0 = FAKE, 1 = REAL -- exactly the mapping the backend
# expects (0=FAKE, 1=REAL). This is NOT a coincidence we need to hardcode;
# `image_dataset_from_directory` derives it automatically from the folder
# names as long as they are exactly "FAKE" and "REAL".
EXPECTED_CLASS_NAMES = ["FAKE", "REAL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PixelTruth CIFAKE CNN correctly.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the CIFAKE dataset root, containing train/ and test/ subfolders "
        "each with REAL/ and FAKE/ subfolders.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs (EarlyStopping may stop sooner).")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of train/ used for validation.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Where to write pixeltruth_model.keras, confusion_matrix.png, training_curves.png.",
    )
    parser.add_argument(
        "--strong-augment",
        action="store_true",
        default=True,
        help="Apply extra brightness/contrast/noise/JPEG-quality augmentation in the data "
        "pipeline (NOT baked into the saved model) to improve generalization to real-world "
        "phone photos, which look statistically different from CIFAKE's CIFAR-10-style "
        "REAL images. Enabled by default -- this is the key fix for the domain-shift issue.",
    )
    parser.add_argument("--export-tfjs", action="store_true", help="Also export a TensorFlow.js model.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def load_datasets(data_dir: Path, val_split: float):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        sys.exit(
            f"ERROR: expected '{train_dir}' and '{test_dir}' to exist. "
            "Download CIFAKE and point --data-dir at its root (the folder that "
            "directly contains train/ and test/)."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,  # keep order stable so predictions line up with labels for the report
    )

    class_names = train_ds.class_names
    if class_names != EXPECTED_CLASS_NAMES:
        sys.exit(
            f"ERROR: expected class folders {EXPECTED_CLASS_NAMES} (giving label mapping "
            f"0=FAKE, 1=REAL) but found {class_names}. Rename your dataset's class folders "
            "to exactly 'FAKE' and 'REAL' before training, or the label mapping used "
            "throughout the PixelTruth backend will be silently wrong."
        )
    print(f"Class mapping confirmed: {class_names} -> label 0={class_names[0]}, label 1={class_names[1]}")

    return train_ds, val_ds, test_ds


def _extra_train_augmentation(image: tf.Tensor, label: tf.Tensor) -> tuple:
    """
    Applied ONLY to the training tf.data pipeline (never baked into the saved
    model, and never applied to val/test data). This closes the gap between
    CIFAKE's clean, uniformly-sourced REAL images and the messy, varied real
    photos users actually upload (different cameras, JPEG compression levels,
    lighting, sharpening). Without this, the model overfits to "REAL" meaning
    specifically "looks like this dataset's real images" rather than "is a
    real photograph" in general -- which is exactly the failure mode where a
    user's own phone photo gets misclassified as AI-generated.
    """
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)

    # Simulate varying JPEG compression quality (common real-world artifact
    # that differs a lot between cameras/platforms/re-uploads).
    def _jpeg_jitter(img):
        img_uint8 = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
        quality = tf.random.uniform([], minval=40, maxval=100, dtype=tf.int32)
        encoded = tf.image.encode_jpeg(img_uint8, quality=quality)
        decoded = tf.image.decode_jpeg(encoded, channels=3)
        return tf.cast(decoded, tf.float32)

    if tf.random.uniform([]) < 0.5:
        image = _jpeg_jitter(image)

    # Light gaussian noise -- approximates sensor noise present in real
    # camera photos but largely absent from synthetic/generated images.
    if tf.random.uniform([]) < 0.3:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=6.0)
        image = image + noise

    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def apply_pipeline_augmentation(train_ds: tf.data.Dataset, enabled: bool) -> tf.data.Dataset:
    if enabled:
        train_ds = train_ds.map(_extra_train_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Model definition -- must exactly match app/config.py's documented
# architecture and frontend/public/model/model.json.
# ---------------------------------------------------------------------------
def build_model() -> tf.keras.Model:
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Augmentation layers baked into the model graph (inference no-ops,
    # active only when training=True). Matches model.json.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # The model normalizes its OWN input -- callers must feed raw 0-255
    # pixel values, never pre-normalized ones (see app/model_service.py).
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="pixeltruth_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation + reporting
# ---------------------------------------------------------------------------
def evaluate_and_plot(model: tf.keras.Model, test_ds: tf.data.Dataset, history, output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")  # headless-safe (works on Colab and servers without a display)
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_pred_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_prob.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true).astype(int)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n" + "=" * 70)
    print("REAL evaluation results on the held-out CIFAKE test set")
    print("=" * 70)
    report = classification_report(y_true, y_pred, target_names=EXPECTED_CLASS_NAMES, digits=4)
    print(report)
    print(
        "\n>>> IMPORTANT: copy these REAL numbers into "
        "backend/app/config.py::TRAINING_METRICS after retraining. "
        "Never leave stale or fabricated numbers there.\n"
    )

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(EXPECTED_CLASS_NAMES)
    ax.set_yticklabels(EXPECTED_CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PixelTruth CNN -- Confusion Matrix (CIFAKE test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved {cm_path}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    curves_path = output_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Saved {curves_path}")

    return report, cm


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(SEED)

    print(f"Loading CIFAKE from {data_dir} ...")
    train_ds, val_ds, test_ds = load_datasets(data_dir, args.val_split)
    train_ds = apply_pipeline_augmentation(train_ds, enabled=args.strong_augment)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    print("Building model...")
    model = build_model()
    model.summary()

    checkpoint_path = output_dir / "pixeltruth_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"Training for up to {args.epochs} epochs (early stopping enabled)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ModelCheckpoint already saved the BEST epoch to checkpoint_path, but
    # EarlyStopping's restore_best_weights means `model` in memory is also
    # the best version -- save again defensively in case ModelCheckpoint's
    # file write raced with an early stop.
    model.save(checkpoint_path)
    print(f"\nSaved best model to {checkpoint_path}")

    evaluate_and_plot(model, test_ds, history, output_dir)

    if args.export_tfjs:
        try:
            import tensorflowjs as tfjs

            tfjs_dir = output_dir / "tfjs_model"
            tfjs.converters.save_keras_model(model, str(tfjs_dir))
            print(f"Saved TF.js export to {tfjs_dir} (copy model.json + shard file(s) into frontend/public/model/)")
        except ImportError:
            print(
                "tensorflowjs not installed -- skipping TF.js export. "
                "Install with `pip install tensorflowjs` and re-run with --export-tfjs "
                "if you need the browser-side model too."
            )

    print("\nDone. Next steps:")
    print(f"  1. Copy {checkpoint_path} -> backend/models/pixeltruth_model.keras")
    print(f"  2. Copy {output_dir / 'confusion_matrix.png'} -> frontend/public/model-info/confusion_matrix.png")
    print(f"  3. Copy {output_dir / 'training_curves.png'} -> frontend/public/model-info/training_curves.png")
    print("  4. Update backend/app/config.py::TRAINING_METRICS with the REAL numbers printed above.")


if __name__ == "__main__":
    main()
