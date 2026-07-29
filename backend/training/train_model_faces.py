"""
PixelTruth -- CNN training script for the 140k Real and Fake Faces dataset
(Kaggle: xhlulu/140k-real-and-fake-faces).

WHY THIS SCRIPT EXISTS
------------------------
Testing of the original CIFAKE-trained model (see train_model.py) confirmed
it was relying on image sharpness/blur as a shortcut signal rather than
learning genuine AI-generation artifacts -- CIFAKE's source images are
natively only 32x32 pixels, so the model never saw real-world-resolution
photos of either class. Fine-tuning on a small custom dataset
(finetune_model.py) is one fix; this script is a more thorough one: it
trains a NEW model from scratch on the 140k Real and Fake Faces dataset,
which:
  - uses REAL, full-resolution photographed faces (from Flickr, via NVIDIA's
    FFHQ dataset) instead of 32x32-upscaled CIFAR-10 images
  - uses GAN-generated (StyleGAN) FAKE faces instead of CIFAKE's Stable
    Diffusion images, which is a different (and for faces specifically,
    often harder) generation artifact to detect
  - is a much larger, higher-resolution dataset (140,000 images total)

This also bumps the model's input resolution from 64x64 to 128x128 --
64x64 was found to be part of why fine details needed to catch modern
generators get lost. IMPORTANT: this means INPUT_SIZE in
backend/app/config.py and the preprocessing in backend/app/model_service.py
MUST be updated to (128, 128) to match -- this is not automatic, see the
"After training" steps at the bottom of this file.

TRADE-OFF TO UNDERSTAND: this dataset is FACES-ONLY (real photographed
faces vs. GAN-generated faces). A model trained on it will likely perform
worse on non-face AI-generated images (landscapes, objects, illustrations,
etc.) than the original CIFAKE-based model did on those. If your use case
is detecting AI-generated images in general (not just faces), keep that in
mind when evaluating this model's real-world results.

EXPECTED DATASET STRUCTURE
-----------------------------
After downloading and unzipping the Kaggle dataset, the standard published
layout is:

    <data-dir>/
      real_vs_fake/
        real-vs-fake/
          train/
            real/   *.jpg
            fake/   *.jpg
          valid/
            real/   *.jpg
            fake/   *.jpg
          test/
            real/   *.jpg
            fake/   *.jpg

This script AUTO-DETECTS that nested layout (and a couple of simpler
variants, in case Kaggle changes the packaging) by searching --data-dir for
a subdirectory that itself contains train/valid/test folders, each with
real/ and fake/ subfolders. If it can't find this layout, it prints exactly
what it found so you can paste that back for a quick fix.

Label mapping: this dataset's own folder names are lowercase "real" and
"fake" (not CIFAKE's uppercase "REAL"/"FAKE"). Alphabetically, Keras will
still assign label 0="fake", 1="real" -- the SAME 0=FAKE/1=REAL convention
used everywhere else in this project, just with different folder-name
casing. No change needed to how the backend interprets the model's output.

USAGE (Colab)
--------------
    !pip install -q tensorflow scikit-learn matplotlib
    !kaggle datasets download -d xhlulu/140k-real-and-fake-faces
    !unzip -q 140k-real-and-fake-faces.zip -d /content/data_faces
    !python train_model_faces.py --data-dir /content/data_faces --epochs 20 --output-dir ./output_faces
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (128, 128)
BATCH_SIZE = 64
SEED = 42
EXPECTED_CLASS_NAMES = ["fake", "real"]  # lowercase in this dataset; label 0=fake, 1=real


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PixelTruth CNN on the 140k Real and Fake Faces dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the unzipped 140k-real-and-fake-faces dataset root "
        "(the folder you passed to `unzip -d`).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs (EarlyStopping may stop sooner).")
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
        help="Apply extra brightness/contrast/JPEG-quality/noise augmentation in the training "
        "data pipeline (not baked into the saved model), same rationale as train_model.py.",
    )
    parser.add_argument("--export-tfjs", action="store_true", help="Also export a TensorFlow.js model.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset auto-detection
# ---------------------------------------------------------------------------
def _looks_like_split_root(candidate: Path) -> bool:
    """True if `candidate` directly contains train/valid/test dirs, each
    with real/ and fake/ subfolders (case-insensitive)."""
    split_names = {"train", "valid", "test"}
    subdirs = {p.name.lower() for p in candidate.iterdir() if p.is_dir()}
    if not split_names.issubset(subdirs):
        return False
    for split in split_names:
        split_dir = next(p for p in candidate.iterdir() if p.is_dir() and p.name.lower() == split)
        class_subdirs = {p.name.lower() for p in split_dir.iterdir() if p.is_dir()}
        if not {"real", "fake"}.issubset(class_subdirs):
            return False
    return True


def find_split_root(data_dir: Path) -> Optional[Path]:
    """Search up to 4 levels deep under data_dir for the train/valid/test
    layout, since Kaggle's zip nests this dataset under
    real_vs_fake/real-vs-fake/ by default."""
    if _looks_like_split_root(data_dir):
        return data_dir

    candidates = [data_dir]
    for depth in range(4):
        next_candidates = []
        for c in candidates:
            if not c.is_dir():
                continue
            for child in c.iterdir():
                if child.is_dir():
                    if _looks_like_split_root(child):
                        return child
                    next_candidates.append(child)
        candidates = next_candidates
    return None


def _resolve_class_subdir(split_dir: Path, want: str) -> Path:
    """Find the real/ or fake/ subdir under split_dir regardless of case."""
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.lower() == want:
            return child
    raise FileNotFoundError(f"Could not find a '{want}' subfolder under {split_dir}")


def load_datasets(data_dir: Path):
    split_root = find_split_root(data_dir)
    if split_root is None:
        print(f"\nERROR: could not find the expected train/valid/test + real/fake layout under {data_dir}.")
        print("Here is what was actually found (paste this back if you need help adjusting the script):\n")
        for path in sorted(data_dir.rglob("*")):
            if path.is_dir():
                print(f"  DIR:  {path}")
        sys.exit(1)

    print(f"Found dataset split root at: {split_root}")

    def _norm_split_dir(name: str) -> Path:
        for child in split_root.iterdir():
            if child.is_dir() and child.name.lower() == name:
                return child
        raise FileNotFoundError(f"Missing '{name}' split under {split_root}")

    train_dir = _norm_split_dir("train")
    valid_dir = _norm_split_dir("valid")
    test_dir = _norm_split_dir("test")

    # image_dataset_from_directory needs the class subfolders to exist with
    # consistent naming across the directory passed to it. This dataset's
    # folders are already lowercase "real"/"fake" in every split, so we can
    # pass each split dir directly.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir, seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names
    if class_names != EXPECTED_CLASS_NAMES:
        sys.exit(
            f"ERROR: expected class folders {EXPECTED_CLASS_NAMES} (giving label mapping "
            f"0=fake, 1=real) but found {class_names}. This dataset's folders should be "
            "lowercase 'real' and 'fake' -- check your download."
        )
    print(f"Class mapping confirmed: {class_names} -> label 0={class_names[0]}, label 1={class_names[1]}")

    train_count = sum(1 for _ in train_dir.rglob("*") if _.is_file())
    valid_count = sum(1 for _ in valid_dir.rglob("*") if _.is_file())
    test_count = sum(1 for _ in test_dir.rglob("*") if _.is_file())
    print(f"train: {train_count} images | valid: {valid_count} images | test: {test_count} images")

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Training-only augmentation (same rationale as train_model.py)
# ---------------------------------------------------------------------------
def _extra_train_augmentation(image: tf.Tensor, label: tf.Tensor) -> tuple:
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)

    def _jpeg_jitter_single(img):
        img_uint8 = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
        jittered = tf.image.random_jpeg_quality(img_uint8, min_jpeg_quality=40, max_jpeg_quality=100)
        return tf.cast(jittered, tf.float32)

    def _jpeg_jitter_batch(img_batch):
        return tf.map_fn(_jpeg_jitter_single, img_batch, fn_output_signature=tf.float32)

    if tf.random.uniform([]) < 0.5:
        image = _jpeg_jitter_batch(image)

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
# Model definition -- same block structure as train_model.py, but at
# 128x128 input. After 3x MaxPool(2), spatial dims go 128 -> 64 -> 32 -> 16,
# so the Flatten layer here is 4x larger (16x16x128=32768) than the 64x64
# version's (8x8x128=8192). This is expected and fine.
# ---------------------------------------------------------------------------
def build_model() -> tf.keras.Model:
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
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

    model = models.Model(inputs, outputs, name="pixeltruth_cnn_faces")
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

    matplotlib.use("Agg")
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
    print("REAL evaluation results on the held-out 140k-faces test set")
    print("=" * 70)
    report = classification_report(
        y_true, y_pred, target_names=[n.upper() for n in EXPECTED_CLASS_NAMES], digits=4
    )
    print(report)
    print(
        "\n>>> IMPORTANT: copy these REAL numbers into "
        "backend/app/config.py::TRAINING_METRICS after retraining. "
        "Never leave stale or fabricated numbers there. Also update the "
        "'dataset' section to describe the 140k Real and Fake Faces dataset "
        "instead of CIFAKE.\n"
    )

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels_upper = [n.upper() for n in EXPECTED_CLASS_NAMES]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_upper)
    ax.set_yticklabels(labels_upper)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PixelTruth CNN (faces) -- Confusion Matrix (140k faces test set)")
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

    print(f"Loading 140k Real and Fake Faces dataset from {data_dir} ...")
    train_ds, val_ds, test_ds = load_datasets(data_dir)
    train_ds = apply_pipeline_augmentation(train_ds, enabled=args.strong_augment)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    print(f"Building model at {IMG_SIZE[0]}x{IMG_SIZE[1]} input resolution...")
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
            filepath=str(checkpoint_path), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print(f"Training for up to {args.epochs} epochs (early stopping enabled)...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

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
                "Install separately (see requirements-tfjs.txt) and re-run with --export-tfjs "
                "if you need the browser-side model too."
            )

    print("\nDone. Next steps (IMPORTANT -- more than just copying the model file this time):")
    print(f"  1. Copy {checkpoint_path} -> backend/models/pixeltruth_model.keras")
    print(f"  2. Copy {output_dir / 'confusion_matrix.png'} -> frontend/public/model-info/confusion_matrix.png")
    print(f"  3. Copy {output_dir / 'training_curves.png'} -> frontend/public/model-info/training_curves.png")
    print("  4. Update backend/app/config.py:")
    print("       - INPUT_SIZE = (128, 128)   <-- MUST change from (64, 64), or the backend")
    print("         will resize uploads to the wrong size and predictions will be meaningless.")
    print("       - TRAINING_METRICS with the REAL numbers printed above, AND update the")
    print("         'dataset' section to describe the 140k Real and Fake Faces dataset")
    print("         (140,000 images; real faces from Flickr/FFHQ, fake faces from StyleGAN)")
    print("         instead of CIFAKE -- do not leave the old CIFAKE description in place.")
    print("  5. Test against BOTH face photos and non-face photos/AI images before deploying --")
    print("     this model is trained ONLY on faces and may not generalize to other image types.")


if __name__ == "__main__":
    main()
