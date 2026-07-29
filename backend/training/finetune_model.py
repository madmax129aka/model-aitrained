"""
PixelTruth -- fine-tuning script for real-world generalization.

WHY THIS EXISTS
----------------
The base model (trained by train_model.py on CIFAKE alone) was found to be
biased: testing showed that EVERY real photo scored as "Likely AI-Generated"
before a preprocessing experiment, and after that experiment was reverted,
blurring the input flipped the bias so that EVERY image -- including an
actual AI-generated test image -- scored as "Likely Authentic". This proves
the model latched onto a shortcut signal (image sharpness/blur level) rather
than genuine AI-generation artifacts. This happened because CIFAKE's source
images are natively only 32x32 pixels (they mirror CIFAR-10's resolution
exactly), so the model never saw real-world-resolution sharp photos or
real-world AI-generated images during training.

Mixing a small number of your own photos into the full 100k-image CIFAKE
training set from scratch would not work -- your images would be a rounding
error, contributing almost nothing to the trained weights. Instead, this
script FINE-TUNES the already-trained model: it loads the existing
pixeltruth_model.keras, freezes nothing (all layers stay trainable, but with
a much lower learning rate and few epochs), and continues training for a
short time using ONLY your custom real-world dataset. This nudges the
model's decision boundary toward real-world images without needing anywhere
close to 100,000 examples.

HOW MANY IMAGES YOU NEED
--------------------------
    Minimum viable : 150-200 REAL + 150-200 FAKE (roughly balanced)
    Good           : 400-600 REAL + 400-600 FAKE
    Ideal          : 1,000+  REAL + 1,000+  FAKE

Diversity matters more than raw count:
  REAL: multiple different cameras/phones, varied lighting, varied subjects
        (people, objects, scenery, groups), some screenshots, some heavily
        compressed/re-uploaded images, not just one device's photos.
  FAKE: multiple different AI generators (Gemini, Midjourney, DALL-E, Stable
        Diffusion, etc.), varied styles and subjects, not just one tool.

EXPECTED FOLDER STRUCTURE
---------------------------
    <custom-data-dir>/
      REAL/   *.jpg / *.png / *.jpeg  (your real photos)
      FAKE/   *.jpg / *.png / *.jpeg  (your AI-generated images)

Folder names must be exactly "REAL" and "FAKE" (uppercase), same convention
as CIFAKE, so the label mapping (0=FAKE, 1=REAL) stays consistent with
everything else in this project.

USAGE
------
    python finetune_model.py \
      --base-model ../models/pixeltruth_model.keras \
      --custom-data-dir /path/to/your/real_and_fake_images \
      --epochs 8 \
      --output-dir ./finetune_output

This produces a NEW pixeltruth_model.keras (fine-tuned) plus an updated
confusion_matrix.png / training_curves.png and a classification report --
same next-steps workflow as train_model.py (copy files into the app,
update TRAINING_METRICS with the real printed numbers).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG_SIZE = (64, 64)
BATCH_SIZE = 16  # small custom datasets -> small batches
SEED = 42
EXPECTED_CLASS_NAMES = ["FAKE", "REAL"]

# How much to shrink the batch further for validation, since custom
# datasets are much smaller than CIFAKE's 100k images.
MIN_VAL_SPLIT = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the PixelTruth CNN on a custom real-world dataset."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to the existing pixeltruth_model.keras to fine-tune "
        "(e.g. ../models/pixeltruth_model.keras).",
    )
    parser.add_argument(
        "--custom-data-dir",
        type=str,
        required=True,
        help="Path to your custom dataset root, containing REAL/ and FAKE/ "
        "subfolders with your own real photos and AI-generated images.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Fine-tuning epochs. Keep this LOW (5-10) -- fine-tuning for too "
        "long on a small custom dataset will overfit to it and can destroy "
        "the general knowledge learned from the original 100k-image CIFAKE "
        "training. EarlyStopping is still enabled as a safety net.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Fine-tuning learning rate. MUST be much lower than the original "
        "1e-3 training rate, or fine-tuning will overwrite the model's "
        "existing knowledge instead of gently nudging it.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of your custom dataset held out for validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Where to write the fine-tuned pixeltruth_model.keras, "
        "confusion_matrix.png, training_curves.png.",
    )
    return parser.parse_args()


def load_custom_dataset(custom_data_dir: Path, val_split: float):
    if not custom_data_dir.exists():
        sys.exit(f"ERROR: custom data directory '{custom_data_dir}' does not exist.")

    real_dir = custom_data_dir / "REAL"
    fake_dir = custom_data_dir / "FAKE"
    if not real_dir.exists() or not fake_dir.exists():
        sys.exit(
            f"ERROR: expected '{real_dir}' and '{fake_dir}' to exist. "
            "Your custom dataset must have exactly two subfolders named "
            "'REAL' and 'FAKE' (uppercase)."
        )

    real_count = sum(1 for _ in real_dir.glob("*") if _.is_file())
    fake_count = sum(1 for _ in fake_dir.glob("*") if _.is_file())
    print(f"Found {real_count} REAL images and {fake_count} FAKE images in {custom_data_dir}")

    if real_count < 50 or fake_count < 50:
        print(
            "\nWARNING: You have fewer than 50 images in one or both classes. "
            "Fine-tuning with this little data is unlikely to meaningfully "
            "improve generalization, and may make results worse/unstable. "
            "See this script's docstring for recommended dataset sizes "
            "(150-200 minimum per class, 1,000+ ideal).\n"
        )

    if abs(real_count - fake_count) > max(real_count, fake_count) * 0.5:
        print(
            f"\nWARNING: your REAL ({real_count}) and FAKE ({fake_count}) counts are "
            "very imbalanced. Try to keep them roughly balanced (within 2x of "
            "each other) so the model doesn't just learn to favor whichever "
            "class has more examples.\n"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        custom_data_dir,
        validation_split=val_split,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        custom_data_dir,
        validation_split=val_split,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    class_names = train_ds.class_names
    if class_names != EXPECTED_CLASS_NAMES:
        sys.exit(
            f"ERROR: expected class folders {EXPECTED_CLASS_NAMES} (giving label "
            f"mapping 0=FAKE, 1=REAL) but found {class_names}. Rename your "
            "dataset's folders to exactly 'FAKE' and 'REAL'."
        )
    print(f"Class mapping confirmed: {class_names} -> label 0={class_names[0]}, label 1={class_names[1]}")

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def evaluate_and_plot(model: tf.keras.Model, val_ds: tf.data.Dataset, history, output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_pred_prob = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred_prob.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true).astype(int)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n" + "=" * 70)
    print("Fine-tuned model evaluation on your custom validation split")
    print("(NOTE: this is a small custom validation set, not the full 20,000")
    print(" image CIFAKE test set -- treat these numbers as a rough signal,")
    print(" not a precise accuracy claim.)")
    print("=" * 70)
    report = classification_report(
        y_true, y_pred, target_names=EXPECTED_CLASS_NAMES, digits=4, zero_division=0
    )
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(EXPECTED_CLASS_NAMES)
    ax.set_yticklabels(EXPECTED_CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PixelTruth CNN (fine-tuned) -- Confusion Matrix\n(custom validation split)")
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
    axes[0].set_title("Accuracy (fine-tuning)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss (fine-tuning)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    curves_path = output_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Saved {curves_path}")


def main():
    args = parse_args()
    base_model_path = Path(args.base_model)
    custom_data_dir = Path(args.custom_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_model_path.exists():
        sys.exit(f"ERROR: base model '{base_model_path}' does not exist.")

    tf.random.set_seed(SEED)

    print(f"Loading base model from {base_model_path} ...")
    model = tf.keras.models.load_model(str(base_model_path))
    print(
        f"Loaded. Re-compiling with a low fine-tuning learning rate "
        f"({args.learning_rate}) -- much lower than the original training "
        f"rate, so we nudge the existing weights rather than overwrite them."
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Loading your custom dataset from {custom_data_dir} ...")
    train_ds, val_ds = load_custom_dataset(custom_data_dir, args.val_split)

    checkpoint_path = output_dir / "pixeltruth_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print(f"Fine-tuning for up to {args.epochs} epochs ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(checkpoint_path)
    print(f"\nSaved fine-tuned model to {checkpoint_path}")

    evaluate_and_plot(model, val_ds, history, output_dir)

    print("\nDone. Next steps:")
    print(f"  1. Test this fine-tuned model against several of your OWN real photos and")
    print(f"     AI-generated images BEFORE replacing the deployed model -- confirm it")
    print(f"     actually improved (not just memorized your validation split).")
    print(f"  2. If it looks good: copy {checkpoint_path} -> backend/models/pixeltruth_model.keras")
    print(f"  3. Copy {output_dir / 'confusion_matrix.png'} -> frontend/public/model-info/confusion_matrix.png")
    print(f"  4. Copy {output_dir / 'training_curves.png'} -> frontend/public/model-info/training_curves.png")
    print("  5. Update backend/app/config.py::TRAINING_METRICS. Since this evaluation is on")
    print("     your own small custom validation split (not the full 20,000-image CIFAKE")
    print("     test set), consider re-running the ORIGINAL evaluation from train_model.py")
    print("     against the full CIFAKE test set with this fine-tuned model to get numbers")
    print("     comparable to before, OR clearly label the new metrics as coming from your")
    print("     custom validation set so the Model Info page isn't misleading.")


if __name__ == "__main__":
    main()
