# Training PixelTruth's CNN correctly

This folder contains `train_model.py`, a from-scratch training script for
PixelTruth's CNN on the CIFAKE dataset. It is **not** run by the deployed
app — it's a standalone script you run once (or whenever you want to
retrain) to (re)produce `backend/models/pixeltruth_model.keras`.

> **Why this exists:** if you train on CIFAKE with only the "textbook"
> RandomFlip/RandomRotation/RandomZoom augmentation, the model tends to
> overfit to what CIFAKE's REAL images specifically look like (they're
> sourced from CIFAR-10 — small, uniformly compressed, clean images) rather
> than learning "real photograph" in a way that generalizes to a modern
> phone photo. That's the most likely reason a real photo you upload gets
> scored as high AI-probability. This script adds extra augmentation
> (brightness/contrast/saturation jitter, JPEG-quality jitter, light sensor
> noise) **in the training data pipeline only** — never baked into the saved
> model — specifically to close that gap.

## 1. Get the CIFAKE dataset

CIFAKE is hosted on Kaggle: <https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images>
(100,000 images total: 50,000 REAL sourced from CIFAR-10, 50,000 FAKE
generated with Stable Diffusion — 20,000 of those 100,000 are held out as
a separate test set).

**On Google Colab** (recommended — free GPU, has internet access, which
this sandbox does not):

```python
import kagglehub
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print(path)  # this is your --data-dir
```

Or download the zip manually from Kaggle and unzip it. Either way, you need
a directory with this exact structure:

```
<data-dir>/
  train/
    REAL/   *.jpg
    FAKE/   *.jpg
  test/
    REAL/   *.jpg
    FAKE/   *.jpg
```

The folder names **must** be exactly `REAL` and `FAKE` (uppercase) — the
script derives the label mapping (`0=FAKE, 1=REAL`) from these folder names
and will refuse to continue if they don't match, since a silently wrong
label mapping is worse than a crash.

## 2. Install training dependencies

```bash
pip install -q -r requirements-training.txt
```

(These are separate from `backend/requirements.txt` — you don't need
matplotlib/scikit-learn/kagglehub on your production server.)

> **Do not** also install `requirements-tfjs.txt` in this same step/environment
> — it pins its own `tensorflow` version and will conflict with the
> tensorflow this step needs (and with Colab's preinstalled tensorflow),
> causing a `ResolutionImpossible` pip error. TF.js export is optional and
> covered separately in step 4 below.

## 3. Run training

```bash
python train_model.py \
  --data-dir /path/to/cifake \
  --epochs 30 \
  --output-dir ./training_output
```

This will:
1. Build the exact architecture the backend expects (documented in
   `backend/app/config.py` and `frontend/public/model/model.json`).
2. Train with `EarlyStopping` (stops once validation accuracy stops
   improving, patience=5) and `ReduceLROnPlateau` (halves the learning rate
   if validation loss plateaus) — this avoids both undertraining and wasting
   epochs overfitting.
3. Save the **best** epoch (by validation accuracy) via `ModelCheckpoint`,
   not just whatever the last epoch happened to be.
4. Evaluate on the real 20,000-image CIFAKE test set and print a full
   `classification_report` (accuracy, per-class precision/recall/F1).
5. Save `confusion_matrix.png` and `training_curves.png`.

Training 30 epochs over 80,000 training images at 64x64 typically takes
roughly 15–40 minutes on a Colab T4 GPU, depending on how early
`EarlyStopping` kicks in.

## 4. Wire the new model back into the app

```bash
cp training_output/pixeltruth_model.keras  ../models/pixeltruth_model.keras
cp training_output/confusion_matrix.png    ../../frontend/public/model-info/confusion_matrix.png
cp training_output/training_curves.png     ../../frontend/public/model-info/training_curves.png
```

Then **open `backend/app/config.py`** and replace the numbers inside
`TRAINING_METRICS` with whatever `train_model.py` actually printed for your
run (overall accuracy, per-class precision/recall/f1, test set size). Do
not leave the old numbers in place if you retrained — they must reflect the
model file that's actually being served.

If you also want an updated browser-side (TensorFlow.js) copy of the model,
do this in a **separate step** (a fresh Colab runtime, or a new virtualenv)
to avoid the dependency conflict mentioned above:

```bash
pip install -q -r requirements-tfjs.txt
python train_model.py --data-dir /path/to/cifake --epochs 30 --export-tfjs
```

Then copy the resulting `model.json` + `.bin` shard file(s) into
`frontend/public/model/`.

## 5. Optional: further improving real-world generalization

If, after retraining with the added augmentation, real-world photos are
still misclassified more than you'd like, the next most effective levers
(in rough order of effort) are:

1. **Mix in some of your own labeled real photos** as additional REAL
   training examples, so the model directly sees phone-camera-style images
   during training, not just CIFAR-10-style ones.
2. **Increase input resolution** (e.g. 96x96 or 128x128 instead of 64x64) —
   more detail can help the CNN pick up on genuine generation artifacts
   rather than dataset-specific quirks, at the cost of a larger/slower model.
3. **Try a slightly larger backbone** (e.g. add a fourth Conv block or more
   filters) if you have the compute budget — but only after step 1, since
   more capacity without more diverse data usually just overfits harder.
