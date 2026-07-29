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

## 5. Fixing systematic real-world bias: fine-tuning on your own images

**Confirmed issue:** testing showed the base model relies on image
sharpness/blur as a shortcut rather than learning genuine AI-generation
artifacts (CIFAKE's source images are natively only 32x32 pixels, so the
model never saw real-world-resolution photos of either class during
training). This caused every real photo to score as "Likely AI-Generated"
in one test, and — after an experimental blur-matching preprocessing
change — every image (including an actual AI-generated one) to score as
"Likely Authentic" instead. Both results point to the same root cause: the
model needs to see real-world-resolution examples of BOTH classes during
training, not just CIFAKE's 32x32-sourced images.

**Do not** just mix a few hundred of your own images into the full 100k
CIFAKE set and retrain from scratch — your images would be statistically
drowned out and have almost no effect. Instead, use
**`finetune_model.py`**, which loads the already-trained model and
continues training it briefly, at a much lower learning rate, using ONLY
your custom dataset. This nudges the model's decision boundary without
needing anywhere close to 100,000 examples.

### How many images you need

| Tier | REAL images | FAKE (AI-generated) images |
|---|---|---|
| Minimum viable | 150–200 | 150–200 |
| Good | 400–600 | 400–600 |
| Ideal | 1,000+ | 1,000+ |

Diversity matters more than raw count:
- **REAL**: multiple different cameras/phones, varied lighting, varied
  subjects (people, objects, scenery, groups), some screenshots, some
  heavily compressed/re-uploaded images — not all from one device.
- **FAKE**: multiple different AI generators (Gemini, Midjourney, DALL-E,
  Stable Diffusion, etc.), varied styles and subjects — not just one tool.

### Prepare your dataset

```
<custom-data-dir>/
  REAL/   *.jpg / *.png   (your real photos)
  FAKE/   *.jpg / *.png   (your AI-generated images)
```

Folder names must be exactly `REAL` and `FAKE` (uppercase), same convention
as CIFAKE.

### Run fine-tuning

```bash
python finetune_model.py \
  --base-model ../models/pixeltruth_model.keras \
  --custom-data-dir /path/to/your/real_and_fake_images \
  --epochs 8 \
  --output-dir ./finetune_output
```

Key differences from `train_model.py`:
- Loads the **existing** trained model instead of building a new one from scratch.
- Uses a much lower learning rate (`1e-5` by default vs. the original `1e-3`)
  so it gently nudges the existing weights instead of overwriting them.
- Keeps epochs low (default 8) — fine-tuning too long on a small custom
  dataset will overfit to it and can destroy the general knowledge learned
  from the original 100k-image CIFAKE training.
- Warns if your dataset is too small (<50 per class) or too imbalanced
  (>2x difference between REAL and FAKE counts).

### After fine-tuning: verify before deploying

**Test the fine-tuned model against several of your own real photos and
AI-generated images that were NOT part of the fine-tuning dataset**, before
replacing the deployed model. A model can look great on its own validation
split while still being biased on genuinely new images — the whole point
of fine-tuning is to fix real-world generalization, so real-world testing
is the only way to confirm it worked.

If it looks good, follow the same wiring steps as section 4 above, using
`finetune_output/` instead of `training_output/`. Since the fine-tuning
evaluation runs against your own small custom validation split (not the
full 20,000-image CIFAKE test set), either re-run the original
`train_model.py` evaluation against the full CIFAKE test set with this
fine-tuned model to get comparable numbers, or clearly label the new
metrics in `TRAINING_METRICS` as coming from your custom validation set so
the Model Info page isn't misleading.

## 5b. Alternative: retrain from scratch on 140k Real and Fake Faces

Instead of fine-tuning, `train_model_faces.py` trains a brand-new model
from scratch on the [140k Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
(real photographed faces from Flickr/FFHQ vs. StyleGAN-generated fake
faces), and bumps the input resolution from 64x64 to **128x128** to
preserve more fine detail.

**Trade-off to understand first:** this dataset is faces-only. A model
trained on it will likely perform worse on non-face AI-generated images
(landscapes, objects, illustrations) than the CIFAKE-based model did on
those. Only take this path if detecting fake/AI faces specifically is your
priority, or if you plan to later combine this with CIFAKE (see the
"Option 3" idea below).

### 1. Get the dataset (Colab)

```python
!pip install -q tensorflow scikit-learn matplotlib kaggle
!kaggle datasets download -d xhlulu/140k-real-and-fake-faces
!unzip -q 140k-real-and-fake-faces.zip -d /content/data_faces
```

(Requires a `kaggle.json` API token uploaded to Colab -- same as
`kagglehub` needed in section 1, but this dataset is downloaded via the
`kaggle` CLI instead since it's not available through `kagglehub` directly.)

### 2. Run training

```bash
python train_model_faces.py \
  --data-dir /content/data_faces \
  --epochs 20 \
  --output-dir ./output_faces
```

The script **auto-detects** this dataset's nested folder layout
(`real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/`) by searching
under `--data-dir`. If it can't find that layout, it prints every directory
it found under `--data-dir` -- paste that output back if you hit this.

### 3. Wire the new model in -- MORE steps than usual this time

Because this changes the input resolution, there are two extra steps
beyond the usual "copy the model file" workflow:

1. Copy `output_faces/pixeltruth_model.keras` -> `backend/models/pixeltruth_model.keras`
2. Copy `output_faces/confusion_matrix.png` -> `frontend/public/model-info/confusion_matrix.png`
3. Copy `output_faces/training_curves.png` -> `frontend/public/model-info/training_curves.png`
4. **Open `backend/app/config.py` and change `INPUT_SIZE = (64, 64)` to
   `INPUT_SIZE = (128, 128)`.** This is not automatic -- if you skip this,
   the backend will resize uploads to the wrong size and every prediction
   will be meaningless (the model expects 128x128 input, not 64x64).
5. Update `TRAINING_METRICS` in the same file with the real numbers the
   script printed, AND update the `dataset` section to describe the 140k
   Real and Fake Faces dataset instead of CIFAKE -- don't leave the old
   CIFAKE description in place, it would misrepresent what the deployed
   model was actually trained on.
6. Test against **both** face photos and non-face images/AI content before
   fully switching over, given the trade-off noted above.

### Future option: combine CIFAKE + 140k Faces

For a more robust model that handles both general objects and faces, the
two datasets could be combined into one training run (doubling the data
pipeline complexity, since their folder layouts and file namings differ).
This is not implemented in either script yet -- if you want to pursue this,
the folder-merging approach would be to copy/symlink both datasets' REAL
and FAKE images into one combined directory structure before training with
a version of `train_model.py`'s pipeline adjusted for the larger, mixed
dataset size.

## 6. Other options for further improving real-world generalization

1. **Increase input resolution** (e.g. 96x96 or 128x128 instead of 64x64) —
   more detail can help the CNN pick up on genuine generation artifacts
   rather than dataset-specific quirks, at the cost of a larger/slower model.
2. **Try a slightly larger backbone** (e.g. add a fourth Conv block or more
   filters) if you have the compute budget — but only after fine-tuning on
   real-world data, since more capacity without more diverse data usually
   just overfits harder.
