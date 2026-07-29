# PixelTruth — AI Image Detector

PixelTruth detects whether an image is likely AI-generated or an authentic
photograph. It is powered by a **custom-trained convolutional neural
network** — trained from scratch on the [140k Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
(real human face photographs vs. StyleGAN-generated fake faces; 140,000
images total), combined with two lightweight supporting signals
(frequency-domain / FFT analysis and EXIF metadata checks).
**All inference happens server-side in the FastAPI backend — there is no
browser-based TensorFlow.js, and no third-party AI-detection API is used
anywhere in this app.**

## Project structure

```
backend/
  app/
    main.py            FastAPI app: routes, startup model loading, static serving
    config.py           Paths, weights, thresholds, and the REAL reported training metrics
    model_service.py     Loads pixeltruth_model.keras, runs inference + saliency
    analyzer.py          Combines the 3 signals into a final verdict
    image_utils.py       Image <-> base64, heatmap compositing helpers
    report.py            One-page PDF report generation (reportlab)
    schemas.py            Pydantic response models
    signals/
      fft_signal.py       Frequency-domain (FFT) artifact analysis (numpy only)
      exif_signal.py       EXIF metadata presence check (exifread)
  models/
    pixeltruth_model.keras   <-- the trained model (128x128 input, 4-conv-block CNN)
  training/
    train_model_faces.py Standalone script to (re)train the current model (140k Faces, 128x128)
    train_model.py        Earlier CIFAKE-based (64x64) training script, kept for reference
    finetune_model.py     Fine-tune an existing model on a small custom dataset
    README.md             How to get each dataset and run training
    requirements-training.txt
  requirements.txt

frontend/
  src/
    pages/               Analyzer, Model Info, Batch, History, About
    components/          Gauge, VerdictCard, HeatmapViewer, FFTViewer, MetadataBadges, etc.
  public/
    model-info/          confusion_matrix.png + training_curves.png
  package.json / vite.config.js / tailwind.config.js

run.sh                   Single command: install deps, build frontend, start server
```

## Model details

- Architecture: 4x [Conv2D → BatchNorm → MaxPooling2D] blocks (32→64→128→256
  filters) → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid).
- Input size: 128×128×3. The model has an **internal `Rescaling(1/255)`
  layer**, so the backend feeds raw 0–255 pixel values — it does **not**
  normalize again before calling `model.predict()`.
- Label mapping used during training: `0 = FAKE (AI-generated)`, `1 = REAL`.
  The model's sigmoid output is `P(REAL)`; the backend reports
  `AI probability = 1 - P(REAL)`.
- Trained on the **140k Real and Fake Faces** dataset: 100,000 training /
  20,000 validation / 20,000 test images, perfectly balanced between real
  photographed faces (FFHQ/Flickr) and StyleGAN-generated fake faces.

> **This model is faces-only.** Detection is most accurate on photos
> containing a clear human face — that's what the training data consists
> of. Non-face images (landscapes, objects, illustrations) can still be
> analyzed, but the CNN signal was not trained on that kind of content.

### Reported test-set metrics (20,000 held-out images, 10,000 per class)

| Metric | Value |
|---|---|
| Overall accuracy | **94%** |
| FAKE — precision / recall / F1 | 96% / 91% / 93% |
| REAL — precision / recall / F1 | 91% / 96% / 94% |

These are the real, reported numbers (see `backend/app/config.py::TRAINING_METRICS`)
and are also displayed on the app's **Model Info** page alongside
`confusion_matrix.png` and `training_curves.png` as proof of training.

## How a verdict is computed

`POST /api/analyze` combines three signals into one weighted score (0–100 =
"AI probability"):

| Signal | Weight | What it does |
|---|---|---|
| Custom CNN model | 60% | Resizes to 128×128, runs `pixeltruth_model.keras`, inverts label mapping |
| FFT frequency analysis | 20% | Looks for periodic upsampling artifacts in the frequency spectrum (numpy) |
| EXIF metadata check | 20% | Checks for camera make/model/EXIF tags (exifread) |

Final score → verdict: **≥65 → Likely AI-Generated**, **≤35 → Likely
Authentic**, otherwise **Uncertain**.

The explainability heatmap is a genuine **input-gradient saliency map** —
the gradient of the CNN's output with respect to the input pixels, computed
via `tf.GradientTape` directly against our own model's weights (not an
approximation and not from a third-party API). To keep the heatmap
meaningful even for very confident predictions, gradients are computed
against a pre-sigmoid "logits" sub-model rather than the final probability
output (see `model_service.py::_build_logits_model` for why).

## Retraining the model

`backend/training/train_model_faces.py` is the standalone script used to
(re)produce the currently deployed model, training from scratch on the
140k Real and Fake Faces dataset at 128×128 input. See
`backend/training/README.md` for full step-by-step instructions (getting
the dataset, running training, and wiring a retrained model back into this
app — note that changing input resolution requires also updating
`backend/app/config.py::INPUT_SIZE`).

Two earlier scripts are also kept for reference:
- `train_model.py` — the original CIFAKE-based (64×64) training script.
  Testing found this approach caused the model to key off image
  sharpness/blur rather than genuine AI-generation artifacts (CIFAKE's
  source images are natively only 32×32 pixels), which is why the project
  moved to the current faces-based dataset at higher resolution.
- `finetune_model.py` — fine-tunes an already-trained model on a small
  custom dataset (your own real photos + AI-generated images) without
  needing to retrain from scratch.

**Training must be run on a machine with internet access and ideally a GPU
(e.g. Google Colab)** — this project's training scripts cannot be executed
inside this sandboxed development environment.

## Running locally / deploying (e.g. on Replit)

```bash
./run.sh
```

This single command:
1. Installs backend Python deps (`pip install -r backend/requirements.txt`).
2. Installs frontend Node deps (`npm install`) and runs `npm run build`.
3. Starts FastAPI (`uvicorn`) bound to `0.0.0.0:${PORT:-8000}`, which serves
   both the JSON API (`/api/*`) and the built frontend (`frontend/dist`) from
   the same process/port.

On startup, the server logs clearly whether the custom model loaded
successfully — check `GET /api/health` at any time for the same status.

> **Note on this environment:** This project's `run.sh` was written and
> syntax-checked in a sandboxed environment without outbound access to PyPI
> or the npm registry, so the actual `pip install` / `npm install` / build
> steps could not be executed and verified end-to-end here. They are
> expected to work on a machine with normal internet access (e.g. Replit).
