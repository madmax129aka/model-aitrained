# PixelTruth — AI Image Detector

PixelTruth detects whether an image is likely AI-generated or an authentic
photograph. It is powered by a **custom-trained convolutional neural
network** — trained from scratch on the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
(100,000 real vs. AI-generated images). The AI-probability score is based
solely on this model's own prediction.
**No third-party AI-detection API is used anywhere in this app.**

## Project structure

```
backend/
  app/
    main.py            FastAPI app: routes, startup model loading, static serving
    config.py           Paths, weights, thresholds, and the REAL reported training metrics
    model_service.py     Loads pixeltruth_model.keras, runs inference + saliency
    analyzer.py          Runs the CNN and builds the final verdict + heatmap
    image_utils.py       Image <-> base64, heatmap compositing helpers
    report.py            One-page PDF report generation (reportlab)
    schemas.py            Pydantic response models
  models/
    pixeltruth_model.keras   <-- the trained model (moved here from repo root)
  training/
    train_model.py       Standalone script to (re)train the CNN correctly (see below)
    README.md             How to get the dataset and run training
    requirements-training.txt
  requirements.txt

frontend/
  src/
    pages/               Analyzer, Model Info, Batch, History, About
    components/          Gauge, VerdictCard, HeatmapViewer, etc.
  public/
    model/               model.json + group1-shard1of1.bin (TF.js export, for reference)
    model-info/          confusion_matrix.png + training_curves.png
  package.json / vite.config.js / tailwind.config.js

run.sh                   Single command: install deps, build frontend, start server
```

## Model details

- Architecture: 3x [Conv2D → BatchNorm → MaxPooling2D] blocks (32→64→128
  filters) → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid).
- Input size: 64×64×3. The model has an **internal `Rescaling(1/255)` layer**,
  so the backend feeds raw 0–255 pixel values — it does **not** normalize
  again before calling `model.predict()`.
- Label mapping used during training: `0 = FAKE (AI-generated)`, `1 = REAL`.
  The model's sigmoid output is `P(REAL)`; the backend reports
  `AI probability = 1 - P(REAL)`.
- Trained on CIFAKE (100,000 images, 50,000 per class, 80/20 train/val split).

### Reported test-set metrics (20,000 held-out images, 10,000 per class)

| Metric | Value |
|---|---|
| Overall accuracy | **95.62%** |
| FAKE — precision / recall / F1 | 93.29% / 98.30% / 95.73% |
| REAL — precision / recall / F1 | 98.20% / 92.93% / 95.49% |

> Retrained via `backend/training/train_model.py` with extra training-only
> augmentation (brightness/contrast/JPEG-quality/noise jitter) to improve
> generalization to real-world phone photos. See
> `backend/training/README.md` for how this was produced.

These are the real, reported numbers (see `backend/app/config.py::TRAINING_METRICS`)
and are also displayed on the app's **Model Info** page alongside
`confusion_matrix.png` and `training_curves.png` as proof of training.

## How a verdict is computed

`POST /api/analyze` resizes the uploaded image to 64×64, runs
`pixeltruth_model.keras`, and inverts the label mapping (`AI probability =
1 - P(REAL)`) to produce a single 0–100 "AI probability" score. This score
is the model's own prediction — no other signal is blended in.

> Earlier versions of this project also combined in a frequency-domain
> (FFT) analysis and an EXIF metadata check at 20% weight each. Those have
> been removed; the score is now the CNN's output alone.

Final score → verdict: **≥65 → Likely AI-Generated**, **≤35 → Likely
Authentic**, otherwise **Uncertain**.

The explainability heatmap is a genuine **input-gradient saliency map** —
the gradient of the CNN's output with respect to the input pixels, computed
via `tf.GradientTape` directly against our own model's weights (not an
approximation and not from a third-party API).

## Retraining the model

`backend/training/train_model.py` is a standalone script for (re)training
`pixeltruth_model.keras` from scratch on CIFAKE. It builds the exact
architecture the backend expects, trains with `EarlyStopping` +
`ReduceLROnPlateau` + best-checkpoint saving, evaluates on the real 20,000
-image held-out test set, and adds extra training-only augmentation
(brightness/contrast/JPEG-quality/noise jitter) aimed at closing the
domain gap between CIFAKE's CIFAR-10-style REAL images and real-world phone
photos. **This must be run on a machine with internet access and ideally a
GPU (e.g. Google Colab)** — see `backend/training/README.md` for full
step-by-step instructions, including where to get the dataset and how to
wire a retrained model back into this app.

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
