# PixelTruth — AI Image Detector

PixelTruth detects whether an image is likely AI-generated or an authentic
photograph. It is powered by a **custom-trained convolutional neural
network** — trained from scratch on the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
(100,000 real vs. AI-generated images) — combined with two lightweight
supporting signals (frequency-domain / FFT analysis and EXIF metadata
checks). **No third-party AI-detection API is used anywhere in this app.**

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
    pixeltruth_model.keras   <-- the trained model (moved here from repo root)
  requirements.txt

frontend/
  src/
    pages/               Analyzer, Model Info, Batch, History, About
    components/          Gauge, VerdictCard, HeatmapViewer, FFTViewer, etc.
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
| Overall accuracy | **91%** |
| FAKE — precision / recall / F1 | 86% / 97% / 91% |
| REAL — precision / recall / F1 | 97% / 84% / 90% |

These are the real, reported numbers (see `backend/app/config.py::TRAINING_METRICS`)
and are also displayed on the app's **Model Info** page alongside
`confusion_matrix.png` and `training_curves.png` as proof of training.

## How a verdict is computed

`POST /api/analyze` combines three signals into one weighted score (0–100 =
"AI probability"):

| Signal | Weight | What it does |
|---|---|---|
| Custom CNN model | 60% | Resizes to 64×64, runs `pixeltruth_model.keras`, inverts label mapping |
| FFT frequency analysis | 20% | Looks for periodic upsampling artifacts in the frequency spectrum (numpy) |
| EXIF metadata check | 20% | Checks for camera make/model/EXIF tags (exifread) |

Final score → verdict: **≥65 → Likely AI-Generated**, **≤35 → Likely
Authentic**, otherwise **Uncertain**.

The explainability heatmap is a genuine **input-gradient saliency map** —
the gradient of the CNN's output with respect to the input pixels, computed
via `tf.GradientTape` directly against our own model's weights (not an
approximation and not from a third-party API).

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
