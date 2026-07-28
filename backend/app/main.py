"""
PixelTruth backend -- FastAPI application.

PixelTruth is an AI image detector powered by a CUSTOM-TRAINED convolutional
neural network (pixeltruth_model.keras), trained from scratch by the project
author on the CIFAKE dataset (100,000 real vs. AI-generated images). This
backend does NOT call any third-party AI-detection API at any point --
every prediction comes from local weights loaded once at startup, and the
explainability heatmap is a genuine input-gradient saliency map computed via
backpropagation through our own model.

Run directly with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
or via the repo-root `run.sh` script, which also builds and serves the
frontend from the same process/port.
"""
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import model_service
from app.analyzer import run_full_analysis
from app.config import (
    FRONTEND_DIST_DIR,
    MODEL_INFO_ASSETS_DIR,
    MODEL_PATH,
    TRAINING_METRICS,
)
from app.image_utils import load_image_rgb
from app.report import build_report_pdf
from app.schemas import AnalyzeResponse, HealthResponse, ModelInfoResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pixeltruth")

# In-memory cache of the most recent analysis results, keyed by analysis_id,
# so /api/report can regenerate a PDF without re-uploading the image.
_analysis_cache: dict[str, dict] = {}
_MAX_CACHE_ENTRIES = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 72)
    logger.info("PixelTruth backend starting up...")
    logger.info("Loading CUSTOM-TRAINED CNN model (no external AI API): %s", MODEL_PATH)
    model_service.load_model()
    if model_service.is_loaded():
        logger.info("STARTUP STATUS: Custom model loaded successfully. Ready to serve.")
    else:
        logger.error(
            "STARTUP STATUS: Custom model FAILED to load (%s). "
            "The /api/analyze endpoint will return errors until this is fixed.",
            model_service.load_error(),
        )
    logger.info("=" * 72)
    yield
    logger.info("PixelTruth backend shutting down.")


app = FastAPI(title="PixelTruth API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _remember_analysis(result: dict) -> None:
    if len(_analysis_cache) >= _MAX_CACHE_ENTRIES:
        # Drop the oldest entry (dict preserves insertion order in Py3.7+).
        _analysis_cache.pop(next(iter(_analysis_cache)))
    _analysis_cache[result["analysis_id"]] = result


@app.get("/api/health", response_model=HealthResponse)
async def health():
    loaded = model_service.is_loaded()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        custom_model_loaded=loaded,
        model_load_error=model_service.load_error(),
        message=(
            "Custom-trained PixelTruth CNN loaded successfully."
            if loaded
            else "Custom-trained PixelTruth CNN failed to load. See model_load_error."
        ),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        dataset=TRAINING_METRICS["dataset"],
        test_set=TRAINING_METRICS["test_set"],
        overall_accuracy=TRAINING_METRICS["overall_accuracy"],
        per_class=TRAINING_METRICS["per_class"],
        architecture_summary=TRAINING_METRICS["architecture_summary"],
        training_details=TRAINING_METRICS["training_details"],
        confusion_matrix_url="/model-info/confusion_matrix.png",
        training_curves_url="/model-info/training_curves.png",
        model_source=(
            "Custom-trained CNN (trained from scratch by the project author on the "
            "CIFAKE dataset). No third-party AI-detection API is used."
        ),
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail=f"Custom model is not loaded: {model_service.load_error()}",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = load_image_rgb(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    try:
        result = run_full_analysis(image, raw_bytes)
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    _remember_analysis(result)
    return AnalyzeResponse(**result)


@app.post("/api/report")
async def report(
    file: Optional[UploadFile] = File(None),
    analysis_id: Optional[str] = Form(None),
):
    """
    Generate a one-page PDF report.

    Preferred usage: pass `analysis_id` from a previous /api/analyze call to
    reuse its cached result. Alternatively, pass a fresh `file` upload to
    run analysis + report generation in one call.
    """
    result = None
    if analysis_id and analysis_id in _analysis_cache:
        result = _analysis_cache[analysis_id]
    elif file is not None:
        if not model_service.is_loaded():
            raise HTTPException(
                status_code=503,
                detail=f"Custom model is not loaded: {model_service.load_error()}",
            )
        raw_bytes = await file.read()
        try:
            image = load_image_rgb(raw_bytes)
            result = run_full_analysis(image, raw_bytes)
            _remember_analysis(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not analyze image: {exc}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either an analysis_id from a prior /api/analyze call, or a file upload.",
        )

    pdf_bytes = build_report_pdf(result)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="pixeltruth_report_{result["analysis_id"][:8]}.pdf"'
        },
    )


# ---------------------------------------------------------------------------
# Static file serving: confusion_matrix.png / training_curves.png, and (in
# production) the built React frontend, all from this same FastAPI process.
# ---------------------------------------------------------------------------
if MODEL_INFO_ASSETS_DIR.exists():
    app.mount("/model-info", StaticFiles(directory=str(MODEL_INFO_ASSETS_DIR)), name="model-info")

if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def root_placeholder():
        return JSONResponse(
            {
                "message": (
                    "PixelTruth API is running, but the frontend has not been built yet. "
                    "Run the frontend build (e.g. via run.sh) to serve the full app here."
                ),
                "custom_model_loaded": model_service.is_loaded(),
            }
        )
