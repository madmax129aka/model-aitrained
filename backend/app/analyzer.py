"""
Orchestrates the three PixelTruth detection signals into one combined result:

    a) Our custom-trained CNN model               -- weight 60%
    b) Frequency-domain / FFT artifact analysis    -- weight 20%
    c) Metadata / EXIF presence check              -- weight 20%

None of these signals call any third-party AI-detection API. Signal (a) is
our own model's weights running locally; signals (b) and (c) are plain numpy
/ exifread analysis with no network calls.
"""
import uuid

from PIL import Image

from app.config import THRESHOLD_AI, THRESHOLD_AUTHENTIC, WEIGHT_CNN, WEIGHT_EXIF, WEIGHT_FFT
from app.image_utils import (
    grayscale_array_to_image,
    make_thumbnail,
    pil_to_data_uri,
    saliency_to_heatmap_overlay,
)
from app.model_service import predict_with_saliency
from app.signals import exif_signal, fft_signal


def verdict_from_score(score: float) -> str:
    if score >= THRESHOLD_AI:
        return "Likely AI-Generated"
    if score <= THRESHOLD_AUTHENTIC:
        return "Likely Authentic"
    return "Uncertain"


def run_full_analysis(image: Image.Image, raw_bytes: bytes) -> dict:
    """
    Run all three signals against a decoded PIL image (+ original raw bytes
    for EXIF, which must come from the original file, not a re-encoded copy).
    """
    # --- Signal A: our custom CNN ------------------------------------------------
    cnn_result = predict_with_saliency(image)
    cnn_score = cnn_result["ai_probability"] * 100.0

    # --- Signal B: FFT frequency-domain analysis --------------------------------
    fft_result = fft_signal.analyze(image)
    fft_score = fft_result["score"]

    # --- Signal C: EXIF metadata check ------------------------------------------
    exif_result = exif_signal.analyze(raw_bytes)
    exif_score = exif_result["score"]

    final_score = (
        cnn_score * WEIGHT_CNN + fft_score * WEIGHT_FFT + exif_score * WEIGHT_EXIF
    )
    verdict = verdict_from_score(final_score)

    # --- Visual artifacts --------------------------------------------------------
    thumbnail = make_thumbnail(image)
    heatmap_overlay = saliency_to_heatmap_overlay(cnn_result["saliency"], image)
    fft_spectrum_img = grayscale_array_to_image(fft_result["spectrum"])

    cnn_explanation = (
        f"Our custom-trained CNN (trained from scratch on the CIFAKE dataset) "
        f"estimates a {cnn_score:.1f}% probability this image is AI-generated. "
        f"The model's raw output favors the '{cnn_result['predicted_label']}' class."
    )

    signals = [
        {
            "name": "Custom CNN Model",
            "weight": WEIGHT_CNN,
            "score": round(cnn_score, 2),
            "explanation": cnn_explanation,
            "details": {
                "predicted_label": cnn_result["predicted_label"],
                "raw_sigmoid_output": round(cnn_result["raw_output"], 4),
            },
        },
        {
            "name": "Frequency Domain (FFT) Analysis",
            "weight": WEIGHT_FFT,
            "score": round(fft_score, 2),
            "explanation": fft_result["explanation"],
            "details": {"anomaly_ratio": round(fft_result["anomaly_ratio"], 4)},
        },
        {
            "name": "Metadata / EXIF Check",
            "weight": WEIGHT_EXIF,
            "score": round(exif_score, 2),
            "explanation": exif_result["explanation"],
            "details": {
                "has_exif": exif_result["has_exif"],
                "has_camera_info": exif_result["has_camera_info"],
                "tags": exif_result["tags"],
            },
        },
    ]

    return {
        "analysis_id": str(uuid.uuid4()),
        "verdict": verdict,
        "final_score": round(final_score, 2),
        "signals": signals,
        "model_predicted_label": cnn_result["predicted_label"],
        "model_raw_output": round(cnn_result["raw_output"], 4),
        "original_image": pil_to_data_uri(thumbnail),
        "heatmap_image": pil_to_data_uri(make_thumbnail(heatmap_overlay)),
        "fft_spectrum_image": pil_to_data_uri(make_thumbnail(fft_spectrum_img)),
    }
