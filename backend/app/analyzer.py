"""
Runs PixelTruth's AI-image-detection analysis using ONLY our custom-trained
CNN model. This does NOT call any third-party AI-detection API -- the
prediction and the saliency gradient both come from our own model's weights
running locally.

(Note: earlier versions of this project also blended in a frequency-domain
FFT signal and an EXIF metadata signal at 20% weight each. Those have been
removed -- the final AI-probability score is now the CNN's output alone.)
"""
import uuid

from PIL import Image

from app.config import THRESHOLD_AI, THRESHOLD_AUTHENTIC, WEIGHT_CNN
from app.image_utils import make_thumbnail, pil_to_data_uri, saliency_to_heatmap_overlay
from app.model_service import predict_with_saliency


def verdict_from_score(score: float) -> str:
    if score >= THRESHOLD_AI:
        return "Likely AI-Generated"
    if score <= THRESHOLD_AUTHENTIC:
        return "Likely Authentic"
    return "Uncertain"


def run_full_analysis(image: Image.Image, raw_bytes: bytes) -> dict:
    """
    Run the custom CNN model against a decoded PIL image and produce the
    full analysis result (verdict, score, saliency heatmap).

    `raw_bytes` is accepted for API-compatibility with callers but is no
    longer used now that the EXIF signal has been removed.
    """
    # --- Our custom CNN model (the only signal used for scoring) ---------------
    cnn_result = predict_with_saliency(image)
    cnn_score = cnn_result["ai_probability"] * 100.0

    final_score = cnn_score * WEIGHT_CNN
    verdict = verdict_from_score(final_score)

    # --- Visual artifacts --------------------------------------------------------
    thumbnail = make_thumbnail(image)
    heatmap_overlay = saliency_to_heatmap_overlay(cnn_result["saliency"], image)

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
    }
