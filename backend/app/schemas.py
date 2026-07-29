"""Pydantic response models for the PixelTruth API."""
from typing import Dict, List, Optional

from pydantic import BaseModel


class SignalBreakdown(BaseModel):
    name: str
    weight: float
    score: float  # 0-100, this signal's own "AI likelihood"
    explanation: str
    details: Dict[str, object] = {}


class AnalyzeResponse(BaseModel):
    verdict: str  # "Likely AI-Generated" | "Likely Authentic" | "Uncertain"
    final_score: float  # 0-100 combined AI-probability score (weighted across 3 signals)
    signals: List[SignalBreakdown]
    heatmap_image: Optional[str] = None  # data URI
    original_image: Optional[str] = None  # data URI (thumbnail)
    fft_spectrum_image: Optional[str] = None  # data URI
    model_predicted_label: str
    model_raw_output: float
    analysis_id: str


class HealthResponse(BaseModel):
    status: str
    custom_model_loaded: bool
    model_load_error: Optional[str] = None
    message: str


class ModelInfoResponse(BaseModel):
    dataset: Dict[str, object]
    test_set: Dict[str, object]
    overall_accuracy: float
    per_class: Dict[str, Dict[str, float]]
    architecture_summary: List[str]
    training_details: Dict[str, object]
    confusion_matrix_url: str
    training_curves_url: str
    model_source: str
