"""
Frequency-domain (FFT) analysis signal.

AI image generators (especially GAN/diffusion upsampling pipelines) tend to
leave periodic checkerboard / grid artifacts in the frequency domain that
real camera photos generally don't exhibit, because of repeated
transposed-convolution or pixel-shuffle upsampling steps. This module scores
an image purely from its 2D FFT magnitude spectrum -- no ML model involved,
just numpy signal processing.

This is a heuristic, secondary signal (weighted 20% in the final score) --
it is NOT a substitute for the trained CNN, but it captures a class of
artifacts the CNN might not focus on and gives reviewers something concrete
and visual (the spectrum image) to inspect.
"""
import numpy as np
from PIL import Image


def _to_grayscale_array(image: Image.Image, max_dim: int = 512) -> np.ndarray:
    """Downscale (if needed) and convert to a float32 grayscale array."""
    img = image.convert("L")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def compute_fft_magnitude(image: Image.Image) -> np.ndarray:
    """Return the log-scaled, shifted FFT magnitude spectrum (2D array)."""
    gray = _to_grayscale_array(image)
    # Hann window reduces edge-of-image spectral leakage before FFT.
    win_y = np.hanning(gray.shape[0])
    win_x = np.hanning(gray.shape[1])
    window = np.outer(win_y, win_x)
    windowed = gray * window

    spectrum = np.fft.fft2(windowed)
    spectrum_shifted = np.fft.fftshift(spectrum)
    magnitude = np.abs(spectrum_shifted)
    log_magnitude = np.log1p(magnitude)
    return log_magnitude


def _radial_profile(magnitude: np.ndarray) -> np.ndarray:
    """Average spectral energy over concentric rings, from center to edge."""
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    max_r = r.max()
    ring_sum = np.bincount(r.ravel(), weights=magnitude.ravel(), minlength=max_r + 1)
    ring_count = np.bincount(r.ravel(), minlength=max_r + 1)
    ring_count[ring_count == 0] = 1
    return ring_sum / ring_count


def analyze(image: Image.Image) -> dict:
    """
    Analyze the image's frequency spectrum for periodic upsampling artifacts.

    Heuristic: real photographic content tends to have a fairly smooth,
    monotonically decaying radial power spectrum (natural images follow an
    approximate 1/f^2 power law). AI-generated / upsampled images often show
    sharp peaks or a "bump" at high or mid frequencies caused by repeated
    strided/transposed convolutions, which show up as anomalies (local
    spikes) relative to a smoothed version of the same radial profile.

    Returns a dict with:
        score        float in [0, 100] -- "AI-likelihood" contribution from this signal
        spectrum     np.ndarray        -- log magnitude spectrum (for visualization)
        explanation  str               -- plain-English summary
        anomaly_ratio float            -- raw anomaly metric (for debugging/report)
    """
    magnitude = compute_fft_magnitude(image)
    radial = _radial_profile(magnitude)

    # Ignore the DC-heavy innermost region (average brightness) and the very
    # noisy extreme edge (often just sensor/compression noise for both classes).
    n = len(radial)
    lo = max(2, int(n * 0.03))
    hi = max(lo + 4, int(n * 0.85))
    segment = radial[lo:hi]

    if len(segment) < 8:
        return {
            "score": 50.0,
            "spectrum": magnitude,
            "explanation": "Image too small for reliable frequency analysis.",
            "anomaly_ratio": 0.0,
        }

    # Smooth the profile with a simple moving average to model the "expected"
    # natural falloff, then measure how much the real profile spikes above it.
    kernel_size = max(3, len(segment) // 20)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(segment, kernel, mode="same")

    residual = segment - smoothed
    positive_residual = np.clip(residual, 0, None)

    # Normalize spikiness relative to the overall energy scale of the segment.
    energy_scale = np.mean(np.abs(segment)) + 1e-6
    anomaly_ratio = float(np.mean(positive_residual) / energy_scale)

    # Also check for a small number of very sharp, isolated peaks (classic
    # periodic-grid signature) using the max residual relative to std dev.
    residual_std = float(np.std(residual)) + 1e-6
    peak_z = float(np.max(residual) / residual_std)

    # Combine both cues into a 0-100 score via a smooth logistic-ish mapping.
    raw = anomaly_ratio * 40.0 + max(0.0, peak_z - 3.0) * 8.0
    score = float(np.clip(raw, 0.0, 100.0))

    if score >= 65:
        explanation = (
            "The frequency spectrum shows sharp, periodic spikes consistent with "
            "repeated upsampling operations often found in AI image generators."
        )
    elif score <= 35:
        explanation = (
            "The frequency spectrum falls off smoothly, consistent with natural "
            "camera photographs and without strong periodic upsampling artifacts."
        )
    else:
        explanation = (
            "The frequency spectrum shows some irregularities, but not a strong "
            "or conclusive periodic-artifact signature either way."
        )

    return {
        "score": score,
        "spectrum": magnitude,
        "explanation": explanation,
        "anomaly_ratio": anomaly_ratio,
    }
