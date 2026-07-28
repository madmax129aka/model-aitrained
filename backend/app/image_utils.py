"""Small helpers for image <-> base64 conversions and heatmap compositing."""
import base64
import io
from typing import Optional

import numpy as np
from PIL import Image


def load_image_rgb(image_bytes: bytes) -> Image.Image:
    """Decode arbitrary uploaded image bytes into a PIL RGB image."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    return img


def pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data: URI string for embedding in JSON."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"


def data_uri_to_bytes(data_uri: Optional[str]) -> Optional[bytes]:
    """Decode a `data:image/...;base64,...` URI back into raw image bytes."""
    if not data_uri or "," not in data_uri:
        return None
    _, encoded = data_uri.split(",", 1)
    try:
        return base64.b64decode(encoded)
    except Exception:
        return None


def make_thumbnail(img: Image.Image, max_size: int = 320) -> Image.Image:
    thumb = img.copy()
    thumb.thumbnail((max_size, max_size))
    return thumb


def _apply_colormap(normalized: np.ndarray) -> np.ndarray:
    """
    Map a 2D array of values in [0, 1] to an RGB "hot" style colormap without
    requiring matplotlib. Returns an (H, W, 3) uint8 array.

    Colormap ramp: black -> deep blue -> cyan -> yellow -> red (roughly a
    "thermal" look consistent with the forensic-lab UI theme).
    """
    normalized = np.clip(normalized, 0.0, 1.0)

    # Control points (position, R, G, B) for a simple piecewise-linear ramp.
    stops = np.array(
        [
            [0.00, 8, 12, 32],
            [0.25, 20, 60, 160],
            [0.50, 0, 200, 200],
            [0.75, 255, 210, 40],
            [1.00, 255, 40, 40],
        ],
        dtype=np.float32,
    )
    positions = stops[:, 0]
    colors = stops[:, 1:]

    flat = normalized.reshape(-1)
    r = np.interp(flat, positions, colors[:, 0])
    g = np.interp(flat, positions, colors[:, 1])
    b = np.interp(flat, positions, colors[:, 2])
    rgb = np.stack([r, g, b], axis=-1).reshape(normalized.shape + (3,))
    return rgb.astype(np.uint8)


def saliency_to_heatmap_overlay(
    saliency: np.ndarray, base_image: Image.Image, alpha: float = 0.5
) -> Image.Image:
    """
    Convert a 2D saliency map (any resolution, float values) into a colorized
    heatmap resized to `base_image`'s size, then alpha-blend it on top of the
    (unmodified) base image.
    """
    sal = saliency.astype(np.float32)
    sal_min, sal_max = float(sal.min()), float(sal.max())
    if sal_max - sal_min < 1e-8:
        normalized = np.zeros_like(sal)
    else:
        normalized = (sal - sal_min) / (sal_max - sal_min)

    heat_rgb = _apply_colormap(normalized)
    heat_img = Image.fromarray(heat_rgb, mode="RGB").resize(
        base_image.size, resample=Image.BILINEAR
    )

    base_rgba = base_image.convert("RGBA")
    heat_rgba = heat_img.convert("RGBA")

    # Use saliency intensity itself (resized) to modulate per-pixel alpha, so
    # low-importance regions stay closer to the original image.
    alpha_map = Image.fromarray((normalized * 255).astype(np.uint8), mode="L").resize(
        base_image.size, resample=Image.BILINEAR
    )
    alpha_arr = (np.asarray(alpha_map, dtype=np.float32) / 255.0) * alpha
    alpha_channel = Image.fromarray((alpha_arr * 255).astype(np.uint8), mode="L")
    heat_rgba.putalpha(alpha_channel)

    composited = Image.alpha_composite(base_rgba, heat_rgba)
    return composited.convert("RGB")


def grayscale_array_to_image(arr: np.ndarray) -> Image.Image:
    """Normalize a float array to 0-255 uint8 grayscale and wrap as a PIL image."""
    a = arr.astype(np.float32)
    a_min, a_max = float(a.min()), float(a.max())
    if a_max - a_min < 1e-8:
        norm = np.zeros_like(a)
    else:
        norm = (a - a_min) / (a_max - a_min)
    return Image.fromarray((norm * 255).astype(np.uint8), mode="L")
