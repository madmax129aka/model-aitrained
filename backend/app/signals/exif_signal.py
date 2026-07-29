"""
Metadata / EXIF analysis signal.

Real camera photos generally carry EXIF metadata (camera make/model,
exposure settings, sometimes GPS). AI-generated images and images that have
been through generation/editing pipelines frequently have no EXIF data, or
only generic/software tags, because that metadata is either never written or
is stripped in the generation pipeline.

This is a weak, heuristic secondary signal (weighted 20% in the final
score) -- absence of EXIF is common for real screenshots, WhatsApp-compressed
photos, and web-downloaded real images too, so it is intentionally not
treated as strong evidence on its own.
"""
import io

import exifread


CAMERA_TAG_PREFIXES = ("Image Make", "Image Model", "EXIF LensModel", "EXIF FocalLength")


def analyze(image_bytes: bytes) -> dict:
    """
    Inspect EXIF tags in the original uploaded bytes.

    Returns a dict with:
        score        float in [0, 100] -- "AI-likelihood" contribution from this signal
        has_exif     bool
        has_camera_info bool
        tags         dict[str, str] -- a small curated subset of readable tags
        explanation  str
    """
    try:
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
    except Exception:
        tags = {}

    has_exif = len(tags) > 0
    has_camera_info = any(
        any(tag_name.startswith(prefix) for prefix in CAMERA_TAG_PREFIXES)
        for tag_name in tags.keys()
    )

    readable_tags = {}
    for key in ("Image Make", "Image Model", "EXIF LensModel", "EXIF FocalLength",
                "EXIF DateTimeOriginal", "EXIF ExposureTime", "EXIF FNumber",
                "EXIF ISOSpeedRatings", "Image Software"):
        if key in tags:
            readable_tags[key] = str(tags[key])

    if has_camera_info:
        score = 10.0
        explanation = (
            "Camera metadata (make/model/lens or exposure info) was found, which is "
            "typical of a real photograph and uncommon in AI-generated images."
        )
    elif has_exif:
        score = 45.0
        explanation = (
            "Some metadata is present, but no camera make/model or lens information "
            "was found. This is inconclusive -- many real images (screenshots, "
            "web-downloaded, or re-compressed photos) also lack camera tags."
        )
    else:
        score = 60.0
        explanation = (
            "No EXIF metadata was found at all. This is common in AI-generated "
            "images (which are not produced by a camera), but is also common for "
            "images that were re-saved, screenshotted, or downloaded from the web, "
            "so this signal alone is not conclusive."
        )

    return {
        "score": score,
        "has_exif": has_exif,
        "has_camera_info": has_camera_info,
        "tags": readable_tags,
        "explanation": explanation,
    }
