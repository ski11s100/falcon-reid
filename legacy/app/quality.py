"""Explainable input-image quality checks."""

from __future__ import annotations

import numpy as np
from PIL import Image


def analyze_quality(image: Image.Image) -> dict:
    rgb = image.convert("RGB")
    array = np.asarray(rgb.resize((256, 160)), dtype=np.float32) / 255.0
    gray = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]

    brightness = float(gray.mean())
    contrast = float(gray.std())
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()
    sharpness = float((horizontal + vertical) / 2.0)

    issues: list[str] = []
    deductions = 0
    if rgb.width < 240 or rgb.height < 140:
        issues.append("низкое разрешение")
        deductions += 25
    if brightness < 0.12:
        issues.append("кадр слишком тёмный")
        deductions += 25
    elif brightness > 0.92:
        issues.append("кадр пересвечен")
        deductions += 20
    if contrast < 0.075:
        issues.append("низкий контраст")
        deductions += 20
    if sharpness < 0.018:
        issues.append("возможное размытие")
        deductions += 25

    score = max(0, 100 - deductions)
    return {
        "score": score,
        "status": "good" if score >= 75 else "warning" if score >= 45 else "poor",
        "issues": issues,
        "measurements": {
            "width": rgb.width,
            "height": rgb.height,
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "sharpness": round(sharpness, 4),
        },
    }

