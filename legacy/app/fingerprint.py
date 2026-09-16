"""Deterministic visual fingerprint used as the offline MVP baseline."""

from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

FINGERPRINT_VERSION = "baseline-v3-bbox"


def _configured_threshold(name: str, default: float) -> float:
    try:
        return float(np.clip(float(os.environ.get(name, default)), 0.0, 1.0))
    except ValueError:
        return default


REVIEW_THRESHOLD = _configured_threshold("FALCON_REVIEW_THRESHOLD", 0.86)
HIGH_THRESHOLD = max(REVIEW_THRESHOLD, _configured_threshold("FALCON_HIGH_THRESHOLD", 0.94))


class InvalidImage(ValueError):
    """Raised when an uploaded image cannot be decoded."""


def crop_bbox(image: Image.Image, bbox: object | None) -> Image.Image:
    """Crop an image by the challenge bbox format (x, y, width, height).

    A missing bbox means that the supplied image is already a vehicle crop.
    """
    if bbox is None:
        return image.copy()
    if isinstance(bbox, dict):
        values = [bbox.get(key) for key in ("x", "y", "w", "h")]
        if values[2] is None:
            values[2] = bbox.get("width")
        if values[3] is None:
            values[3] = bbox.get("height")
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        values = list(bbox)
    else:
        raise InvalidImage("bbox должен быть объектом {x,y,w,h} или массивом из четырёх чисел")
    try:
        x, y, width, height = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise InvalidImage("Координаты bbox должны быть числами") from exc
    if not all(math.isfinite(v) for v in (x, y, width, height)):
        raise InvalidImage("bbox должен содержать конечные числа")
    if x >= image.width or y >= image.height or x + width <= 0 or y + height <= 0:
        raise InvalidImage("bbox не пересекается с изображением")
    if width <= 1 or height <= 1:
        raise InvalidImage("Ширина и высота bbox должны быть больше 1 пикселя")
    left = max(0, min(image.width - 1, int(round(x))))
    top = max(0, min(image.height - 1, int(round(y))))
    right = max(left + 1, min(image.width, int(round(x + width))))
    bottom = max(top + 1, min(image.height, int(round(y + height))))
    if right - left <= 1 or bottom - top <= 1:
        raise InvalidImage("bbox не пересекается с изображением")
    return image.crop((left, top, right, bottom))


def decode_image(value: str, max_bytes: int = 10 * 1024 * 1024) -> Image.Image:
    if not isinstance(value, str) or not value.strip():
        raise InvalidImage("Поле image_base64 пустое")
    encoded = value.split(",", 1)[1] if "," in value else value
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise InvalidImage("Некорректная строка Base64") from exc
    if len(raw) > max_bytes:
        raise InvalidImage("Изображение превышает лимит 10 МБ")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise InvalidImage("Не удалось прочитать изображение") from exc


def _l2(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def _relative_colors(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate chromaticity from lighting intensity."""
    intensity = array.mean(axis=2)
    chromaticity = array / np.maximum(array.sum(axis=2, keepdims=True), 1e-4)
    relative_intensity = np.clip(intensity / max(float(intensity.mean()), 1e-4), 0.0, 2.0)
    return chromaticity, relative_intensity


def _channel_histogram(array: np.ndarray, bins: int = 16) -> np.ndarray:
    chromaticity, relative_intensity = _relative_colors(array)
    parts = []
    for channel in range(3):
        hist, _ = np.histogram(chromaticity[..., channel], bins=bins, range=(0.0, 1.0))
        parts.append(hist.astype(np.float32) / max(1, array.shape[0] * array.shape[1]))
    light_hist, _ = np.histogram(relative_intensity, bins=bins, range=(0.0, 2.0))
    parts.append(light_hist.astype(np.float32) / max(1, array.shape[0] * array.shape[1]))
    return np.concatenate(parts)


def _spatial_color(array: np.ndarray, cells: int = 4) -> np.ndarray:
    chromaticity, relative_intensity = _relative_colors(array)
    height, width, _ = chromaticity.shape
    features: list[float] = []
    for row in range(cells):
        for column in range(cells):
            y0, y1 = row * height // cells, (row + 1) * height // cells
            x0, x1 = column * width // cells, (column + 1) * width // cells
            patch = chromaticity[y0:y1, x0:x1]
            light_patch = relative_intensity[y0:y1, x0:x1]
            features.extend(patch.mean(axis=(0, 1)).tolist())
            features.extend(patch.std(axis=(0, 1)).tolist())
            features.extend([float(light_patch.mean()), float(light_patch.std())])
    return np.asarray(features, dtype=np.float32)


def _gradient_histogram(gray: np.ndarray, cells: int = 4, bins: int = 8) -> np.ndarray:
    dy, dx = np.gradient(gray)
    magnitude = np.hypot(dx, dy)
    angle = (np.arctan2(dy, dx) + np.pi) % np.pi
    height, width = gray.shape
    features: list[float] = []
    for row in range(cells):
        for column in range(cells):
            y0, y1 = row * height // cells, (row + 1) * height // cells
            x0, x1 = column * width // cells, (column + 1) * width // cells
            hist, _ = np.histogram(
                angle[y0:y1, x0:x1],
                bins=bins,
                range=(0.0, np.pi),
                weights=magnitude[y0:y1, x0:x1],
            )
            features.extend(_l2(hist).tolist())
    return np.asarray(features, dtype=np.float32)


def _shape_signature(gray: np.ndarray) -> np.ndarray:
    small = np.asarray(
        Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255))).resize((16, 8), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    small -= small.mean()
    return _l2(small)


def create_fingerprint(image: Image.Image) -> np.ndarray:
    """Return a compact normalized vector robust to moderate resize/lighting changes."""
    normalized = ImageOps.fit(image.convert("RGB"), (256, 160), method=Image.Resampling.LANCZOS)
    array = np.asarray(normalized, dtype=np.float32) / 255.0
    gray = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]

    color = _l2(_channel_histogram(array)) * 0.38
    spatial = _l2(_spatial_color(array)) * 0.22
    gradients = _l2(_gradient_histogram(gray)) * 0.25
    shape = _l2(_shape_signature(gray)) * 0.15
    return _l2(np.concatenate([color, spatial, gradients, shape]))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Размерности цифровых признаков не совпадают")
    return float(np.clip(np.dot(_l2(left), _l2(right)), -1.0, 1.0))


@dataclass(frozen=True)
class MatchDecision:
    score: float
    verdict: str


def classify_match(score: float) -> MatchDecision:
    if score >= HIGH_THRESHOLD:
        verdict = "высокая вероятность совпадения"
    elif score >= REVIEW_THRESHOLD:
        verdict = "требуется дополнительная проверка"
    else:
        verdict = "совпадение маловероятно"
    return MatchDecision(round(float(score), 4), verdict)
