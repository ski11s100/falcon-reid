"""Evaluate baseline retrieval on dataset/<vehicle_id>/<images>."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.fingerprint import create_fingerprint  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def average_precision(labels: list[str], target: str) -> float:
    relevant = 0
    precision_sum = 0.0
    total_relevant = sum(label == target for label in labels)
    if total_relevant == 0:
        return 0.0
    for rank, label in enumerate(labels, start=1):
        if label == target:
            relevant += 1
            precision_sum += relevant / rank
    return precision_sum / total_relevant


def evaluate(root: Path) -> dict:
    samples = [
        (path.parent.name, path)
        for path in sorted(root.rglob("*"))
        if path.suffix.lower() in EXTENSIONS and path.parent != root
    ]
    if len(samples) < 2:
        raise ValueError("Для оценки нужно минимум два изображения")

    started = time.perf_counter()
    vectors = []
    valid_samples = []
    errors = []
    per_image_ms = []
    for label, path in samples:
        tick = time.perf_counter()
        try:
            with Image.open(path) as image:
                vectors.append(create_fingerprint(image.convert("RGB")))
            valid_samples.append((label, path))
            per_image_ms.append((time.perf_counter() - tick) * 1000)
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})

    if len(vectors) < 2:
        raise ValueError("После чтения изображений осталось меньше двух образцов")
    matrix = np.vstack(vectors)
    similarities = matrix @ matrix.T
    recall1 = recall5 = 0
    aps = []
    eligible = 0
    for index, (target, _) in enumerate(valid_samples):
        order = np.argsort(-similarities[index])
        order = [position for position in order if position != index]
        ranked_labels = [valid_samples[position][0] for position in order]
        if target not in ranked_labels:
            continue
        eligible += 1
        recall1 += int(target in ranked_labels[:1])
        recall5 += int(target in ranked_labels[:5])
        aps.append(average_precision(ranked_labels, target))

    if eligible == 0:
        raise ValueError("Нет классов минимум с двумя изображениями; Recall посчитать нельзя")
    return {
        "samples": len(valid_samples),
        "identities": len({label for label, _ in valid_samples}),
        "eligible_queries": eligible,
        "recall_at_1": round(recall1 / eligible, 4),
        "recall_at_5": round(recall5 / eligible, 4),
        "mean_average_precision": round(float(np.mean(aps)), 4),
        "latency_ms": {
            "mean": round(float(np.mean(per_image_ms)), 2),
            "p95": round(float(np.percentile(per_image_ms, 95)), 2),
            "total": round((time.perf_counter() - started) * 1000, 2),
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка retrieval-качества baseline")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()
    report = evaluate(args.dataset)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

