"""Calibrate same/different thresholds from labelled vehicle folders."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.fingerprint import create_fingerprint  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_vectors(root: Path) -> list[tuple[str, np.ndarray]]:
    samples = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS or path.parent == root:
            continue
        with Image.open(path) as image:
            samples.append((path.parent.name, create_fingerprint(image.convert("RGB"))))
    return samples


def collect_scores(samples: list[tuple[str, np.ndarray]], max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for label, vector in samples:
        by_label[label].append(vector)

    positive = []
    for vectors in by_label.values():
        for left in range(len(vectors)):
            for right in range(left + 1, len(vectors)):
                positive.append(float(vectors[left] @ vectors[right]))
                if len(positive) >= max_pairs:
                    break
            if len(positive) >= max_pairs:
                break
        if len(positive) >= max_pairs:
            break

    rng = random.Random(seed)
    negative = []
    labels = list(by_label)
    attempts = 0
    while len(negative) < max_pairs and len(labels) >= 2 and attempts < max_pairs * 20:
        first_label, second_label = rng.sample(labels, 2)
        first = rng.choice(by_label[first_label])
        second = rng.choice(by_label[second_label])
        negative.append(float(first @ second))
        attempts += 1

    return np.asarray(positive), np.asarray(negative)


def metrics_at(positive: np.ndarray, negative: np.ndarray, threshold: float) -> dict:
    tp = int(np.sum(positive >= threshold))
    fn = int(np.sum(positive < threshold))
    fp = int(np.sum(negative >= threshold))
    tn = int(np.sum(negative < threshold))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fp / max(1, fp + tn), 6),
    }


def calibrate(root: Path, max_pairs: int = 50_000, target_fpr: float = 0.01, seed: int = 42) -> dict:
    samples = load_vectors(root)
    positive, negative = collect_scores(samples, max_pairs=max_pairs, seed=seed)
    if not positive.size or not negative.size:
        raise ValueError("Нужны минимум два автомобиля и по два изображения хотя бы одного автомобиля")

    combined = np.concatenate([positive, negative])
    candidates = np.unique(np.quantile(combined, np.linspace(0.0, 1.0, 501)))
    evaluated = [metrics_at(positive, negative, value) for value in candidates]
    best_f1 = max(evaluated, key=lambda item: (item["f1"], item["precision"]))
    safe = [item for item in evaluated if item["false_positive_rate"] <= target_fpr]
    high_confidence = min(safe, key=lambda item: item["threshold"]) if safe else metrics_at(positive, negative, 1.0)

    return {
        "samples": len(samples),
        "identities": len({label for label, _ in samples}),
        "positive_pairs": int(positive.size),
        "negative_pairs": int(negative.size),
        "positive_scores": {
            "min": round(float(positive.min()), 6),
            "median": round(float(np.median(positive)), 6),
            "max": round(float(positive.max()), 6),
        },
        "negative_scores": {
            "min": round(float(negative.min()), 6),
            "median": round(float(np.median(negative)), 6),
            "max": round(float(negative.max()), 6),
        },
        "recommended_review_threshold": best_f1,
        "recommended_high_threshold": high_confidence,
        "target_false_positive_rate": target_fpr,
        "warning": "Порог применим только к данным, на которых выполнена калибровка.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Подбор порогов совпадения")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("thresholds.json"))
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()
    report = calibrate(args.dataset, max_pairs=max(100, args.max_pairs), target_fpr=args.target_fpr)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

