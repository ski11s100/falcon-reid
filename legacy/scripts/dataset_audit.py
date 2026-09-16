"""Audit an image dataset before training.

Expected layout: dataset/<vehicle_id>/<image files>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def audit(root: Path) -> dict:
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in EXTENSIONS)
    identities: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    broken: list[dict] = []
    sizes: list[int] = []

    for path in files:
        relative = path.relative_to(root)
        identity = relative.parts[0] if len(relative.parts) > 1 else "_unassigned"
        try:
            raw = path.read_bytes()
            with Image.open(path) as image:
                image.verify()
                dimensions[f"{image.width}x{image.height}"] += 1
            identities[identity] += 1
            sizes.append(len(raw))
            hashes[hashlib.sha256(raw).hexdigest()].append(str(relative))
        except Exception as exc:
            broken.append({"file": str(relative), "error": str(exc)})

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    counts = list(identities.values())
    return {
        "root": str(root.resolve()),
        "images_total": len(files),
        "images_valid": sum(counts),
        "identities": len(identities),
        "unassigned_images": identities.get("_unassigned", 0),
        "images_per_identity": {
            "min": min(counts, default=0),
            "median": statistics.median(counts) if counts else 0,
            "max": max(counts, default=0),
        },
        "bytes_total": sum(sizes),
        "top_dimensions": dimensions.most_common(10),
        "broken": broken,
        "exact_duplicate_groups": duplicate_groups,
        "identities_with_one_image": sorted(key for key, value in identities.items() if value == 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка датасета автомобилей")
    parser.add_argument("dataset", type=Path, help="Папка dataset/<vehicle_id>/<images>")
    parser.add_argument("--output", type=Path, default=Path("dataset_report.json"))
    args = parser.parse_args()
    report = audit(args.dataset)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("images_total", "images_valid", "identities")}, ensure_ascii=False))
    print(f"Полный отчёт: {args.output.resolve()}")


if __name__ == "__main__":
    main()

