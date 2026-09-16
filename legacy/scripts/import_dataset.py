"""Create local profiles from dataset/<vehicle_id>/<images> without HTTP overhead."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.fingerprint import create_fingerprint  # noqa: E402
from app.storage import VehicleStore  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Импорт эталонных профилей")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--db", type=Path, default=Path("data/falcon.sqlite3"))
    args = parser.parse_args()
    store = VehicleStore(args.db)
    imported = failed = 0
    for identity_dir in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
        images = sorted(path for path in identity_dir.iterdir() if path.suffix.lower() in EXTENSIONS)
        if not images:
            continue
        identity_imported = 0
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    fingerprint = create_fingerprint(image.convert("RGB"))
                store.upsert(
                    identity_dir.name,
                    fingerprint,
                    {"source": str(identity_dir), "images": len(images)},
                )
                identity_imported += 1
            except Exception as exc:
                failed += 1
                print(f"Ошибка {image_path}: {exc}")
        imported += int(identity_imported > 0)
    print(f"Импортировано профилей: {imported}; ошибок изображений: {failed}")


if __name__ == "__main__":
    main()
