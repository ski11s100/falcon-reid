"""Приведение VeRi-776 к конкурсному формату манифеста.

VeRi-776 — публичный датасет городского видеонаблюдения: 49 357 снимков,
776 автомобилей, 20 камер. Ближе всего к нашей задаче по природе данных:
кросс-камерная съёмка, те же ракурсы и условия освещения.

Зачем: в конкурсном train.csv 9 556 снимков и 1 541 автомобиль. Предобучение на
вчетверо большем наборе с последующим дообучением на конкурсных данных — самый
крупный прирост точности, доступный без смены архитектуры. Раздел 7 ТЗ прямо
разрешает сторонние открытые датасеты при условии перечисления их в README.

Отличие от конкурсных данных: снимки VeRi уже вырезаны по автомобилю, тогда как
у нас кроп берётся из полного кадра 1920x1080. Поэтому BBox здесь равен всему
изображению, а разметка извлекается из имени файла:

    0001_c001_00016450_0.jpg
    ^^^^ ^^^^
    ТС   камера

Источник: https://github.com/VehicleReId/VeRidataset
Доступ получен по заявке автору (Xinchen Liu, BUPT), некоммерческое
исследовательское использование.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NAME_PATTERN = re.compile(r"^(\d+)_c(\d+)_")


def parse_name(name: str) -> tuple[str, str] | None:
    """Идентификатор ТС и камеры из имени файла VeRi."""
    match = NAME_PATTERN.match(name)
    if not match:
        return None
    # Префикс veri- защищает от коллизии с числовыми vehicle_id конкурсного
    # набора, если наборы когда-нибудь окажутся в одном манифесте.
    return f"veri-{int(match.group(1))}", f"veri-c{int(match.group(2))}"


def build(veri_root: Path, output: Path, subsets: tuple[str, ...]) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped = 0
    identities: set[str] = set()
    cameras: set[str] = set()

    for subset in subsets:
        folder = veri_root / subset
        if not folder.is_dir():
            raise FileNotFoundError(f"Не найден каталог {folder}")

        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(json.dumps({"subset": subset, "files": len(files)}), flush=True)

        for index, path in enumerate(files):
            parsed = parse_name(path.name)
            if parsed is None:
                skipped += 1
                continue
            vehicle_id, camera_id = parsed

            # Размер читается из заголовка: PIL не декодирует пиксели до load().
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                skipped += 1
                continue
            if min(width, height) <= 1:
                skipped += 1
                continue

            rows.append({
                "image_id": f"{subset}/{path.name}",
                "x": 0, "y": 0, "w": width, "h": height,
                "vehicle_id": vehicle_id, "camera_id": camera_id,
            })
            identities.add(vehicle_id)
            cameras.add(camera_id)

            if (index + 1) % 10000 == 0:
                print(json.dumps({"subset": subset, "processed": index + 1}), flush=True)

    if not rows:
        raise ValueError("Ни одной пригодной строки — проверьте путь к VeRi")

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "x", "y", "w", "h",
                                                    "vehicle_id", "camera_id"])
        writer.writeheader()
        writer.writerows(rows)

    per_identity: dict[str, int] = {}
    for row in rows:
        per_identity[row["vehicle_id"]] = per_identity.get(row["vehicle_id"], 0) + 1

    return {
        "rows": len(rows),
        "identities": len(identities),
        "cameras": len(cameras),
        "skipped": skipped,
        "shots_per_identity_mean": round(len(rows) / len(identities), 2),
        "shots_per_identity_min": min(per_identity.values()),
        "shots_per_identity_max": max(per_identity.values()),
        "manifest": str(output),
        "images_dir": str(veri_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VeRi-776 -> манифест ФАЛЬКОН")
    parser.add_argument("veri", type=Path, help="Каталог VeRi (внутри image_train и т. д.)")
    parser.add_argument("--output", type=Path, default=Path("D:/falcon-external/veri_train.csv"))
    parser.add_argument("--subsets", nargs="+", default=["image_train"],
                        help="Какие части использовать. image_test/image_query содержат "
                             "те же 200 отложенных ТС и для предобучения не нужны")
    args = parser.parse_args()

    report = build(args.veri, args.output, tuple(args.subsets))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
