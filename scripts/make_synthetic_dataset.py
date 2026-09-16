"""Генератор синтетического датасета в конкурсном формате.

Нужен, чтобы проверять весь конвейер (чтение CSV, сплит, обучение, метрика,
сдача) без настоящих данных и без риска их испортить. Устроен так, чтобы задача
была решаемой, но не тривиальной:

  * у каждого ТС свой цвет кузова, форма и набор «особых примет» (наклейки);
  * каждая камера накладывает свой стиль: яркость, оттенок, шум, размытие —
    это имитирует разное освещение и разные матрицы;
  * ТС помещается в полный кадр в случайное место, координаты пишутся в CSV;
  * есть пары разных ТС одного цвета и формы — как «две одинаковые машины»
    из раздела 5 ТЗ.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

FRAME_SIZE = (1280, 720)


def vehicle_sprite(rng: random.Random, hue: float, body: str, badges: list[tuple[float, float]],
                   size: tuple[int, int]) -> Image.Image:
    """Рисует условный автомобиль: кузов, окна, колёса и «особые приметы»."""
    width, height = size
    sprite = Image.new("RGB", (width, height), (30, 30, 34))
    draw = ImageDraw.Draw(sprite)

    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    colour = (int(r * 255), int(g * 255), int(b * 255))

    if body == "sedan":
        draw.rectangle([0, int(height * 0.45), width, int(height * 0.85)], fill=colour)
        draw.polygon([(int(width * 0.2), int(height * 0.45)), (int(width * 0.35), int(height * 0.2)),
                      (int(width * 0.7), int(height * 0.2)), (int(width * 0.82), int(height * 0.45))],
                     fill=colour)
    elif body == "suv":
        draw.rectangle([0, int(height * 0.3), width, int(height * 0.85)], fill=colour)
        draw.rectangle([int(width * 0.12), int(height * 0.12), int(width * 0.88), int(height * 0.35)],
                       fill=colour)
    else:  # van
        draw.rectangle([0, int(height * 0.15), width, int(height * 0.85)], fill=colour)

    # Остекление
    glass = (int(colour[0] * 0.25), int(colour[1] * 0.3), int(colour[2] * 0.4))
    draw.rectangle([int(width * 0.28), int(height * 0.24), int(width * 0.68), int(height * 0.44)],
                   fill=glass)
    # Колёса
    for cx in (0.22, 0.76):
        x0 = int(width * cx - height * 0.11)
        draw.ellipse([x0, int(height * 0.74), x0 + int(height * 0.22), int(height * 0.96)],
                     fill=(18, 18, 20))
    # «Особые приметы»: наклейки и повреждения — то, что задача просит различать.
    for fx, fy in badges:
        x = int(width * fx)
        y = int(height * fy)
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(250, 240, 90))
    return sprite


def apply_camera_style(image: Image.Image, rng: random.Random, camera: int) -> Image.Image:
    """Стиль камеры: свой баланс белого, яркость, шум и резкость."""
    style = random.Random(camera * 9973)
    tint = [style.uniform(0.75, 1.25) for _ in range(3)]
    gain = style.uniform(0.6, 1.3)

    pixels = image.load()
    for y in range(0, image.height, 2):
        for x in range(0, image.width, 2):
            r, g, b = pixels[x, y]
            noise = rng.randint(-12, 12)
            value = (
                min(255, max(0, int(r * tint[0] * gain) + noise)),
                min(255, max(0, int(g * tint[1] * gain) + noise)),
                min(255, max(0, int(b * tint[2] * gain) + noise)),
            )
            for dy in range(2):
                for dx in range(2):
                    if x + dx < image.width and y + dy < image.height:
                        pixels[x + dx, y + dy] = value

    if style.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(style.uniform(0.4, 1.2)))
    return image


def generate(output: Path, identities: int, cameras: int, shots: int, seed: int) -> dict:
    rng = random.Random(seed)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    bodies = ["sedan", "suv", "van"]
    profiles = []
    for vid in range(identities):
        # Каждое третье ТС намеренно клонирует цвет и кузов предыдущего:
        # это пары «одинаковая марка и цвет» из списка сложных сценариев ТЗ.
        if vid % 3 == 2 and profiles:
            base = profiles[-1]
            profiles.append({"hue": base["hue"], "body": base["body"],
                             "badges": [(rng.random(), rng.uniform(0.3, 0.7))]})
        else:
            profiles.append({
                "hue": rng.random(),
                "body": rng.choice(bodies),
                "badges": [(rng.random(), rng.uniform(0.3, 0.7)) for _ in range(rng.randint(1, 3))],
            })

    rows = []
    for vid, profile in enumerate(profiles):
        chosen_cameras = rng.sample(range(cameras), min(cameras, rng.randint(2, 4)))
        for index in range(shots):
            camera = chosen_cameras[index % len(chosen_cameras)]
            crop_w = rng.randint(120, 260)
            crop_h = int(crop_w * rng.uniform(0.55, 0.75))
            sprite = vehicle_sprite(rng, profile["hue"], profile["body"], profile["badges"],
                                    (crop_w, crop_h))
            sprite = apply_camera_style(sprite, rng, camera)

            frame = Image.new("RGB", FRAME_SIZE, (60 + camera * 3 % 40, 62, 68))
            x = rng.randint(0, FRAME_SIZE[0] - crop_w)
            y = rng.randint(0, FRAME_SIZE[1] - crop_h)
            frame.paste(sprite, (x, y))

            image_id = f"img_{vid:05d}_{index:02d}.jpg"
            frame.save(images_dir / image_id, quality=88)
            rows.append({"image_id": image_id, "x": x, "y": y, "w": crop_w, "h": crop_h,
                         "vehicle_id": str(vid), "camera_id": str(camera)})

    with (output / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "x", "y", "w", "h",
                                                    "vehicle_id", "camera_id"])
        writer.writeheader()
        writer.writerows(rows)

    # Отдельные test_query.csv и test_gallery.csv без разметки — как у организаторов.
    holdout = [r for r in rows if int(r["vehicle_id"]) >= identities - max(2, identities // 5)]
    query = holdout[::3]
    gallery = [r for r in holdout if r not in query]
    for name, subset in (("test_query.csv", query), ("test_gallery.csv", gallery)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "x", "y", "w", "h"])
            writer.writeheader()
            writer.writerows({k: r[k] for k in ("image_id", "x", "y", "w", "h")} for r in subset)

    return {"rows": len(rows), "identities": identities, "cameras": cameras,
            "query": len(query), "gallery": len(gallery), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Синтетический датасет в конкурсном формате")
    parser.add_argument("output", type=Path)
    parser.add_argument("--identities", type=int, default=60)
    parser.add_argument("--cameras", type=int, default=6)
    parser.add_argument("--shots", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import json
    print(json.dumps(generate(args.output, args.identities, args.cameras, args.shots, args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
