"""Наполнение демонстрационной галереи сервиса снимками из train.csv.

    python scripts/seed_demo_gallery.py D:/falcon-data --url http://localhost:8000

Двенадцать машин по 4–6 снимков с разных камер — та же галерея, на которой
записано демо (docs/DEMO_CHECKLIST.md), и пара двойников для показа правила
«требуется проверка». Снимки регистрируются через
/api/gallery/register-batch, то есть векторы считает сам сервис текущим
ансамблем; при смене модели галерею достаточно пересоздать этим скриптом.

Выбор снимков детерминирован: для каждой машины снимки сортируются по
image_id, и берётся по одному с каждой камеры по кругу, пока не наберётся
нужное число. С --from-sqlite берутся ровно те image_id, что лежат в старой
базе сервиса (например, после смены размерности вектора).

Двойники — два фургона Lamoda (Lada Largus) в одинаковой оклейке: разные
машины, которые без номера различаются только мелочами (у ТС-1022 на задней
двери наклейка с QR-кодом). Кадр ТС-925 с камеры 87 в галерею не идёт — он
сохраняется кропом в service/static/_test_twin.jpg как запрос: по нему обе
машины оказываются над порогом почти вровень (0.968 и 0.965 на отложенных
машинах, и без правила первой шла чужая).
"""

from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.data import read_manifest  # noqa: E402

DEMO_VEHICLES = {1099: 6, 1461: 5, 44: 4, 583: 6, 1172: 5, 1265: 5,
                 530: 4, 964: 5, 1080: 5, 1120: 5, 239: 6, 842: 5}
# По четыре снимка: «Попробовать на примере» берёт машины с пятью и больше,
# поэтому основной сценарий показа двойники не задевают.
TWIN_VEHICLES = {925: 4, 1022: 4}
TWIN_QUERY_IMAGE = "91abb61b837c4e0ea8f0fe2dd68db516"   # ТС-925, камера 87
TWIN_QUERY_CAMERA = "87"


def pick(rows, vehicle: int, count: int, skip_camera: str | None = None):
    by_camera = defaultdict(list)
    for row in sorted((r for r in rows if r.vehicle_id == str(vehicle)), key=lambda r: r.image_id):
        if row.camera_id != skip_camera:
            by_camera[row.camera_id].append(row)
    chosen, round_ = [], 0
    while len(chosen) < count and any(len(v) > round_ for v in by_camera.values()):
        for camera in sorted(by_camera):
            if len(by_camera[camera]) > round_ and len(chosen) < count:
                chosen.append(by_camera[camera][round_])
        round_ += 1
    return chosen


def save_twin_query(row, target: Path) -> None:
    from PIL import Image
    x, y, w, h = row.bbox
    crop = Image.open(row.path).convert("RGB").crop((int(x), int(y), int(x + w), int(y + h)))
    crop.save(target, quality=95)
    print(f"кадр-запрос двойников: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Демонстрационная галерея сервиса ФАЛЬКОН")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--from-sqlite", type=Path, default=None,
                        help="Взять image_id и подписи машин из старой базы SQLite сервиса")
    parser.add_argument("--twin-query", type=Path,
                        default=ROOT / "service" / "static" / "_test_twin.jpg",
                        help="Куда сохранить кадр-запрос для показа двойников")
    args = parser.parse_args()

    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    by_id = {r.image_id: r for r in rows}
    if args.from_sqlite:
        with sqlite3.connect(args.from_sqlite) as db:
            listed = db.execute("SELECT image_id, vehicle_id FROM gallery ORDER BY created_at").fetchall()
        chosen = [(by_id[i], label) for i, label in listed if i in by_id]
    else:
        chosen = [(row, f"ТС-{vehicle}") for vehicle, count in DEMO_VEHICLES.items()
                  for row in pick(rows, vehicle, count)]
        chosen += [(row, f"ТС-{vehicle}") for vehicle, count in TWIN_VEHICLES.items()
                   for row in pick(rows, vehicle, count, skip_camera=TWIN_QUERY_CAMERA)]
        save_twin_query(by_id[TWIN_QUERY_IMAGE], args.twin_query)

    items = [{
        "image_id": row.image_id,
        "vehicle_id": label,
        "image_base64": base64.b64encode(Path(row.path).read_bytes()).decode("ascii"),
        "bbox": dict(zip("xywh", row.bbox)),
    } for row, label in chosen]

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    registered = 0
    for start in range(0, len(items), 16):   # пачки поменьше лимита тела запроса
        body = json.dumps({"items": items[start:start + 16]}).encode("utf-8")
        request = urllib.request.Request(f"{args.url}/api/gallery/register-batch", data=body,
                                         headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        registered += result["registered"]
        for failure in result.get("failed", []):
            print("не зарегистрирован:", failure, flush=True)
    print(json.dumps({"registered": registered, "vehicles": len({label for _, label in chosen})},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
