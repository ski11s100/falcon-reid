"""Чтение конкурсных CSV, загрузка кропов и сэмплирование батчей.

Формат данных подтверждён организаторами (ответы 1, 4, 5, 8, 9, 27):
  train.csv        image_id, x, y, w, h, vehicle_id, camera_id
  test_query.csv   image_id, x, y, w, h
  test_gallery.csv image_id, x, y, w, h
Один кадр = один image_id = один объект. image_id уникален внутри каждого файла.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile

from .metrics import Identity

# Конкурсные кадры приходят из камерных комплексов; отдельные файлы бывают
# дописаны не полностью. Лучше отдать слегка усечённый кадр, чем уронить прогон.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class Observation:
    """Одна строка CSV: кадр, рамка целевого ТС и (для train) разметка."""

    row_id: int
    image_id: str
    path: Path
    bbox: tuple[float, float, float, float]
    vehicle_id: str | None = None
    camera_id: str | None = None

    @property
    def identity(self) -> Identity:
        if self.vehicle_id is None:
            raise ValueError(f"Строка {self.row_id}: нет vehicle_id, разметка недоступна")
        return Identity(self.vehicle_id, self.camera_id)


def _resolve_image(root: Path, image_id: str) -> Path:
    """Находит файл кадра. image_id может быть как с расширением, так и без."""
    direct = (root / image_id).resolve()
    if not direct.is_relative_to(root):
        raise ValueError(f"image_id выходит за пределы каталога images: {image_id!r}")
    if direct.is_file():
        return direct
    for suffix in IMAGE_EXTENSIONS:
        candidate = direct.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Не найден файл кадра для image_id={image_id!r}")


def read_manifest(csv_path: Path | str, images_dir: Path | str, *, require_labels: bool = False) -> list[Observation]:
    """Читает конкурсный CSV в список наблюдений с проверкой целостности."""
    csv_path, root = Path(csv_path), Path(images_dir).resolve()
    rows: list[Observation] = []
    seen: set[str] = set()

    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        width_key = "w" if "w" in fields else "width"
        height_key = "h" if "h" in fields else "height"
        required = {"image_id", "x", "y", width_key, height_key}
        if require_labels:
            required.add("vehicle_id")
        missing = required - fields
        if missing:
            raise ValueError(f"{csv_path.name}: отсутствуют колонки {sorted(missing)}")

        for index, row in enumerate(reader):
            image_id = (row["image_id"] or "").strip()
            if not image_id:
                raise ValueError(f"{csv_path.name}, строка {index}: пустой image_id")
            if image_id in seen:
                # Организаторы гарантировали уникальность (ответ 27). Нарушение
                # означает битый файл — молча склеивать такое нельзя.
                raise ValueError(f"{csv_path.name}, строка {index}: image_id {image_id!r} повторяется")
            seen.add(image_id)

            bbox = tuple(float(row[key]) for key in ("x", "y", width_key, height_key))
            if not all(math.isfinite(value) for value in bbox):
                raise ValueError(f"{csv_path.name}, строка {index}: bbox содержит не-числа")
            if min(bbox[2], bbox[3]) <= 1:
                raise ValueError(f"{csv_path.name}, строка {index}: вырожденный bbox {bbox}")

            vehicle_id = (row.get("vehicle_id") or "").strip() or None
            if require_labels and vehicle_id is None:
                raise ValueError(f"{csv_path.name}, строка {index}: нет vehicle_id")

            rows.append(
                Observation(
                    row_id=index,
                    image_id=image_id,
                    path=_resolve_image(root, image_id),
                    bbox=bbox,
                    vehicle_id=vehicle_id,
                    camera_id=(row.get("camera_id") or "").strip() or None,
                )
            )

    if not rows:
        raise ValueError(f"{csv_path.name}: пустой манифест")
    return rows


def load_crop(observation: Observation, target: tuple[int, int] | None = None) -> Image.Image:
    """Читает кадр и вырезает ТС по bbox.

    Чтение и декодирование входят в замеряемый организаторами цикл (ответ 31),
    поэтому используется draft(): libjpeg умеет отдавать кадр уменьшенным в 2/4/8
    раз прямо при DCT-декодировании, что кратно дешевле полного декодирования
    кадра 1920x1080 ради кропа, который всё равно ужимается до 256 пикселей.
    """
    x, y, width, height = observation.bbox
    with Image.open(observation.path) as image:
        original_width, original_height = image.size

        if target is not None and image.format == "JPEG":
            # Максимальное уменьшение, при котором кроп ещё не мельче цели.
            scale = 1
            while (
                scale < 8
                and width / (scale * 2) >= target[0]
                and height / (scale * 2) >= target[1]
            ):
                scale *= 2
            if scale > 1:
                image.draft("RGB", (original_width // scale, original_height // scale))

        image.load()
        # libjpeg округляет размер вверх, поэтому фактический масштаб берётся из
        # реальных размеров кадра, а не из запрошенного делителя.
        factor_x = image.width / original_width
        factor_y = image.height / original_height

        left = max(0, min(image.width - 1, int(round(x * factor_x))))
        top = max(0, min(image.height - 1, int(round(y * factor_y))))
        right = max(left + 1, min(image.width, int(round((x + width) * factor_x))))
        bottom = max(top + 1, min(image.height, int(round((y + height) * factor_y))))
        # Кроп до конвертации: переводить в RGB весь кадр ради рамки — лишняя работа.
        return image.crop((left, top, right, bottom)).convert("RGB")


def group_by_identity(rows: list[Observation]) -> dict[str, list[Observation]]:
    groups: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        if row.vehicle_id is not None:
            groups[row.vehicle_id].append(row)
    return dict(groups)


def audit(rows: list[Observation]) -> dict:
    """Сводка по манифесту: сколько ТС, камер, снимков на ТС, кросс-камерных ID."""
    groups = group_by_identity(rows)
    per_identity = [len(v) for v in groups.values()]
    cameras_per_identity = [len({r.camera_id for r in v}) for v in groups.values()]
    return {
        "rows": len(rows),
        "identities": len(groups),
        "cameras": len({r.camera_id for r in rows if r.camera_id is not None}),
        "shots_per_identity_min": min(per_identity) if per_identity else 0,
        "shots_per_identity_max": max(per_identity) if per_identity else 0,
        "shots_per_identity_mean": round(sum(per_identity) / len(per_identity), 2) if per_identity else 0,
        "identities_with_multiple_cameras": sum(1 for c in cameras_per_identity if c >= 2),
        "singleton_identities": sum(1 for n in per_identity if n == 1),
        "camera_id_available": all(r.camera_id is not None for r in rows),
    }


@dataclass
class LocalSplit:
    """Локальный аналог закрытого теста, включая open-set запросы без пары.

    Организаторы подтвердили (ответ 17), что в закрытом тесте ~20% запросов не
    имеют пары в галерее, и рекомендовали собрать такой же сплит из train.csv.
    Без него порог отказа калибровать не на чем, а это 10% итогового балла.
    """

    train: list[Observation]
    gallery: list[Observation]
    query: list[Observation]
    open_set_query_ids: set[str]

    def summary(self) -> dict:
        return {
            "train_rows": len(self.train),
            "train_identities": len({r.vehicle_id for r in self.train}),
            "gallery_rows": len(self.gallery),
            "gallery_identities": len({r.vehicle_id for r in self.gallery}),
            "query_rows": len(self.query),
            "open_set_queries": len(self.open_set_query_ids),
            "open_set_share": round(len(self.open_set_query_ids) / max(1, len(self.query)), 3),
        }


def build_local_split(
    rows: list[Observation],
    *,
    val_identity_fraction: float = 0.25,
    open_set_fraction: float = 0.20,
    seed: int = 42,
) -> LocalSplit:
    """Режет train.csv на обучение и локальный тест по протоколу организаторов.

    Правила:
      * ID обучения и ID валидации не пересекаются (open-set, раздел 5 ТЗ);
      * у «закрытых» ID снимки делятся на галерею и запросы так, чтобы у запроса
        оставался хотя бы один позитив С ДРУГОЙ КАМЕРЫ — иначе он junk и в mAP
        не участвует;
      * отдельная доля ID идёт целиком в запросы и никогда в галерею: это и есть
        open-set запросы, на которых считается TNR.
    """
    if not 0 < val_identity_fraction < 1 or not 0 <= open_set_fraction < 1:
        raise ValueError("Доли должны лежать в (0, 1)")

    rng = random.Random(seed)
    groups = group_by_identity(rows)
    if any(r.camera_id is None for r in rows):
        raise ValueError("Нет camera_id — честный кросс-камерный сплит построить нельзя")

    # В валидацию имеет смысл брать только ID, снятые минимум двумя камерами:
    # у одно-камерного ID все позитивы окажутся junk.
    multi_camera = sorted(k for k, v in groups.items() if len({r.camera_id for r in v}) >= 2)
    single_camera = sorted(set(groups) - set(multi_camera))
    rng.shuffle(multi_camera)

    val_size = max(2, round(len(groups) * val_identity_fraction))
    val_size = min(val_size, len(multi_camera) - 1)
    val_ids = multi_camera[:val_size]

    train_ids = set(multi_camera[val_size:]) | set(single_camera)
    train = [r for r in rows if r.vehicle_id in train_ids]

    gallery: list[Observation] = []
    query: list[Observation] = []
    open_query_ids: set[str] = set()

    # Сначала закрытая часть: у каждого ID одна камера уходит в запросы,
    # остальные в галерею — так у запроса гарантированно есть кросс-камерный позитив.
    closed_pool: list[str] = []
    for vid in val_ids:
        by_camera: dict[str, list[Observation]] = defaultdict(list)
        for shot in groups[vid]:
            by_camera[str(shot.camera_id)].append(shot)
        cameras = sorted(by_camera)
        rng.shuffle(cameras)
        closed_pool.append(vid)
        query.extend(by_camera[cameras[0]])
        for camera in cameras[1:]:
            gallery.extend(by_camera[camera])

    # Теперь добираем open-set ID так, чтобы они дали примерно open_set_fraction
    # от ИТОГОВОГО числа запросов (ответ 17 говорит о доле запросов, не ID).
    # Их ID целиком выводятся из галереи: пары для них не существует.
    # query на этом шаге уже содержит ВСЕ запросы, поэтому доля берётся от него
    # напрямую: перевод open-set ID не добавляет запросов, а только убирает
    # соответствующие снимки из галереи.
    target_open_rows = open_set_fraction * len(query)
    open_rows = 0
    for vid in list(closed_pool):
        if open_rows >= target_open_rows:
            break
        moved = [r for r in gallery if r.vehicle_id == vid]
        if not moved:
            continue
        # Убираем ID из галереи целиком: его запросы становятся open-set.
        gallery = [r for r in gallery if r.vehicle_id != vid]
        open_query_ids.update(r.image_id for r in query if r.vehicle_id == vid)
        open_rows = sum(1 for r in query if r.image_id in open_query_ids)

    return LocalSplit(train=train, gallery=gallery, query=query, open_set_query_ids=open_query_ids)


class PKSampler:
    """Сэмплер батчей вида P идентичностей x K снимков.

    Batch-hard triplet loss ищет самый трудный позитив и негатив ВНУТРИ батча,
    поэтому случайный батч почти бесполезен: в нём нет пар одного ТС. Здесь же
    каждый батч содержит P машин по K снимков каждая.

    Снимки одной машины по возможности берутся с РАЗНЫХ камер: в метрике
    засчитываются только кросс-камерные совпадения, и учить модель надо ровно им.

    Выдаёт списки ИНДЕКСОВ в rows, поэтому подходит как batch_sampler для
    torch.utils.data.DataLoader — один загрузчик живёт всю эпоху, а воркеры
    не перезапускаются на каждом шаге.
    """

    def __init__(
        self,
        rows: list[Observation],
        *,
        identities_per_batch: int = 8,
        shots_per_identity: int = 4,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ):
        if identities_per_batch < 2 or shots_per_identity < 2:
            raise ValueError("Нужно минимум 2 идентичности и 2 снимка на идентичность")
        self.rows = rows

        # Всё внутри сэмплера — индексы в rows, а не объекты.
        by_identity: dict[str, list[int]] = defaultdict(list)
        by_identity_camera: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            if row.vehicle_id is None:
                continue
            by_identity[row.vehicle_id].append(index)
            by_identity_camera[row.vehicle_id][str(row.camera_id)].append(index)

        self.by_identity = dict(by_identity)
        self.by_identity_camera = {k: dict(v) for k, v in by_identity_camera.items()}
        self.identities = sorted(k for k, v in self.by_identity.items() if len(v) >= 2)
        if len(self.identities) < identities_per_batch:
            raise ValueError(
                f"Всего {len(self.identities)} пригодных ID, а на батч нужно {identities_per_batch}"
            )

        self.P = identities_per_batch
        self.K = shots_per_identity
        self.batch_size = self.P * self.K
        self.batches_per_epoch = batches_per_epoch or max(1, len(rows) // self.batch_size)
        self.rng = random.Random(seed)

    def _sample_identity(self, vid: str) -> list[int]:
        """K снимков одной машины, максимально разнесённых по камерам."""
        cameras = list(self.by_identity_camera[vid])
        self.rng.shuffle(cameras)
        picked: list[int] = []
        # Сначала по одному снимку с каждой камеры — это даёт кросс-камерные пары.
        for camera in cameras:
            if len(picked) >= self.K:
                break
            picked.append(self.rng.choice(self.by_identity_camera[vid][camera]))
        # Если камер меньше K, добираем оставшееся любыми снимками.
        pool = self.by_identity[vid]
        while len(picked) < self.K:
            picked.append(self.rng.choice(pool))
        return picked

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen = self.rng.sample(self.identities, self.P)
            batch: list[int] = []
            for vid in chosen:
                batch.extend(self._sample_identity(vid))
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch
