"""Извлечение признаков: то, что организаторы замеряют на производительность.

Ответ 31 определяет замеряемый цикл: чтение файла с диска, декодирование, crop
по BBox, препроцессинг, forward, постобработка, L2-нормализация. Всё это лежит
внутри extract(), поэтому каждая строчка здесь влияет на 20% итогового балла.

Что сделано ради скорости:
  * fp16 и channels_last — Ampere (и 3060, и A5000 жюри) считает такое вдвое быстрее;
  * DataLoader с воркерами — декодирование JPEG идёт параллельно forward-проходу;
  * draft() при декодировании (см. data.load_crop) — libjpeg отдаёт кадр сразу
    уменьшенным, а не декодирует 1920x1080 целиком ради рамки;
  * cudnn.benchmark — подбор быстрейших алгоритмов свёртки под фиксированный размер.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import Observation, load_crop
from .model import VehicleReID, build_model
from .transforms import DEFAULT_SIZE, build_eval_transform


class CropDataset(Dataset):
    """Наблюдения -> тензоры. Вся тяжёлая работа с диском идёт в воркерах."""

    def __init__(self, rows: list[Observation], transform, size: tuple[int, int] = DEFAULT_SIZE,
                 labels: dict[str, int] | None = None):
        self.rows = rows
        self.transform = transform
        self.size = size
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = load_crop(row, target=self.size)
        tensor = self.transform(image)
        if self.labels is None:
            return tensor, index
        return tensor, self.labels[row.vehicle_id], _camera_code(row.camera_id), index


def shutdown_loader(loader: DataLoader) -> None:
    """Принудительно останавливает воркеров загрузчика.

    DataLoader с persistent_workers держит процессы, пока жив сам объект, а сбор
    мусора в CPython происходит не сразу. На Windows каждый воркер — полноценный
    процесс с загруженным torch, и накопление десятков таких процессов быстро
    съедает память.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        try:
            iterator._shutdown_workers()
        except Exception:
            pass
        loader._iterator = None


def _camera_code(camera_id: str | None) -> int:
    """Камера как целое: нужна triplet-потере для отбора кросс-камерных пар."""
    if camera_id is None:
        return -1
    try:
        return int(camera_id)
    except ValueError:
        return abs(hash(camera_id)) % 1_000_000


@dataclass
class ExtractorConfig:
    size: tuple[int, int] = DEFAULT_SIZE
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda"
    half: bool = True
    flip_tta: bool = True
    channels_last: bool = True


class FeatureExtractor:
    """Загружает чекпоинт и превращает наблюдения в матрицу эмбеддингов."""

    def __init__(self, checkpoint: Path | str | None = None, model: VehicleReID | None = None,
                 config: ExtractorConfig | None = None):
        self.config = config or ExtractorConfig()
        if not torch.cuda.is_available() and self.config.device.startswith("cuda"):
            self.config.device = "cpu"
            self.config.half = False
        self.device = torch.device(self.config.device)

        # Модель, переданную извне, экстрактор НЕ ВЛАДЕЕТ и потому не имеет права
        # менять: конвертация в half происходила бы прямо в объекте вызывающего.
        # При валидации во время обучения это означало round-trip fp32 -> fp16 ->
        # fp32 на каждой проверке: веса теряли точность, а повторное выделение
        # всех параметров фрагментировало видеопамять до вытеснения в системную.
        self._owns_model = model is None
        if model is not None:
            self.model = model
            self.metadata: dict = {"architecture": VehicleReID.ARCHITECTURE, "source": "in-memory"}
        else:
            if checkpoint is None:
                raise ValueError("Нужен либо checkpoint, либо готовая модель")
            self.model, self.metadata = load_checkpoint(checkpoint)

        self.model.eval().to(self.device)

        # Раскладка памяти меняется ТОЛЬКО у собственной модели. nn.Module.to()
        # перекладывает параметры на месте, поэтому для заимствованной модели это
        # тихо портит обучение: после валидации веса остаются в channels_last,
        # обучение продолжает подавать NCHW, и cuDNN перекладывает память на
        # каждой свёртке. Эпоха после валидации дорожала с 55 до 400 секунд при
        # полной загрузке GPU — снаружи выглядело как нехватка памяти, хотя
        # видеокарта просто перетасовывала байты.
        self._channels_last = self.config.channels_last and self._owns_model
        if self._channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # Рабочий путь идёт по заранее сконвертированным half-весам: это быстрее
        # autocast на batch=1 примерно в полтора раза, а latency batch=1 — половина
        # балла за производительность. Grad-CAM работает с отдельной fp32-копией
        # (см. explain_model): градиенты в half неустойчивы, а конвертировать
        # модель туда-обратно на каждый запрос — это десятки секунд.
        #
        # Пре-каст допустим только для собственной модели. Для заимствованной
        # ускорение даёт autocast: он ничего не меняет в самом объекте.
        self._explain_model: VehicleReID | None = None
        self._fp32_state: dict | None = None
        self._pre_cast = self.config.half and self.device.type == "cuda" and self._owns_model
        self._autocast = self.config.half and self.device.type == "cuda" and not self._owns_model
        if self._pre_cast:
            self._fp32_state = {k: v.detach().clone()
                                for k, v in self.model.state_dict().items()}
            self.model = self.model.half()

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.transform = build_eval_transform(self.config.size)
        self.feature_dim = self.model.feature_dim

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, non_blocking=True)
        if self._channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)

        if self._pre_cast:
            batch = batch.half()

        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=self._autocast):
            if self.config.flip_tta:
                # Отражение по горизонтали — единственная TTA, разрешённая ответом 38
                # (на уровне одного изображения).
                #
                # Оригинал и отражение идут ОДНИМ батчем, а не двумя вызовами.
                # Два отдельных forward оказались втрое дороже ожидаемого: flip
                # возвращает тензор, потерявший раскладку channels_last, и cuDNN
                # уходит на медленный путь с переупаковкой памяти. Склейка с
                # явным восстановлением contiguous(channels_last) убирает это и
                # заодно лучше загружает GPU при batch=1.
                flipped = torch.flip(batch, dims=[3])
                merged = torch.cat([batch, flipped], dim=0)
                if self._channels_last:
                    merged = merged.contiguous(memory_format=torch.channels_last)
                both = self.model(merged)
                features = both[: len(batch)] + both[len(batch):]
            else:
                features = self.model(batch)
        return torch.nn.functional.normalize(features.float(), dim=1)

    def extract(self, rows: list[Observation], progress: bool = True) -> np.ndarray:
        """Матрица (len(rows), feature_dim) в порядке, строго совпадающем с rows."""
        loader = DataLoader(
            CropDataset(rows, self.transform, self.config.size),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )
        output = np.zeros((len(rows), self.feature_dim), dtype=np.float32)
        started = time.perf_counter()
        done = 0
        try:
            for batch, indices in loader:
                output[indices.numpy()] = self._forward(batch).cpu().numpy()
                done += len(indices)
                if progress and done % (self.config.batch_size * 20) < self.config.batch_size:
                    rate = done / max(1e-9, time.perf_counter() - started)
                    print(json.dumps({"extracted": done, "total": len(rows),
                                      "fps": round(rate, 1)}), flush=True)
        finally:
            # persistent_workers держит процессы живыми, пока жив загрузчик.
            # Во время обучения extract() вызывается на каждой валидации, и без
            # явного закрытия воркеры накапливаются десятками процессов.
            shutdown_loader(loader)
        return output

    @property
    def explain_model(self) -> VehicleReID:
        """Копия модели в fp32 для расчёта градиентов (Grad-CAM).

        Создаётся лениво и только если основная модель работает в half: держать
        её постоянно нет смысла, объяснение запрашивается редко. Занимает около
        107 МБ видеопамяти при пиковых 505 МБ на инференсе.
        """
        if self._fp32_state is None:
            return self.model
        if self._explain_model is None:
            model = build_model(num_classes=self.model.num_classes,
                                embedding_dim=self.model.feature_dim,
                                pretrained=False, verbose=False)
            model.load_state_dict(self._fp32_state, strict=True)
            self._explain_model = model.eval().to(self.device)
        return self._explain_model

    def encode_image(self, crop) -> np.ndarray:
        """Эмбеддинг готового кропа ТС (PIL.Image). Точка входа для сервиса."""
        return self._forward(self.transform(crop).unsqueeze(0))[0].cpu().numpy()

    def extract_one(self, row: Observation) -> np.ndarray:
        """Полный цикл на одно ТС — ровно то, что меряется при batch=1."""
        return self.encode_image(load_crop(row, target=self.config.size))

    def benchmark(self, rows: list[Observation], *, latency_runs: int = 300,
                  latency_warmup: int = 50, throughput_seconds: float = 10.0,
                  batch_sizes: tuple[int, ...] = (1, 8, 16, 32)) -> dict:
        """Замер по методике организаторов (ответ 31).

        latency_b1 — медиана полного цикла при batch=1, 300 прогонов после 50
        прогревочных, с синхронизацией CUDA до и после каждого замера.
        throughput — устойчивый FPS, прогон не короче 10 секунд на каждом размере.
        """
        if not rows:
            raise ValueError("Нужны наблюдения для замера")
        synchronize = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)

        for index in range(latency_warmup):
            self.extract_one(rows[index % len(rows)])
        synchronize()

        timings: list[float] = []
        for index in range(latency_runs):
            synchronize()
            started = time.perf_counter()
            self.extract_one(rows[index % len(rows)])
            synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)

        throughput: dict[int, float] = {}
        for batch_size in batch_sizes:
            # Набор повторяется до 40 батчей: на коротком списке итератор
            # перезапускался бы каждые пару шагов, и замер мерил бы не решение,
            # а накладные расходы на обход набора.
            sample = (rows * (1 + 40 * batch_size // max(1, len(rows))))[: 40 * batch_size]
            loader = DataLoader(
                CropDataset(sample, self.transform, self.config.size),
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.device.type == "cuda",
                # Без persistent_workers каждый повторный проход поднимает
                # процессы заново; на Windows это секунды на проход.
                persistent_workers=self.config.num_workers > 0,
                prefetch_factor=4 if self.config.num_workers > 0 else None,
            )

            # Прогрев: первый проход оплачивает запуск воркеров и подбор
            # алгоритмов свёртки, в устойчивый FPS это попадать не должно.
            for batch, _ in loader:
                self._forward(batch)
            synchronize()

            processed = 0
            started = time.perf_counter()
            while time.perf_counter() - started < throughput_seconds:
                for batch, _ in loader:
                    self._forward(batch)
                    processed += len(batch)
                    if time.perf_counter() - started >= throughput_seconds:
                        break
            synchronize()
            elapsed = time.perf_counter() - started
            throughput[batch_size] = processed / elapsed
            shutdown_loader(loader)
            del loader

        peak_vram = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0
        return {
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
            "latency_ms_b1_median": round(statistics.median(timings), 3),
            "latency_ms_b1_p95": round(float(np.percentile(timings, 95)), 3),
            "latency_ms_b1_mean": round(statistics.mean(timings), 3),
            "throughput_fps": {str(k): round(v, 1) for k, v in throughput.items()},
            "best_throughput_fps": round(max(throughput.values()), 1),
            "peak_vram_mb": round(peak_vram, 1),
            "precision": "fp16" if self.config.half else "fp32",
            "flip_tta": self.config.flip_tta,
            "input_size": list(self.config.size),
        }


def save_checkpoint(model: VehicleReID, path: Path | str, metadata: dict | None = None) -> Path:
    """Сохраняет веса вместе с описанием препроцессинга.

    Препроцессинг пишется в чекпоинт намеренно: рассогласование обучения и
    инференса по размеру входа или нормализации — самая незаметная и самая
    дорогая ошибка в ReID-пайплайне.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": VehicleReID.ARCHITECTURE,
        "model": model.state_dict(),
        "num_classes": model.num_classes,
        "feature_dim": model.feature_dim,
        "preprocessing": {"size": list(DEFAULT_SIZE), "normalize": "imagenet", "interpolation": "bicubic"},
        "metadata": metadata or {},
    }, path)
    return path


def load_checkpoint(path: Path | str) -> tuple[VehicleReID, dict]:
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("architecture") != VehicleReID.ARCHITECTURE:
        raise ValueError(
            f"Чекпоинт собран другой архитектурой: {state.get('architecture')!r}, "
            f"ожидается {VehicleReID.ARCHITECTURE!r}"
        )
    model = build_model(num_classes=state.get("num_classes", 0),
                        embedding_dim=state.get("feature_dim", 2048),
                        pretrained=False, verbose=False)
    model.load_state_dict(state["model"], strict=True)
    return model, {**state.get("metadata", {}), "preprocessing": state.get("preprocessing")}
