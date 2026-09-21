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
import shutil
import statistics
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import Observation, load_crop
from .model import VehicleReID, build_for_checkpoint
from .transforms import DEFAULT_SIZE, build_eval_transform


def shared_memory_mb() -> float | None:
    """Сколько разделяемой памяти доступно. None, если раздела нет (Windows)."""
    try:
        return shutil.disk_usage("/dev/shm").total / 1024 / 1024
    except (FileNotFoundError, OSError):
        return None


# Порога в 256 МБ хватает на батч 64 при 256x256 с запасом на префетч.
SHM_MB = shared_memory_mb()
USE_THREADS = SHM_MB is not None and SHM_MB < 256


class ThreadedBatchLoader:
    """Загрузчик на потоках вместо процессов. Не требует разделяемой памяти.

    Зачем. DataLoader передаёт готовые тензоры между процессами через /dev/shm.
    Контейнеру Docker этот раздел выдаётся размером 64 МБ, чего не хватает даже
    на один батч 256x256, и прогон падает с «No space left on device» посреди
    извлечения признаков. Стратегия file_system эту проблему не решает: она
    использует тот же раздел.

    Полагаться на флаг --shm-size нельзя: команду запуска задают организаторы
    (ответ 40), и требовать от них дополнительных параметров мы не можем.

    Почему потоки здесь работают. Узкое место этой задачи — декодирование JPEG
    кадра 1920x1080, а Pillow на время декодирования освобождает GIL. Поэтому
    потоки дают настоящую параллельность, а тензоры остаются в общей памяти
    процесса, и /dev/shm не участвует вовсе.
    """

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int,
                 prefetch_batches: int = 3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = max(1, num_workers)
        self.prefetch_batches = max(1, prefetch_batches)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        from concurrent.futures import ThreadPoolExecutor
        from torch.utils.data._utils.collate import default_collate

        total = len(self.dataset)
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            pending: list = []
            next_index = 0
            # Окно префетча ограничено, иначе в память уедет весь набор сразу.
            window = self.prefetch_batches * self.batch_size

            while next_index < total or pending:
                while next_index < total and len(pending) < window:
                    pending.append(pool.submit(self.dataset.__getitem__, next_index))
                    next_index += 1

                take = min(self.batch_size, len(pending))
                batch = [future.result() for future in pending[:take]]
                del pending[:take]
                yield default_collate(batch)


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool,
                 threads: bool = False):
    """DataLoader на процессах, либо потоковый вариант.

    Потоки берутся при малом /dev/shm (Docker) или по явной просьбе `threads`.
    Просит о них валидация внутри обучения: запуск новых процессов посреди
    многочасового прогона — единственное место, где он может застрять насовсем.
    На Windows дочерний процесс создаётся через spawn, и родитель передаёт ему
    данные по каналу. Если потомок не стартовал (так было, когда ноутбук ушёл
    в режим ожидания), родитель вечно ждёт в multiprocessing.reduction.dump —
    это показал снимок стека py-spy зависшего прогона.
    """
    if (USE_THREADS or threads) and num_workers > 0:
        return ThreadedBatchLoader(dataset, batch_size, num_workers)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=False,
        prefetch_factor=4 if num_workers > 0 else None,
    )


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


def shutdown_loader(loader) -> None:
    """Освобождает воркеров разового загрузчика.

    Реализация намеренно НЕ трогает приватный `_shutdown_workers()`. Первая
    версия вызывала его напрямую и на Windows приводила к взаимоблоку: все
    процессы вставали с нулевым потреблением CPU, прогон замирал на валидации
    и не завершался часами. Метод приватный, рассчитан на вызов из `__del__`
    и не обязан быть безопасным в произвольный момент.

    Достаточно снять ссылку на итератор: загрузчики здесь создаются без
    persistent_workers, поэтому воркеры завершаются штатно по исчерпании.
    Потоковый загрузчик закрывать не нужно — его пул живёт внутри __iter__.
    """
    if isinstance(loader, DataLoader):
        loader._iterator = None


def _camera_code(camera_id: str | None) -> int:
    """Камера как целое: нужна triplet-потере для отбора кросс-камерных пар.

    Для нечисловых идентификаторов берётся CRC32, а НЕ встроенный hash().
    Питоновский hash() для строк рандомизируется на каждый процесс, а батчи
    собираются в процессах-воркерах DataLoader: одна и та же камера получала бы
    в разных воркерах разные коды. Маска кросс-камерных пар в triplet-потере
    превращалась бы в шум — молча, без единой ошибки. Конкурсные camera_id
    числовые и не задеты, а вот у VeRi они вида "veri-c1".
    """
    if camera_id is None:
        return -1
    try:
        return int(camera_id)
    except ValueError:
        return zlib.crc32(camera_id.encode("utf-8"))


@dataclass
class ExtractorConfig:
    size: tuple[int, int] = DEFAULT_SIZE
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda"
    half: bool = True
    flip_tta: bool = True
    channels_last: bool = True
    # Загрузка кадров потоками, без порождения процессов (см. build_loader).
    threads: bool = False
    # CUDA Graphs для batch=1 и 16 (см. FeatureExtractor._replay).
    cuda_graphs: bool = True


# Формы батча, для которых записывается CUDA Graph: 1 — замер latency и
# поиск в сервисе, 16 — пакетная регистрация (сервис дополняет пачку до 16).
GRAPH_BATCHES = (1, 16)


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
            # Препроцессинг должен совпадать с обучением до пикселя, поэтому
            # размер кропа берётся из чекпойнта, а не из настроек по умолчанию.
            trained_size = (self.metadata.get("preprocessing") or {}).get("size")
            if trained_size and tuple(trained_size) != tuple(self.config.size):
                self.config = replace(self.config, size=tuple(trained_size))

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
        self._graphs: dict = {}
        self._use_graphs = (self.config.cuda_graphs and self.device.type == "cuda"
                            and self._owns_model)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, non_blocking=True)
        if self._channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)

        if self._pre_cast:
            batch = batch.half()

        if self._use_graphs and batch.shape[0] in GRAPH_BATCHES:
            return self._replay(batch)
        return self._compute(batch)

    def _replay(self, batch: torch.Tensor) -> torch.Tensor:
        """Прямой проход через CUDA Graph: вся цепочка ядер запускается одной командой.

        При batch=1 модель упирается не в видеокарту, а в процессор: сотни
        мелких ядер, и на запуск каждого уходят микросекунды. ResNet50-IBN при
        batch=1 шла 10 мс, из них большая часть — запуск ядер. Граф записывается
        один раз на каждую форму входа и дальше воспроизводится целиком. Это
        особенно важно на стенде жюри: у Xeon Gold 6338 на 2.0 ГГц запуск ядер
        медленнее, чем у ноутбучного процессора, на котором мы меряем.
        """
        key = (tuple(batch.shape), batch.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            static_input = batch.clone()
            # Прогрев на отдельном потоке: cuDNN выбирает алгоритмы до записи.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._compute(static_input)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = self._compute(static_input)
            entry = self._graphs[key] = (graph, static_input, static_output)
        graph, static_input, static_output = entry
        static_input.copy_(batch)
        graph.replay()
        return static_output.clone()

    def _compute(self, batch: torch.Tensor) -> torch.Tensor:
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
        loader = build_loader(
            CropDataset(rows, self.transform, self.config.size),
            self.config.batch_size, self.config.num_workers,
            pin_memory=self.device.type == "cuda", threads=self.config.threads,
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
            model = build_for_checkpoint(self.model.ARCHITECTURE, self.model.num_classes,
                                         self.model.feature_dim)
            model.load_state_dict(self._fp32_state, strict=True)
            self._explain_model = model.eval().to(self.device)
        return self._explain_model

    def encode_image(self, crop) -> np.ndarray:
        """Эмбеддинг готового кропа ТС (PIL.Image). Точка входа для сервиса."""
        return self._forward(self.transform(crop).unsqueeze(0))[0].cpu().numpy()

    def encode_images(self, crops, batch_size: int = 16) -> np.ndarray:
        """Эмбеддинги нескольких кропов одним батчем на порцию.

        Для наполнения галереи пачкой: 20 снимков за один проход сети вместо
        двадцати проходов по одному.
        """
        return encode_in_batches(self, crops, batch_size)

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
            loader = build_loader(
                CropDataset(sample, self.transform, self.config.size),
                batch_size, self.config.num_workers,
                pin_memory=self.device.type == "cuda", threads=self.config.threads,
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



class Projection:
    """PCA-проекция склеенного вектора ансамбля в пространство меньшей размерности.

    Главные компоненты считаются по векторам ОБУЧАЮЩЕЙ части данных
    (scripts/fit_projection.py): ни запросы, ни галерея проверки в подгонке не
    участвуют. Проекция делает две вещи сразу:
      * отбрасывает шумовые направления склейки трёх моделей — mAP@10 на
        отложенной выборке растёт на 0.011 (интервал [+0.005; +0.017]);
      * сжимает вектор 3584 -> 256: миллион объектов занимает 1 ГБ вместо
        13.4 ГБ, и pgvector может строить индекс HNSW (он работает до 2000
        измерений).
    back() восстанавливает приближение полного вектора: оно нужно Grad-CAM,
    чтобы разделить эталон на части моделей ансамбля.
    """

    def __init__(self, mean: torch.Tensor, components: torch.Tensor, scale: float,
                 metadata: dict | None = None):
        self.mean = mean.float()
        self.components = components.float()
        self.scale = float(scale)
        self.metadata = metadata or {}

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def load(cls, path: Path | str) -> Projection:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(state["mean"], state["components"], state["scale"], state.get("metadata"))

    def save(self, path: Path | str) -> None:
        torch.save({"mean": self.mean, "components": self.components, "scale": self.scale,
                    "metadata": self.metadata}, path)

    def to(self, device) -> Projection:
        self.mean = self.mean.to(device)
        self.components = self.components.to(device)
        return self

    def apply(self, vectors: torch.Tensor) -> torch.Tensor:
        projected = (vectors.float() - self.mean) @ self.components
        return torch.nn.functional.normalize(projected, dim=1)

    def back(self, projected: np.ndarray) -> np.ndarray:
        components = self.components.cpu().numpy()
        return self.mean.cpu().numpy() + self.scale * (components @ np.asarray(projected, dtype=np.float32))


class EnsembleExtractor:
    """Несколько моделей как один экстрактор.

    Векторы моделей склеиваются с весами и нормируются целиком. Для косинусной
    близости это эквивалентно взвешенной сумме косинусов отдельных моделей, но
    не требует менять ни поиск, ни формат сдачи: наружу по-прежнему выходит один
    вектор на объект.

    Зачем: две независимо обученные модели ошибаются по-разному, и их согласие
    надёжнее, чем уверенность любой из них. Замер на локальном сплите:
    0.6339 и 0.6476 по отдельности против 0.6716 вместе.

    Цена — второй проход модели. Она оказалась мала, потому что в замеряемом
    организаторами цикле преобладает декодирование кадра 1920x1080, а не сеть:
    31.7 мс против 21.9 мс у одиночной модели при бюджете 40 мс.
    """

    def __init__(self, checkpoints: list[Path | str], weights: list[float] | None = None,
                 config: ExtractorConfig | None = None, projection: Path | str | None = None):
        if not checkpoints:
            raise ValueError("Нужен хотя бы один чекпоинт")
        self.members = [FeatureExtractor(checkpoint=c, config=config) for c in checkpoints]
        self.weights = weights or [1.0 / len(self.members)] * len(self.members)
        if len(self.weights) != len(self.members):
            raise ValueError("Число весов не совпадает с числом моделей")

        sizes = {tuple(m.config.size) for m in self.members}
        if len(sizes) > 1:
            raise ValueError(
                f"Модели ансамбля обучены на разных размерах входа: {sorted(sizes)}. "
                "Кадр читается один раз на все модели, поэтому размер должен совпадать.")

        first = self.members[0]
        self.config = first.config
        self.device = first.device
        self.transform = first.transform
        self.feature_dim = sum(m.feature_dim for m in self.members)
        self.metadata = {
            "architecture": " + ".join(m.model.ARCHITECTURE for m in self.members),
            "ensemble": [str(c) for c in checkpoints],
            "weights": self.weights,
        }
        self.projection = None
        if projection is not None:
            self.projection = Projection.load(projection)
            if self.projection.input_dim != self.feature_dim:
                raise ValueError(
                    f"Проекция {projection} рассчитана на вектор {self.projection.input_dim}, "
                    f"а ансамбль выдаёт {self.feature_dim}: её подгоняли под другой состав моделей")
            self.projection.to(self.device)
            self.metadata["projection"] = {"path": str(projection), "dim": self.projection.output_dim}
            self.feature_dim = self.projection.output_dim

    @property
    def model(self):
        return self.members[0].model

    @property
    def explain_model(self) -> VehicleReID:
        """Объяснение строится по первой модели: Grad-CAM не суммируется."""
        return self.members[0].explain_model

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Взвешенная склейка векторов всех моделей, нормированная целиком.

        Тот же контракт, что у FeatureExtractor._forward, поэтому замер
        производительности и всё остальное работает с ансамблем без изменений.
        Раньше метод отсутствовал, и benchmark падал с AttributeError на
        последнем шаге конвейера.
        """
        parts = [m._forward(batch) * w for m, w in zip(self.members, self.weights)]
        joined = torch.nn.functional.normalize(torch.cat(parts, dim=1), dim=1)
        return self.projection.apply(joined) if self.projection is not None else joined

    def encode_image(self, crop) -> np.ndarray:
        tensor = self.transform(crop).unsqueeze(0)
        return self._forward(tensor)[0].cpu().numpy()

    def encode_images(self, crops, batch_size: int = 16) -> np.ndarray:
        return encode_in_batches(self, crops, batch_size)

    def extract_one(self, row: Observation) -> np.ndarray:
        return self.encode_image(load_crop(row, target=self.config.size))

    def extract(self, rows: list[Observation], progress: bool = True) -> np.ndarray:
        # Кадры читаются и декодируются ОДИН раз на все модели: это самая дорогая
        # часть цикла, и дублировать её было бы прямой потерей скорости.
        loader = build_loader(
            CropDataset(rows, self.transform, self.config.size),
            self.config.batch_size, self.config.num_workers,
            pin_memory=self.device.type == "cuda", threads=self.config.threads,
        )
        output = np.zeros((len(rows), self.feature_dim), dtype=np.float32)
        done = 0
        started = time.perf_counter()
        try:
            for batch, indices in loader:
                output[indices.numpy()] = self._forward(batch).cpu().numpy()
                done += len(indices)
                if progress and done % (self.config.batch_size * 20) < self.config.batch_size:
                    print(json.dumps({"extracted": done, "total": len(rows),
                                      "fps": round(done / max(1e-9, time.perf_counter() - started), 1)}),
                          flush=True)
        finally:
            shutdown_loader(loader)
        return output

    benchmark = FeatureExtractor.benchmark


def build_extractor(checkpoints: list[Path | str], config: ExtractorConfig | None = None,
                    weights: list[float] | None = None, projection: Path | str | None = None):
    """Один чекпоинт — обычный экстрактор, несколько — ансамбль.

    projection — файл PCA-проекции склеенного вектора (Projection); с ней
    экстрактор всегда собирается как ансамбль, даже из одной модели.
    """
    if len(checkpoints) == 1 and projection is None:
        return FeatureExtractor(checkpoint=checkpoints[0], config=config)
    return EnsembleExtractor(checkpoints, weights=weights, config=config, projection=projection)

def save_checkpoint(model: VehicleReID, path: Path | str, metadata: dict | None = None,
                    size: tuple[int, int] = DEFAULT_SIZE) -> Path:
    """Сохраняет веса вместе с описанием препроцессинга.

    Препроцессинг пишется в чекпоинт намеренно: рассогласование обучения и
    инференса по размеру входа или нормализации — самая незаметная и самая
    дорогая ошибка в ReID-пайплайне.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": model.ARCHITECTURE,
        "model": model.state_dict(),
        "num_classes": model.num_classes,
        "feature_dim": model.feature_dim,
        "preprocessing": {"size": list(size), "normalize": "imagenet", "interpolation": "bicubic"},
        "metadata": metadata or {},
    }, path)
    return path


def encode_in_batches(extractor, crops, batch_size: int) -> np.ndarray:
    """Кропы -> эмбеддинги батчами ОДНОГО размера.

    Неполный батч дополняется копиями последнего кропа, лишние строки потом
    отбрасываются. Зачем: cudnn.benchmark подбирает алгоритмы свёртки заново
    под каждую новую форму входа — около 6 секунд на форму. Пачки из 20 снимков
    давали формы 16 и 4, и первая загрузка галереи стояла 13 секунд. С
    выравниванием форма одна, и её прогревают при старте сервиса.
    """
    if not crops:
        return np.zeros((0, extractor.feature_dim), dtype=np.float32)
    parts = []
    for start in range(0, len(crops), batch_size):
        tensors = [extractor.transform(c) for c in crops[start:start + batch_size]]
        real = len(tensors)
        tensors += [tensors[-1]] * (batch_size - real)
        parts.append(extractor._forward(torch.stack(tensors)).cpu().numpy()[:real])
    return np.concatenate(parts)


LFS_POINTER_PREFIX = b"version https://git-lfs"


def load_checkpoint(path: Path | str) -> tuple[VehicleReID, dict]:
    path = Path(path)
    # Без Git LFS (или из ZIP-архива GitHub) вместо весов приходит текстовый
    # указатель на 134 байта. torch.load падает на нём с невнятной ошибкой
    # распаковки, поэтому распознаём его сами и говорим, что делать.
    with path.open("rb") as handle:
        if handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            raise RuntimeError(
                f"{path} — указатель Git LFS, а не веса модели. "
                f"Установите Git LFS и выполните: git lfs install && git lfs pull. "
                f"Если квота Git LFS исчерпана: python scripts/fetch_weights.py"
            )
    # weights_only=True: распаковываются только тензоры и простые типы. Полный
    # pickle (weights_only=False) выполнил бы код из подложенного файла весов,
    # то есть файл весов был бы готовым вектором атаки на сервер.
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Архитектура записана в чекпоинт; неизвестная — ошибка с понятным текстом.
    # Размер входа берётся из самого чекпойнта: у ViT позиционные эмбеддинги
    # привязаны к числу патчей, и модель, обученная на 288, не соберётся под 256.
    size = tuple(state.get("preprocessing", {}).get("size") or DEFAULT_SIZE)
    model = build_for_checkpoint(state.get("architecture"), state.get("num_classes", 0),
                                 state.get("feature_dim", 2048), img_size=(int(size[0]), int(size[1])))
    model.load_state_dict(state["model"], strict=True)
    return model, {**state.get("metadata", {}),
                   "preprocessing": state.get("preprocessing", {"size": list(size)})}
