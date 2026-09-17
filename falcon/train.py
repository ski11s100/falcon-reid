"""Обучение ReID-энкодера на конкурсных данных.

Рецепт основан на «Bag of Tricks for Person Re-ID» (Luo et al., 2019), который
остаётся сильнейшим по соотношению «качество / сложность» и переносится на
транспорт практически без изменений:

  * warmup первых 10 эпох    — большой lr по холодным BNNeck и классификатору
                               разрушает предобученный backbone;
  * cosine-затухание lr      — мягкий выход в конце вместо ступенчатых обрывов;
  * label smoothing 0.1      — против переуверенности на 1233 обучающих ID;
  * random erasing 0.5       — устойчивость к перекрытию;
  * BNNeck                   — разводит triplet и классификацию (см. model.py);
  * AMP fp16                 — вдвое меньше памяти, что критично на 6 ГБ VRAM.

Валидация на каждой эпохе считается ОФИЦИАЛЬНОЙ метрикой mAP@10 по локальному
сплиту с open-set запросами. Лучший чекпоинт выбирается по ней, а не по loss.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.data import Observation, PKSampler, audit, build_local_split, read_manifest  # noqa: E402
from falcon.extract import CropDataset, ExtractorConfig, FeatureExtractor, save_checkpoint, _camera_code  # noqa: E402
from falcon.losses import ReIDCriterion  # noqa: E402
from falcon.metrics import Identity, evaluate_ranking, rank_from_embeddings  # noqa: E402
from falcon.model import build_model  # noqa: E402
from falcon.transforms import DEFAULT_SIZE, build_train_transform  # noqa: E402


@dataclass
class TrainConfig:
    epochs: int = 60
    # Стандарт рецепта Bag of Tricks — P=16, K=4, то есть батч 64. При нехватке
    # видеопамяти autoconfigure_batch снижает P автоматически.
    identities_per_batch: int = 16
    shots_per_identity: int = 4
    base_lr: float = 3.5e-4
    warmup_epochs: int = 10
    warmup_factor: float = 0.01
    weight_decay: float = 5e-4
    triplet_weight: float = 1.0
    id_weight: float = 1.0
    center_weight: float = 0.0005
    label_smoothing: float = 0.1
    embedding_dim: int = 2048
    size: tuple[int, int] = DEFAULT_SIZE
    num_workers: int = 4
    amp: bool = True
    seed: int = 42
    eval_every: int = 2
    # На 6 ГБ видеопамяти окупается всегда: батч 64 иначе уходит в
    # вытеснение и замедляется восьмикратно (см. autoconfigure_batch).
    grad_checkpointing: bool = True


def autoconfigure_batch(config: TrainConfig, device: str) -> TrainConfig:
    """Подбирает размер батча под доступную видеопамять.

    Замерено на RTX 3060 Laptop для ResNet50-IBN-a при 256x256 с AMP:
        батч 32 -> 2.8 ГБ,  батч 48 -> 4.2 ГБ,  батч 64 -> 5.5 ГБ.

    Важно для Windows: при нехватке видеопамяти драйвер WDDM не бросает OOM, а
    начинает вытеснять тензоры в системную память. Обучение не падает, но
    замедляется в разы, и причина со стороны выглядит необъяснимой. Поэтому
    размер батча ограничивается заранее, а не по факту ошибки.

    Batch-hard triplet тем лучше, чем больше батч: в маленьком не находится
    по-настоящему трудных пар. Поэтому уменьшать batch стоит в последнюю очередь
    и за счёт числа идентичностей P, а не снимков на идентичность K.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return config

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1e9
    requested = config.identities_per_batch * config.shots_per_identity

    # Коэффициенты измерены на RTX 3060 Laptop, ResNet50-IBN-a, 256x256, AMP
    # (scripts/measure_vram.py). Пик линеен по размеру батча:
    #   без чекпоинтинга  0.085 ГБ/элемент,  с чекпоинтингом  0.037 ГБ/элемент.
    per_item = 0.037 if config.grad_checkpointing else 0.085
    base = 0.35  # веса, состояние Adam и центры классов

    # Резерв аллокатора превышает пик примерно на 15%, плюс закреплённая память
    # загрузчика и рост занятого другими приложениями. Берём 70% свободного:
    # прошлый прогон с оценкой по 85% ушёл в вытеснение и замедлился впятеро.
    budget = free_gb * 0.70 - base
    affordable = max(8, int(budget / per_item))

    if affordable >= requested:
        return config

    # Сначала пробуем спасти батч чекпоинтингом: на 6 ГБ он превращает батч 64
    # из невозможного (5.47 ГБ и 3.42 с/шаг из-за вытеснения) в комфортный
    # (2.37 ГБ и 0.44 с/шаг). Уменьшать батч — крайняя мера: batch-hard triplet
    # ищет трудные пары внутри батча, и на маленьком их просто нет.
    if not config.grad_checkpointing:
        with_checkpointing = max(8, int((free_gb * 0.70 - base) / 0.037))
        if with_checkpointing >= requested:
            config.grad_checkpointing = True
            print(json.dumps({"vram_autoconfig": {
                "free_gb": round(free_gb, 2),
                "batch": requested,
                "action": "включён градиентный чекпоинтинг",
                "reason": "батч сохранён целиком, память втрое меньше, цена около 30% скорости",
            }}, ensure_ascii=False), flush=True)
            return config
        config.grad_checkpointing = True
        affordable = with_checkpointing

    safe_identities = max(4, affordable // config.shots_per_identity)
    print(json.dumps({
        "vram_autoconfig": {
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_bytes / 1e9, 2),
            "requested_batch": requested,
            "reduced_batch": safe_identities * config.shots_per_identity,
            "identities_per_batch": safe_identities,
            "grad_checkpointing": True,
            "hint": "закройте браузер и другие приложения — больший батч заметно "
                    "улучшает batch-hard triplet",
        }
    }, ensure_ascii=False), flush=True)

    config.identities_per_batch = safe_identities
    return config


def build_scheduler(optimizer, config: TrainConfig, steps_per_epoch: int):
    """Warmup + косинусное затухание, посчитанные в шагах, а не в эпохах."""
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    def factor(step: int) -> float:
        if step < warmup_steps:
            alpha = step / max(1, warmup_steps)
            return config.warmup_factor * (1 - alpha) + alpha
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def validate(model, gallery: list[Observation], query: list[Observation],
             config: TrainConfig, device: str) -> dict:
    """Официальная mAP@10 на локальном сплите. Именно по ней выбирается чекпоинт.

    Модель передаётся по ссылке, и экстрактор её не меняет: ускорение внутри даёт
    autocast. Валидация обязана быть побочно-чистой, иначе обучение незаметно
    деградирует от проверки к проверке.
    """
    was_training = model.training
    extractor = FeatureExtractor(
        model=model,
        config=ExtractorConfig(size=config.size, batch_size=48, num_workers=config.num_workers,
                               device=device, half=True, flip_tta=False),
    )
    try:
        gallery_vectors = extractor.extract(gallery, progress=False)
        query_vectors = extractor.extract(query, progress=False)
    finally:
        del extractor
        model.train(was_training)
        # Пики валидации не должны оставаться зарезервированными на следующую
        # эпоху: на 6 ГБ видеопамяти это приводит к вытеснению в системную.
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    gallery_ids = [row.image_id for row in gallery]
    query_ids = [row.image_id for row in query]
    ranking = rank_from_embeddings(query_vectors, gallery_vectors, query_ids, gallery_ids)

    query_labels = {row.image_id: Identity(row.vehicle_id, row.camera_id) for row in query}
    gallery_labels = {row.image_id: Identity(row.vehicle_id, row.camera_id) for row in gallery}
    return evaluate_ranking(ranking, query_labels, gallery_labels).as_dict()


def load_transferred_weights(model, checkpoint: Path, verbose: bool = True) -> dict:
    """Переносит веса предобучения, отбрасывая несовместимые слои.

    Классификатор привязан к числу идентичностей конкретного набора: у VeRi-776
    их 576, у конкурсного train.csv — 1156. Его веса переносить бессмысленно и
    технически невозможно, а вот backbone и BNNeck — именно то, ради чего
    предобучение и делается.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    current = model.state_dict()

    accepted, rejected = {}, []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            accepted[key] = value
        else:
            rejected.append(key)

    model.load_state_dict(accepted, strict=False)
    report = {"transferred": len(accepted), "skipped": rejected,
              "source": str(checkpoint)}
    if verbose:
        print(json.dumps({"transfer": report}, ensure_ascii=False), flush=True)
    if not accepted:
        raise RuntimeError(f"Из {checkpoint} не перенесено ни одного слоя")
    return report


def train(dataset_dir: Path, output_dir: Path, config: TrainConfig,
          device: str = "cuda", resume: Path | None = None,
          csv_path: Path | None = None, images_dir: Path | None = None) -> dict:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    output_dir.mkdir(parents=True, exist_ok=True)
    config = autoconfigure_batch(config, device)

    # Пути можно задать явно: предобучение идёт на стороннем наборе (VeRi-776),
    # где раскладка каталогов своя и отличается от конкурсной.
    manifest = csv_path or (dataset_dir / "train.csv")
    images = images_dir or (dataset_dir / "images")
    rows = read_manifest(manifest, images, require_labels=True)
    dataset_audit = audit(rows)
    print(json.dumps({"audit": dataset_audit}, ensure_ascii=False), flush=True)

    split = build_local_split(rows, seed=config.seed)
    print(json.dumps({"split": split.summary()}, ensure_ascii=False), flush=True)
    (output_dir / "split.json").write_text(json.dumps({
        "train_ids": sorted({r.vehicle_id for r in split.train}),
        "gallery_ids": sorted({r.vehicle_id for r in split.gallery}),
        "query_image_ids": [r.image_id for r in split.query],
        "open_set_query_ids": sorted(split.open_set_query_ids),
        "seed": config.seed,
    }, indent=2), encoding="utf-8")

    labels = sorted({r.vehicle_id for r in split.train})
    label_index = {name: i for i, name in enumerate(labels)}

    # Без resume backbone инициализируется весами ImageNet+IBN. С resume поверх
    # них ложится предобучение на стороннем наборе — ImageNet скачивать не нужно.
    model = build_model(num_classes=len(labels), embedding_dim=config.embedding_dim,
                        pretrained=resume is None)
    if resume is not None:
        load_transferred_weights(model, resume)
    model.set_gradient_checkpointing(config.grad_checkpointing)
    model.to(device)

    criterion = ReIDCriterion(
        num_classes=len(labels), feature_dim=model.feature_dim,
        id_weight=config.id_weight, triplet_weight=config.triplet_weight,
        center_weight=config.center_weight, label_smoothing=config.label_smoothing,
    ).to(device)

    parameters = [{"params": model.parameters()}]
    if criterion.center is not None:
        # У center loss своя, сильно большая скорость обучения — это из статьи.
        parameters.append({"params": criterion.center.parameters(), "lr": 0.5, "weight_decay": 0.0})
    optimizer = torch.optim.Adam(parameters, lr=config.base_lr, weight_decay=config.weight_decay)

    sampler = PKSampler(split.train, identities_per_batch=config.identities_per_batch,
                        shots_per_identity=config.shots_per_identity, seed=config.seed)
    scheduler = build_scheduler(optimizer, config, len(sampler))
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.startswith("cuda"))

    # Один долгоживущий загрузчик на всё обучение: PKSampler отдаёт индексы,
    # воркеры поднимаются один раз, декодирование идёт параллельно forward-проходу.
    train_loader = DataLoader(
        CropDataset(split.train, build_train_transform(config.size), config.size, labels=label_index),
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )

    history: list[dict] = []
    best_map = -1.0
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str), encoding="utf-8")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        running: dict[str, float] = {}

        for step, (images, targets, cameras, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            cameras = cameras.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                features, logits = model(images)
                loss, parts = criterion(features.float(), logits.float(), targets, cameras)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Эпоха {epoch}, шаг {step}: потеря разошлась в NaN/inf")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            # При переполнении fp16 GradScaler пропускает шаг оптимизатора.
            # Расписание двигаем только вместе с реально сделанным шагом, иначе
            # learning rate уползает вперёд относительно обучения.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value

        record = {
            "epoch": epoch,
            "lr": round(scheduler.get_last_lr()[0], 7),
            "seconds": round(time.perf_counter() - epoch_started, 1),
            **{k: round(v / len(sampler), 4) for k, v in running.items()},
        }

        if epoch % config.eval_every == 0 or epoch == config.epochs:
            record["validation"] = validate(model, split.gallery, split.query, config, device)
            current = record["validation"]["mAP@10"]
            if current > best_map:
                best_map = current
                save_checkpoint(model, output_dir / "best.pt", metadata={
                    "epoch": epoch, "validation": record["validation"],
                    "dataset_audit": dataset_audit, "split": split.summary(),
                    "config": asdict(config),
                })
                record["checkpoint"] = "saved"
            model.to(device)

        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    save_checkpoint(model, output_dir / "last.pt", metadata={"epoch": config.epochs})
    return {"best_mAP@10": best_map, "epochs": config.epochs, "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение ReID-энкодера ФАЛЬКОН")
    parser.add_argument("dataset", type=Path, help="Каталог с train.csv и images/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Манифест вместо <dataset>/train.csv")
    parser.add_argument("--images", type=Path, help="Каталог кадров вместо <dataset>/images")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--shots-per-identity", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=2,
                        help="Валидировать каждые N эпох")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        identities_per_batch=args.identities_per_batch,
        shots_per_identity=args.shots_per_identity,
        base_lr=args.lr,
        num_workers=args.workers,
        amp=not args.no_amp,
        seed=args.seed,
        eval_every=args.eval_every,
        grad_checkpointing=not args.no_checkpointing,
    )
    result = train(args.dataset, args.output, config, device=args.device, resume=args.resume,
                   csv_path=args.csv, images_dir=args.images)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
