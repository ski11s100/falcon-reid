"""Сколько видеопамяти и времени стоит шаг обучения при разных размерах батча.

    python scripts/measure_vram.py

Нужен, чтобы подбирать батч под конкретную карту, а не угадывать: печатает пик
выделенной памяти, зарезервированный объём и секунды на шаг для батчей 32-80,
с градиентным чекпоинтингом и без него. На эти числа ссылается комментарий в
falcon/train.py про выбор батча.

Считается на случайных данных: измеряется стоимость самого шага, а не качество.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.losses import ReIDCriterion  # noqa: E402
from falcon.model import build_model  # noqa: E402

CLASSES = 1156          # столько машин в обучающей части локального сплита
EMBEDDING = 2048
BATCHES = (32, 48, 64, 80)


def measure(batch: int, checkpointing: bool, size: int = 256) -> tuple[float, float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(num_classes=CLASSES, embedding_dim=EMBEDDING,
                        pretrained=False, verbose=False).cuda().train()
    model.set_gradient_checkpointing(checkpointing)
    criterion = ReIDCriterion(CLASSES, EMBEDDING, center_weight=0.0005).cuda()
    optimiser = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=3.5e-4)
    scaler = torch.amp.GradScaler("cuda")

    images = torch.randn(batch, 3, size, size, device="cuda")
    labels = torch.arange(batch, device="cuda") // 4
    cameras = torch.arange(batch, device="cuda") % 4

    started = 0.0
    for step in range(4):
        # Первый шаг прогревочный: в него попадает подбор алгоритмов cuDNN.
        if step == 1:
            torch.cuda.synchronize()
            started = time.perf_counter()
        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            features, logits = model(images)
            loss, _ = criterion(features.float(), logits.float(), labels, cameras)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
    torch.cuda.synchronize()

    per_step = (time.perf_counter() - started) / 3
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    del model, criterion, optimiser, images, labels, cameras
    torch.cuda.empty_cache()
    return peak, reserved, per_step


def main() -> None:
    free, total = torch.cuda.mem_get_info()
    print(f"свободно {free / 1e9:.2f} ГБ из {total / 1e9:.2f} ГБ "
          f"(занято другим: {(total - free) / 1e9:.2f} ГБ)\n")
    print(f"{'батч':<6} {'чекпоинтинг':<14} {'пик':<11} {'зарезерв.':<11} {'с/шаг':<10}")
    for checkpointing in (False, True):
        for batch in BATCHES:
            label = "да" if checkpointing else "нет"
            try:
                peak, reserved, per_step = measure(batch, checkpointing)
            except torch.OutOfMemoryError:
                print(f"{batch:<6} {label:<14} не влезает", flush=True)
                torch.cuda.empty_cache()
                continue
            print(f"{batch:<6} {label:<14} {peak:.2f} ГБ{'':<5} {reserved:.2f} ГБ{'':<5} "
                  f"{per_step:<10.2f}", flush=True)


if __name__ == "__main__":
    main()
