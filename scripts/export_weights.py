"""Экспорт чекпойнта для сдачи: веса в fp16, вдвое меньше файл.

    python scripts/export_weights.py runs/clip/clip-ours/best.pt models/model-c.pt

Вывод и так идёт в fp16 (ExtractorConfig.half), поэтому хранить веса в fp32
незачем: CLIP ViT-B/16 занимает 347 МБ в fp32 и 174 МБ в fp16. При загрузке
load_state_dict сам приводит тензоры к типу параметров модели (fp32), так что
загрузчик менять не нужно. Целочисленные тензоры (счётчики BatchNorm) не
трогаются. Скрипт сразу сверяет векторы исходного и экспортированного
чекпойнта на нескольких случайных входах.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.extract import load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт весов в fp16")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    state = torch.load(args.source, map_location="cpu", weights_only=True)
    state["model"] = {k: v.half() if v.is_floating_point() else v for k, v in state["model"].items()}
    state.setdefault("metadata", {})["stored_dtype"] = "float16"
    state["metadata"]["exported_from"] = args.source.as_posix()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.target)

    # Сверка: те же входы через исходный и экспортированный чекпойнт. Размер
    # входа берётся из самого чекпойнта: у ViT позиционные эмбеддинги привязаны
    # к числу патчей, и модель, обученная на 288, не примет кадр 256.
    size = tuple(state.get("preprocessing", {}).get("size") or (256, 256))
    torch.manual_seed(0)
    images = torch.randn(8, 3, int(size[0]), int(size[1]))   # (высота, ширина)
    vectors = []
    for path in (args.source, args.target):
        model = load_checkpoint(path)[0].eval()
        with torch.no_grad():
            vectors.append(model(images))  # в режиме eval модель отдаёт нормированный вектор
    cosine = (vectors[0] * vectors[1]).sum(1)
    print(f"{args.target}: {args.target.stat().st_size / 1e6:.1f} МБ "
          f"(было {args.source.stat().st_size / 1e6:.1f}), вход {size[0]}x{size[1]}, "
          f"косинус с исходным: мин {cosine.min():.6f}")


if __name__ == "__main__":
    main()
