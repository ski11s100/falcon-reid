"""PCA-проекция вектора ансамбля, подогнанная на обучающей части данных.

    python scripts/fit_projection.py D:/falcon-data --dim 256

Считает векторы ансамбля из сдачи (без проекции) на ОБУЧАЮЩЕЙ части
локального сплита — тех машинах, на которых модели учились; запросы и галерея
проверки в подгонке не участвуют. По ним находятся главные компоненты, и
первые --dim сохраняются в models/projection.pt (falcon/extract.Projection).

Почему это работает: склейка трёх моделей даёт 3584 признака, заметная часть
которых — шум и дублирование между моделями. Главные компоненты оставляют
направления, по которым машины действительно различаются. Проверка на
отложенной выборке (выбор размерности и цифры — README, раздел 5):
    полный вектор 3584 -> mAP@10 0.7455
    проекция 256     -> mAP@10 0.7564, бутстреп +0.011 [+0.005; +0.017]
Отбеливание (деление на корень собственного значения) проверено и отвергнуто:
оно раздувает шумовые компоненты и роняет mAP@10 на 0.02–0.08.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.data import build_local_split, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, Projection, build_extractor  # noqa: E402
from falcon.submit import SUBMISSION_CHECKPOINTS, SUBMISSION_FLIP_TTA  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA-проекция вектора ансамбля")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        default=[ROOT / p for p in SUBMISSION_CHECKPOINTS])
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "projection.pt")
    args = parser.parse_args()

    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42)
    extractor = build_extractor(args.checkpoints, ExtractorConfig(num_workers=8, threads=True,
                                                                  flip_tta=SUBMISSION_FLIP_TTA))
    vectors = extractor.extract(split.train, progress=False).astype(np.float64)

    mean = vectors.mean(0)
    _, _, vt = np.linalg.svd(vectors - mean, full_matrices=False)
    components = vt[: args.dim].T
    # Средняя длина проекции: back() по ней восстанавливает масштаб полного
    # вектора для Grad-CAM.
    scale = float(np.linalg.norm((vectors - mean) @ components, axis=1).mean())

    projection = Projection(torch.from_numpy(mean.astype(np.float32)),
                            torch.from_numpy(components.astype(np.float32)), scale, metadata={
                                "fitted_on": "обучающая часть локального сплита (split seed 42)",
                                "train_vectors": int(len(vectors)),
                                "dim": args.dim,
                                "checkpoints": {p.name: sha256(p) for p in args.checkpoints},
                            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    projection.save(args.output)
    print(f"{args.output}: {projection.input_dim} -> {projection.output_dim}, "
          f"по {len(vectors)} векторам обучающей части, "
          f"{args.output.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
