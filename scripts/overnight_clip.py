"""Ночной прогон второй архитектуры: CLIP ViT-B/16 и ансамбли с ResNet.

    python scripts/overnight_clip.py D:/falcon-data

Этапы (каждый пропускается, если уже завершён, и продолжается с места обрыва):
  1. clip-ours      — CLIP сразу на конкурсных данных: быстрый результат;
  2. clip-veri      — CLIP на VeRi-776: машинный домен перед нашими данными
                      (для ResNet такое предобучение дало +0.030 mAP@10);
  3. clip-veri-ours — дообучение этапа 2 на конкурсных данных;
  4. перебор ансамблей: CLIP-модели с уже обученными ResNet.

Почему CLIP. Проба без дообучения: mAP@10 0.110 против 0.061 у ResNet50-IBN
и 0.055–0.064 у DINOv2, а по скорости он стоит как одна ResNet. Рецепт — как в
CLIP-ReID: маленький шаг для энкодера (1e-5) и в десять раз больше для головы.

Разбиение у всех моделей одно (split_seed=42), поэтому ансамбли честно
сравниваются на общей отложенной выборке. Задержку здесь не мерим: во время
обучения замер искажён. Её меряют утром на свободной видеокарте.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from overnight import MAX_ATTEMPTS, log, watch  # noqa: E402

from falcon.data import build_local_split, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import Identity, evaluate_ranking, l2_normalize, rank_from_embeddings  # noqa: E402
from falcon.train import keep_system_awake  # noqa: E402

CLIP_ARGS = ["--architecture", "clip-vit-b16", "--lr", "1e-4", "--backbone-lr", "1e-5",
             "--weight-decay", "1e-4", "--warmup-epochs", "5"]


def run_stage(name: str, command: list[str], output: Path) -> Path | None:
    if (output / "last.pt").is_file():
        print(f"уже обучена: {output}", flush=True)
        return output / "best.pt"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"{name}: попытка {attempt}")
        code = watch(subprocess.Popen(command, cwd=ROOT), output / "heartbeat.json")
        if code == 0 and (output / "last.pt").is_file():
            return output / "best.pt"
        if code is not None:
            print(f"{name} завершился с кодом {code}", flush=True)
    print(f"{name}: не удалось за {MAX_ATTEMPTS} попыток", flush=True)
    return None


def train_command(dataset: Path, output: Path, epochs: int, workers: int,
                  extra: list[str]) -> list[str]:
    return [sys.executable, "-u", str(ROOT / "falcon" / "train.py"), str(dataset),
            "--output", str(output), "--epochs", str(epochs), "--seed", "42",
            "--split-seed", "42", "--eval-every", "5", "--workers", str(workers),
            *CLIP_ARGS, *extra]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ночной прогон CLIP ФАЛЬКОН")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--veri-csv", type=Path, default=Path("D:/falcon-external/veri_train.csv"))
    parser.add_argument("--veri-images", type=Path, default=Path("D:/falcon-external/VeRi"))
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "clip")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--veri-epochs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    keep_system_awake()
    started = time.perf_counter()

    models: dict[str, Path] = {}
    ours = run_stage("clip-ours", train_command(args.dataset, args.output / "clip-ours",
                                                args.epochs, args.workers, []),
                     args.output / "clip-ours")
    if ours:
        models["clip-ours"] = ours

    if args.veri_csv.is_file():
        veri = run_stage("clip-veri", train_command(
            args.veri_csv.parent, args.output / "clip-veri", args.veri_epochs, args.workers,
            ["--csv", str(args.veri_csv), "--images", str(args.veri_images)]),
            args.output / "clip-veri")
        if veri:
            tuned = run_stage("clip-veri-ours", train_command(
                args.dataset, args.output / "clip-veri-ours", args.epochs, args.workers,
                ["--resume", str(veri)]), args.output / "clip-veri-ours")
            if tuned:
                models["clip-veri-ours"] = tuned

    resnets = {
        "v1": ROOT / "runs" / "v1" / "model" / "best.pt",
        "v2-veri": ROOT / "runs" / "v2-veri" / "model" / "best.pt",
        "night-a": ROOT / "runs" / "night" / "model-a" / "best.pt",
        "night-b": ROOT / "runs" / "night" / "model-b" / "best.pt",
        "night-c": ROOT / "runs" / "night" / "model-c" / "best.pt",
    }
    pool = {**models, **{k: v for k, v in resnets.items() if v.is_file()}}
    log(f"перебор ансамблей: {sorted(pool)}")

    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42)
    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    ql = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.query}
    gl = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.gallery}

    # Векторы каждой модели считаются один раз; ансамбль — склейка нормированных
    # векторов с равными весами, ровно как в EnsembleExtractor.
    vectors = {}
    for name, path in pool.items():
        extractor = build_extractor([path], ExtractorConfig(batch_size=64, num_workers=args.workers,
                                                            threads=True))
        vectors[name] = (l2_normalize(extractor.extract(split.query, progress=False)),
                         l2_normalize(extractor.extract(split.gallery, progress=False)))
        del extractor
        torch.cuda.empty_cache()

    results = []
    for size in (1, 2, 3):
        for combo in itertools.combinations(sorted(pool), size):
            if size > 1 and not any(name.startswith("clip") for name in combo):
                continue  # ансамбли только из ResNet уже перебраны прошлой ночью
            weight = 1.0 / len(combo)
            q = l2_normalize(np.hstack([vectors[n][0] * weight for n in combo]))
            g = l2_normalize(np.hstack([vectors[n][1] * weight for n in combo]))
            report = evaluate_ranking(rank_from_embeddings(q, g, qids, gids), ql, gl)
            entry = {"models": list(combo), "mAP@10": round(report.mAP, 4),
                     "Rank-1": round(report.rank_1, 4)}
            results.append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)

    results.sort(key=lambda r: -r["mAP@10"])
    summary = {"best": results[:10], "all": results,
               "hours": round((time.perf_counter() - started) / 3600, 2)}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    log("ИТОГ")
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
