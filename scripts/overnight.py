"""Ночной прогон: три модели, перебор ансамблей, выбор лучшего.

    python scripts/overnight.py D:/falcon-data

Зачем длиннее. При 60 эпохах рост качества не выходил на плато (0.616 на 44-й
эпохе, 0.639 на 58-й), то есть обучение обрывалось раньше сходимости. Стандартные
рецепты ReID используют 120-140 эпох, и косинусное затухание при большей длине
спускает скорость обучения плавнее.

Зачем три модели. Ансамбль из двух дал +0.017 к лучшей одиночной модели: модели,
обученные по-разному, ошибаются по-разному, и их согласие надёжнее. Третья модель
может добавить ещё, но может и не добавить — это решается измерением, а не верой.

Разнообразие моделей создаётся двумя способами: разная инициализация (ImageNet
против предобучения на VeRi-776) и разное зерно случайности, от которого зависят
порядок данных и аугментации. Разрешение у всех одинаковое: экстрактор применяет
единый препроцессинг ко всем участникам ансамбля.

Каждый этап пропускается, если его результат уже есть, поэтому прогон можно
прервать и запустить заново без потери сделанного.

Живучесть. Каждая модель учится в отдельном процессе под присмотром сторожа.
Если пульс обучения (heartbeat.json) молчит дольше STALL_MINUTES, процесс
снимается и запускается заново, а обучение продолжается с последней эпохи
(state.pt). Так прогон переживает зависание, которое раньше стоило всей ночи.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.data import build_local_split, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import (  # noqa: E402
    Identity,
    evaluate_ranking,
    l2_normalize,
    performance_score,
    rank_from_embeddings,
)
from falcon.train import keep_system_awake  # noqa: E402

# Порог полного балла за задержку — 40 мс (ответ 34). Держим запас: замер идёт
# на нашей видеокарте, а состав ансамбля влияет на время линейно.
LATENCY_BUDGET_MS = 36.0

# Эпоха идёт около минуты, валидация — около двух. Четверть часа тишины
# означает зависание, а не медленную работу.
STALL_MINUTES = 15
MAX_ATTEMPTS = 4


def log(message: str) -> None:
    print(f"\n{'=' * 64}\n[{time.strftime('%H:%M:%S')}] {message}\n{'=' * 64}", flush=True)


def kill_tree(process: subprocess.Popen) -> None:
    """Снимает процесс вместе с воркерами загрузчика."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                       capture_output=True, check=False)
    else:
        process.kill()
    process.wait()


def watch(process: subprocess.Popen, heartbeat: Path) -> int | None:
    """Ждёт завершения. None — если процесс завис и был снят сторожем."""
    started = time.time()
    while True:
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        last_sign = started
        if heartbeat.is_file():
            last_sign = max(started, heartbeat.stat().st_mtime)
        if time.time() - last_sign > STALL_MINUTES * 60:
            print(f"\n[{time.strftime('%H:%M:%S')}] обучение молчит {STALL_MINUTES} мин — "
                  f"снимаю процесс и продолжаю с последней эпохи", flush=True)
            kill_tree(process)
            return None


def train_model(dataset: Path, output: Path, epochs: int, seed: int,
                resume: Path | None, workers: int, device: str) -> Path:
    checkpoint = output / "best.pt"
    # Признак завершённого обучения — last.pt. Одного best.pt мало: он
    # появляется уже на первой валидации, и прерванный прогон выглядел бы
    # законченным.
    if (output / "last.pt").is_file():
        print(f"уже обучена: {checkpoint}", flush=True)
        return checkpoint

    # Разбиение у всех моделей одно (split_seed=42), различается только зерно
    # обучения. Иначе модели видят при обучении машины, проверочные для других,
    # и сравнение ансамблей на общем сплите становится нечестным.
    command = [sys.executable, "-u", str(ROOT / "falcon" / "train.py"), str(dataset),
               "--output", str(output), "--epochs", str(epochs), "--seed", str(seed),
               "--split-seed", "42", "--eval-every", "10", "--workers", str(workers),
               "--device", device]
    if resume is not None:
        command += ["--resume", str(resume)]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"обучение {output.name}: {epochs} эпох, seed={seed}, "
            f"инициализация {'VeRi' if resume else 'ImageNet'}, попытка {attempt}")
        code = watch(subprocess.Popen(command, cwd=ROOT), output / "heartbeat.json")
        if code == 0 and (output / "last.pt").is_file():
            return checkpoint
        if code is not None:
            print(f"обучение {output.name} завершилось с кодом {code}", flush=True)
    raise RuntimeError(f"{output.name}: {MAX_ATTEMPTS} попыток не хватило")


def evaluate_combo(checkpoints: list[Path], split, workers: int, device: str) -> dict:
    """mAP@10 набора моделей на локальном сплите."""
    extractor = build_extractor(
        checkpoints,
        ExtractorConfig(batch_size=64, num_workers=workers, device=device,
                        half=True, flip_tta=True, threads=True),
    )
    try:
        query = l2_normalize(extractor.extract(split.query, progress=False))
        gallery = l2_normalize(extractor.extract(split.gallery, progress=False))
    finally:
        del extractor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    ql = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.query}
    gl = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.gallery}
    report = evaluate_ranking(rank_from_embeddings(query, gallery, qids, gids), ql, gl)
    return report.as_dict()


def measure_latency(checkpoints: list[Path], rows, workers: int, device: str) -> dict:
    extractor = build_extractor(
        checkpoints,
        # Потоки, как в контейнере на стенде: там /dev/shm 64 МБ и процессный
        # загрузчик не используется (см. ThreadedBatchLoader).
        ExtractorConfig(batch_size=32, num_workers=workers, device=device,
                        half=True, flip_tta=True, threads=True),
    )
    try:
        result = extractor.benchmark(rows[:128], latency_runs=120, latency_warmup=30,
                                     throughput_seconds=6.0)
    finally:
        del extractor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    result["scoring"] = performance_score(result["latency_ms_b1_median"],
                                          result["best_throughput_fps"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ночной прогон ФАЛЬКОН")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "night")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--veri", type=Path, default=ROOT / "runs" / "veri-pretrain" / "best.pt")
    parser.add_argument("--extra", type=Path, nargs="*", default=[
        ROOT / "runs" / "v1" / "model" / "best.pt",
        ROOT / "runs" / "v2-veri" / "model" / "best.pt",
    ], help="Уже обученные модели, которые тоже участвуют в переборе ансамблей")
    parser.add_argument("--max-ensemble", type=int, default=3,
                        help="Больше трёх моделей не укладывается в бюджет задержки")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    keep_system_awake()
    started = time.perf_counter()
    veri = args.veri if args.veri.is_file() else None
    if veri is None:
        print("предобучение VeRi не найдено, все модели пойдут от ImageNet", flush=True)

    # Разнообразие: разная инициализация и разное зерно.
    plan = [
        ("model-a", 42, None),
        ("model-b", 7, veri),
        ("model-c", 2026, veri),
    ]
    checkpoints: dict[str, Path] = {}
    for name, seed, resume in plan:
        try:
            checkpoints[name] = train_model(args.dataset, args.output / name, args.epochs,
                                            seed, resume, args.workers, args.device)
        except Exception as exc:
            # Падение одной модели не должно обнулять ночь: остальные обучатся,
            # а ансамбль соберётся из того, что получилось.
            print(f"ОШИБКА при обучении {name}: {exc}", flush=True)

    # Прежние модели обучались на том же разбиении (зерно 42), поэтому их
    # можно честно сравнивать и комбинировать с новыми.
    for path in args.extra:
        if path.is_file():
            checkpoints[path.parent.parent.name] = path

    available = {k: v for k, v in checkpoints.items() if v.is_file()}
    if not available:
        raise SystemExit("ни одной обученной модели")

    log(f"перебор ансамблей из {len(available)} моделей")
    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42)
    queries = read_manifest(args.dataset / "test_query.csv", args.dataset / "images")

    results: list[dict] = []
    names = sorted(available)
    for size in range(1, min(len(names), args.max_ensemble) + 1):
        for combo in itertools.combinations(names, size):
            paths = [available[n] for n in combo]
            metrics = evaluate_combo(paths, split, args.workers, args.device)
            entry = {"models": list(combo), "mAP@10": metrics["mAP@10"],
                     "Rank-1": metrics["Rank-1"]}
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            results.append(entry)
            (args.output / "combinations.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Кандидаты сортируются по качеству, но выбирается первый, укладывающийся в
    # бюджет задержки: ансамбль из трёх моделей может его пробить.
    log("замер задержки кандидатов")
    for entry in sorted(results, key=lambda r: -r["mAP@10"]):
        paths = [available[n] for n in entry["models"]]
        benchmark = measure_latency(paths, queries, args.workers, args.device)
        entry["latency_ms"] = benchmark["latency_ms_b1_median"]
        entry["fps"] = benchmark["best_throughput_fps"]
        entry["performance_points"] = benchmark["scoring"]["performance_points_of_20"]
        print(json.dumps({k: entry[k] for k in
                          ("models", "mAP@10", "latency_ms", "fps", "performance_points")},
                         ensure_ascii=False), flush=True)
        if entry["latency_ms"] <= LATENCY_BUDGET_MS:
            entry["selected"] = True
            break

    chosen = next((e for e in results if e.get("selected")),
                  max(results, key=lambda r: r["mAP@10"]))
    summary = {
        "chosen": chosen,
        "all": sorted(results, key=lambda r: -r["mAP@10"]),
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "hours": round((time.perf_counter() - started) / 3600, 2),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("ИТОГ НОЧИ")
    print(json.dumps(summary["chosen"], ensure_ascii=False, indent=2))
    print(f"\nдальше: скопировать выбранные веса в models/ и пересобрать сдачу\n"
          f"  python scripts/run_pipeline.py {args.dataset} --output runs/final "
          f"--checkpoints " + " ".join(str(available[n]) for n in chosen["models"]))


if __name__ == "__main__":
    main()
