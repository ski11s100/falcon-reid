"""Ночной прогон: дообучение ансамбля на разрешении 288 вместо 256.

    python scripts/overnight_resolution.py D:/falcon-data

Зачем. Задача просит различать мелкие детали — диски, наклейки, повреждения,
рейлинги. При 256 пикселях на кроп они занимают единицы пикселей. Разбор ошибок
показывает, что модель путает машины одной модели и окраски, а различают их как
раз мелочи. Больший вход стоит около 27% вычислений: запас по скорости есть
(29.5 мс при лимите 40).

Что делает: каждую модель сдачи дообучает 15 эпох на 288 от её нынешнего
чекпойнта (позиционные эмбеддинги ViT растягиваются на новое число патчей),
экспортирует в fp16, заново подгоняет PCA-проекцию и прогоняет две проверки —
эталонным скриптом организаторов и закраской зоны номера. Ничего в сдаче не
меняется: все результаты ложатся в runs/res288, решение принимается утром по
отчётам.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from overnight import MAX_ATTEMPTS, log, watch  # noqa: E402

from falcon.submit import SUBMISSION_CHECKPOINTS  # noqa: E402
from falcon.train import keep_system_awake  # noqa: E402

CLIP = ["--architecture", "clip-vit-b16", "--lr", "3e-5", "--backbone-lr", "3e-6",
        "--weight-decay", "1e-4"]
RESNET = ["--architecture", "resnet50-ibn", "--lr", "3.5e-5"]
STAGES = {
    "clip-ours": CLIP,
    "clip-veri-ours": CLIP,
    "resnet-v2-veri": RESNET,
}


def run(name: str, command: list[str], heartbeat: Path, done: Path) -> bool:
    if done.is_file():
        print(f"уже готово: {name}", flush=True)
        return True
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"{name}: попытка {attempt}")
        code = watch(subprocess.Popen(command, cwd=ROOT), heartbeat)
        if code == 0 and done.is_file():
            return True
        print(f"{name}: код {code}", flush=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Дообучение ансамбля на 288")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "res288")
    parser.add_argument("--size", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    exported = args.output / "models"
    exported.mkdir(exist_ok=True)
    keep_system_awake()
    started = time.perf_counter()

    checkpoints = []
    for name, extra in STAGES.items():
        output = args.output / name
        source = ROOT / "models" / f"{name}.pt"
        command = [sys.executable, "-u", str(ROOT / "falcon" / "train.py"), str(args.dataset),
                   "--output", str(output), "--resume", str(source),
                   "--input-size", str(args.size), "--epochs", str(args.epochs),
                   "--warmup-epochs", "2", "--seed", "42", "--split-seed", "42",
                   "--eval-every", "5", "--workers", str(args.workers),
                   "--plate-erase", "0.5", *extra]
        if not run(name, command, output / "heartbeat.json", output / "last.pt"):
            sys.exit(f"{name}: не удалось обучить")
        target = exported / f"{name}.pt"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "export_weights.py"),
                        str(output / "best.pt"), str(target)], cwd=ROOT, check=True)
        checkpoints.append(str(target))

    log("PCA-проекция под новые веса")
    projection = args.output / "projection.pt"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fit_projection.py"), str(args.dataset),
                    "--checkpoints", *checkpoints, "--output", str(projection)], cwd=ROOT, check=True)

    log("проверки")
    for script, name in (("verify_official.py", "official_check.json"),
                         ("plate_masking_check.py", "plate_masking_check.json")):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), str(args.dataset),
                        "--checkpoints", *checkpoints, "--projection", str(projection),
                        "--output", str(args.output / name)], cwd=ROOT, check=True)
    log(f"ГОТОВО за {(time.perf_counter() - started) / 3600:.2f} ч")


if __name__ == "__main__":
    main()
