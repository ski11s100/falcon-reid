"""Дообучение ансамбля из сдачи с закраской зоны номера (PlateZoneErase).

    python scripts/finetune_plate_erase.py D:/falcon-data

Зачем. Проверка scripts/plate_masking_check.py показала, что закраска зоны
номера роняет mAP@10 сдачи на 0.019, чуть сильнее худшей контрольной зоны той
же площади (0.017). Это не больше, чем у прежних моделей, но на контрольном
тесте организаторов (ответ 48) лучше иметь запас. Если при обучении зона
номера то и дело закрашена, модели выгоднее опираться на кузов.

Каждая модель сдачи дообучается от своего лучшего чекпойнта 15 эпох с малым
шагом и закраской в половине кадров. Затем для нового ансамбля считаются
метрики эталонным скриптом (verify_official.py) и та же проверка закраской
(plate_masking_check.py). Сравнение с текущей сдачей — по этим отчётам.
Каждый этап пропускается, если уже завершён; обрыв и зависание перезапускаются.
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

from falcon.train import keep_system_awake  # noqa: E402

CLIP = ["--architecture", "clip-vit-b16", "--lr", "3e-5", "--backbone-lr", "3e-6",
        "--weight-decay", "1e-4"]
RESNET = ["--architecture", "resnet50-ibn", "--lr", "3.5e-5"]
STAGES = {
    "clip-ours": (ROOT / "runs" / "clip" / "clip-ours" / "best.pt", CLIP),
    "clip-veri-ours": (ROOT / "runs" / "clip" / "clip-veri-ours" / "best.pt", CLIP),
    "resnet-v2-veri": (ROOT / "runs" / "v2-veri" / "model" / "best.pt", RESNET),
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
    parser = argparse.ArgumentParser(description="Дообучение с закраской зоны номера")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "plate")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--plate-erase", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    keep_system_awake()
    started = time.perf_counter()

    checkpoints = []
    for name, (source, extra) in STAGES.items():
        output = args.output / name
        command = [sys.executable, "-u", str(ROOT / "falcon" / "train.py"), str(args.dataset),
                   "--output", str(output), "--resume", str(source), "--epochs", str(args.epochs),
                   "--warmup-epochs", "2", "--seed", "42", "--split-seed", "42", "--eval-every", "5",
                   "--workers", str(args.workers), "--plate-erase", str(args.plate_erase), *extra]
        if not run(name, command, output / "heartbeat.json", output / "last.pt"):
            sys.exit(f"{name}: не удалось обучить")
        checkpoints.append(output / "best.pt")

    log("проверка нового ансамбля")
    paths = [str(p) for p in checkpoints]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_official.py"), str(args.dataset),
                    "--checkpoints", *paths, "--output", str(args.output / "official_check.json")],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plate_masking_check.py"), str(args.dataset),
                    "--checkpoints", *paths, "--output", str(args.output / "plate_masking_check.json")],
                   cwd=ROOT, check=True)
    log(f"ГОТОВО за {(time.perf_counter() - started) / 3600:.2f} ч")


if __name__ == "__main__":
    main()
