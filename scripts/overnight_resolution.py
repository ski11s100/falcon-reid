"""Ночной прогон: дообучение ансамбля на другом размере входа и вердикт.

    python scripts/overnight_resolution.py D:/falcon-data --size 224x320 \
        --epochs 30 --output runs/aspect-224x320

Зачем. Задача просит различать мелкие детали — диски, наклейки, повреждения,
рейлинги. Первая попытка — квадрат 288 вместо 256 — не помогла
(docs/release_choice.json): кузов по-прежнему сжимался в квадрат, а
вычислений стало на 27% больше. Вторая — вход 224x320 (высота x ширина):
машина вдвое шире, чем выше, и так она не сплющивается, а вычислений почти
столько же (14x20 = 280 патчей против 16x16 = 256).

Что делает: каждую модель сдачи дообучает от её нынешнего чекпойнта
(позиционные эмбеддинги ViT переносятся на новую сетку патчей), экспортирует
в fp16, заново подгоняет PCA-проекцию, прогоняет проверки эталонным скриптом
организаторов и закраской номера, калибрует свой порог и в конце сравнивает
результат с нынешней сдачей (scripts/compare_release.py). Сдача не меняется:
всё ложится в --output, утром остаётся прочитать вердикт.
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

MODELS = ("clip-ours", "clip-veri-ours", "resnet-v2-veri")


def stage_arguments(name: str, args) -> list[str]:
    if name.startswith("clip"):
        return ["--architecture", "clip-vit-b16", "--lr", str(args.clip_lr),
                "--backbone-lr", str(args.clip_backbone_lr), "--weight-decay", "1e-4"]
    return ["--architecture", "resnet50-ibn", "--lr", str(args.resnet_lr)]


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
    parser = argparse.ArgumentParser(description="Дообучение ансамбля на другом размере входа")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "res288")
    parser.add_argument("--size", default="288",
                        help="Вход сети: 288 (квадрат) или ВЫСОТАxШИРИНА, например 224x320")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip-lr", type=float, default=3e-5)
    parser.add_argument("--clip-backbone-lr", type=float, default=3e-6)
    parser.add_argument("--resnet-lr", type=float, default=3.5e-5)
    parser.add_argument("--plate-erase", default="0.5")
    parser.add_argument("--extra-train-share", type=float, default=0.0,
                        help="Доля отложенных машин, отданная в обучение; все проверки идут на "
                             "остальных, которых модель не видела")
    parser.add_argument("--quick", action="store_true",
                        help="Только обучение, проекция и сравнение со сдачей — без сверки, "
                             "проверки закраской и калибровки порога")
    args = parser.parse_args()
    share = ["--extra-train-share", str(args.extra_train_share)]

    args.output.mkdir(parents=True, exist_ok=True)
    exported = args.output / "models"
    exported.mkdir(exist_ok=True)
    keep_system_awake()
    started = time.perf_counter()

    checkpoints = []
    for name in MODELS:
        extra = stage_arguments(name, args)
        output = args.output / name
        source = ROOT / "models" / f"{name}.pt"
        command = [sys.executable, "-u", str(ROOT / "falcon" / "train.py"), str(args.dataset),
                   "--output", str(output), "--resume", str(source),
                   "--input-size", args.size, "--epochs", str(args.epochs),
                   "--warmup-epochs", "2", "--seed", "42", "--split-seed", "42",
                   "--eval-every", "5", "--workers", str(args.workers),
                   "--plate-erase", args.plate_erase, *share, *extra]
        if not run(name, command, output / "heartbeat.json", output / "last.pt"):
            sys.exit(f"{name}: не удалось обучить")
        target = exported / f"{name}.pt"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "export_weights.py"),
                        str(output / "best.pt"), str(target)], cwd=ROOT, check=True)
        checkpoints.append(str(target))

    log("PCA-проекция под новые веса")
    projection = args.output / "projection.pt"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fit_projection.py"), str(args.dataset),
                    "--checkpoints", *checkpoints, "--output", str(projection), *share],
                   cwd=ROOT, check=True)

    if args.quick:
        log("сравнение с нынешней сдачей (быстрый режим, порог сдачи)")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_release.py"), str(args.dataset),
                        "--candidate", *checkpoints, "--candidate-projection", str(projection),
                        "--output", str(args.output / "release_choice.json"), *share],
                       cwd=ROOT, check=True)
        log(f"ГОТОВО за {(time.perf_counter() - started) / 3600:.2f} ч")
        return

    log("проверки")
    for script, name in (("verify_official.py", "official_check.json"),
                         ("plate_masking_check.py", "plate_masking_check.json")):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), str(args.dataset),
                        "--checkpoints", *checkpoints, "--projection", str(projection),
                        "--output", str(args.output / name), *share], cwd=ROOT, check=True)

    # Порог зависит от модели: у новых весов шкала сходства своя. Считаем его
    # здесь же, чтобы утром сравнивать варианты по их собственным порогам.
    log("порог отказа под новые веса")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "calibrate_threshold.py"),
                    str(args.dataset), "--checkpoints", *checkpoints,
                    "--projection", str(projection),
                    "--output", str(args.output / "threshold_choice.json")],
                   cwd=ROOT, check=True)
    # Вердикт — тем же скриптом, которым решается смена сдачи: точность с
    # бутстрэпом, балл кандидатов при своём пороге, задержка и пропускная
    # способность. Утром остаётся прочитать release_choice.json.
    log("сравнение с нынешней сдачей")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_release.py"), str(args.dataset),
                    "--candidate", *checkpoints, "--candidate-projection", str(projection),
                    "--candidate-threshold-report", str(args.output / "threshold_choice.json"),
                    "--output", str(args.output / "release_choice.json"), *share],
                   cwd=ROOT, check=True)
    log(f"ГОТОВО за {(time.perf_counter() - started) / 3600:.2f} ч")


if __name__ == "__main__":
    main()
