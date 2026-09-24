"""Живой индикатор хода обучения.

Запуск (в отдельном окне терминала, обучению не мешает):

    .\\.venv\\Scripts\\python.exe scripts\\progress.py

Скрипт только читает файлы и показывает состояние. Сам находит активный прогон,
рисует прогресс по эпохам и внутри эпохи, считает оставшееся время, показывает
динамику качества и отдельно предупреждает, если обучение зависло.

Выход — Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

# Windows-терминал понимает ANSI только после явного включения.
if os.name == "nt":
    os.system("")

CLEAR = "\033[2J\033[H"
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
RED, GREEN, YELLOW, CYAN, GREY = ("\033[31m", "\033[32m", "\033[33m",
                                  "\033[36m", "\033[90m")


def human_time(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} с"
    if seconds < 3600:
        return f"{seconds // 60} мин {seconds % 60:02d} с"
    return f"{seconds // 3600} ч {(seconds % 3600) // 60:02d} мин"


def bar(fraction: float, width: int, colour: str = CYAN) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return f"{colour}{'█' * filled}{GREY}{'░' * (width - filled)}{RESET}"


def sparkline(values: list[float]) -> str:
    """Компактный график динамики качества."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    low, high = min(values), max(values)
    span = high - low
    if span < 1e-9:
        return blocks[len(blocks) // 2] * len(values)
    return "".join(blocks[int((v - low) / span * (len(blocks) - 1))] for v in values)


def find_active_run(root: Path) -> Path | None:
    """Самый свежеобновлённый каталог прогона."""
    candidates: list[tuple[float, Path]] = []
    for base in (root / "runs", Path("D:/falcon-cache/runs")):
        if not base.is_dir():
            continue
        for folder in base.rglob("*"):
            if folder.name in ("history.json", "heartbeat.json"):
                candidates.append((folder.stat().st_mtime, folder.parent))
    if not candidates:
        return None
    return max(candidates)[1]


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def gpu_state() -> dict:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
        return {"util": int(output[0]), "used": int(output[1]),
                "total": int(output[2]), "temp": int(output[3])}
    except Exception:
        return {}


def python_cpu_busy() -> bool:
    """Тратят ли процессы python процессорное время. Ноль означает зависание."""
    if os.name != "nt":
        return True
    script = ("$a=(Get-Process python -EA SilentlyContinue | Measure-Object CPU -Sum).Sum;"
              "Start-Sleep -Milliseconds 1200;"
              "$b=(Get-Process python -EA SilentlyContinue | Measure-Object CPU -Sum).Sum;"
              "[math]::Round($b-$a,2)")
    try:
        value = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                               capture_output=True, text=True, timeout=12).stdout.strip()
        return float(value.replace(",", ".")) > 0.05
    except Exception:
        return True


NIGHT_MODELS = ("clip-ours", "clip-veri-ours", "resnet-v2-veri")
# Этапы после обучения и признак готовности каждого: файл, который он пишет.
NIGHT_STEPS = (("экспорт весов в fp16", "models/resnet-v2-veri.pt"),
               ("PCA-проекция", "projection.pt"),
               ("сверка эталонным скриптом", "official_check.json"),
               ("проверка закраской номера", "plate_masking_check.json"),
               ("порог отказа", "threshold_choice.json"),
               ("сравнение со сдачей", "release_choice.json"))
STEPS_SECONDS = 25 * 60        # проверки после обучения, по прошлым ночам
# Обучение на всех размеченных машинах: проверять не на чем, после обучения
# только сдача на публичном тесте и её сверка с нынешней.
FULL_DATA_STEPS = (("экспорт весов в fp16", "models/resnet-v2-veri.pt"),
                   ("PCA-проекция", "projection.pt"),
                   ("сдача на публичном тесте", "submission/manifest.json"),
                   ("сверка с нынешней сдачей", "submission_agreement.json"))
FULL_DATA_STEPS_SECONDS = 8 * 60


def night_steps(night: Path) -> tuple[tuple, int]:
    config = read_json(night / NIGHT_MODELS[0] / "config.json") or {}
    if (config.get("extra_train_share") or 0) >= 1:
        return FULL_DATA_STEPS, FULL_DATA_STEPS_SECONDS
    return NIGHT_STEPS, STEPS_SECONDS


def night_root(run: Path) -> Path | None:
    """Каталог ночного прогона, если run — одна из его моделей."""
    if run.name in NIGHT_MODELS and sum((run.parent / m).is_dir() for m in NIGHT_MODELS) >= 1:
        return run.parent
    return None


def model_state(folder: Path) -> dict:
    history = read_json(folder / "history.json") or []
    heartbeat = read_json(folder / "heartbeat.json") or {}
    config = read_json(folder / "config.json") or {}
    total = config.get("epochs") or heartbeat.get("epochs_total") or 0
    scored = [r["validation"]["mAP@10"] for r in history if r.get("validation")]
    seconds = sorted(r["seconds"] for r in history)
    return {"done": (folder / "last.pt").is_file(), "epochs": len(history), "total": total,
            "best": max(scored) if scored else None, "last": scored[-1] if scored else None,
            "epoch_seconds": seconds[len(seconds) // 2] if seconds else None,
            "started": folder.is_dir() and bool(history or heartbeat),
            "architecture": config.get("architecture", "")}


def render_night(night: Path, width: int) -> str:
    """Сводка всей ночи: три модели, проверки, время до конца и вердикт."""
    states = {name: model_state(night / name) for name in NIGHT_MODELS}
    steps, steps_seconds = night_steps(night)
    lines = [f"{BOLD}ФАЛЬКОН — ночное обучение{RESET}   {GREY}{time.strftime('%H:%M:%S')}{RESET}",
             f"{GREY}{night}{RESET}", ""]

    clip_epoch = next((s["epoch_seconds"] for s in states.values()
                       if s["epoch_seconds"] and "clip" in s["architecture"]), 95.0)
    remaining = 0.0
    units_done, units_total = 0.0, 0.0
    for name, state in states.items():
        total = state["total"] or 30
        per_epoch = state["epoch_seconds"] or (clip_epoch * (0.6 if "resnet" in name else 1.0))
        done_epochs = total if state["done"] else state["epochs"]
        units_done += done_epochs * per_epoch
        units_total += total * per_epoch
        remaining += (total - done_epochs) * per_epoch
    steps_done = sum((night / marker).is_file() for _, marker in steps)
    units_total += steps_seconds
    units_done += steps_seconds * steps_done / len(steps)
    remaining += steps_seconds * (1 - steps_done / len(steps))
    overall = units_done / units_total if units_total else 0.0
    finish = time.strftime("%H:%M", time.localtime(time.time() + remaining))
    lines.append(f"{BOLD}Вся ночь{RESET}   {bar(overall, width - 28, GREEN)} {overall * 100:5.1f}%")
    lines.append(f"{GREY}осталось примерно {human_time(remaining)}, закончится около {finish}{RESET}")
    lines.append("")

    for name, state in states.items():
        if state["done"]:
            mark, colour, text = "✓", GREEN, "готова"
        elif state["started"]:
            mark, colour, text = "▶", YELLOW, f"эпоха {state['epochs'] + 1}/{state['total'] or '?'}"
        else:
            mark, colour, text = "·", GREY, "ждёт очереди"
        quality = f"   mAP@10 лучшее {state['best']:.4f}" if state["best"] is not None else ""
        lines.append(f"  {colour}{mark} {name:<16}{RESET} {text}{GREEN}{quality}{RESET}")
    lines.append("")
    for title, marker in steps:
        ready = (night / marker).is_file()
        lines.append(f"  {GREEN + '✓' if ready else GREY + '·'} {title}{RESET}")

    verdict = read_json(night / "release_choice.json")
    if verdict:
        colour = GREEN if verdict.get("вердикт") == "менять сдачу" else YELLOW
        candidate = verdict.get("кандидат", {})
        current = verdict.get("сдача", {})
        lines += ["", f"{BOLD}Вердикт:{RESET} {colour}{BOLD}{verdict.get('вердикт')}{RESET}",
                  f"  mAP@10 {candidate.get('mAP@10')} против {current.get('mAP@10')} у сдачи, "
                  f"балл кандидатов {candidate.get('балл кандидатов')} "
                  f"против {current.get('балл кандидатов')}"]
        for reason in verdict.get("причины", []):
            lines.append(f"  {GREY}— {reason}{RESET}")
    agreement = read_json(night / "submission_agreement.json")
    if agreement:
        lines += ["", f"{BOLD}Сверка с нынешней сдачей:{RESET} первый кандидат совпал в "
                      f"{agreement['top1_agreement'] * 100:.1f}% запросов, десятки — на "
                      f"{agreement['top10_overlap'] * 100:.1f}%; отказов "
                      f"{agreement['candidate']['refusal_rate'] * 100:.1f}% против "
                      f"{agreement['current']['refusal_rate'] * 100:.1f}%"]
    return "\n".join(lines)


def render(run: Path, width: int) -> str:
    history = read_json(run / "history.json") or []
    heartbeat = read_json(run / "heartbeat.json")
    config = read_json(run / "config.json") or {}
    summary = read_json(run.parent / "summary.json") if run.name == "model" else None

    lines: list[str] = []
    lines.append(f"{BOLD}ФАЛЬКОН — обучение{RESET}   {GREY}{time.strftime('%H:%M:%S')}{RESET}")
    lines.append(f"{GREY}прогон: {run}{RESET}")
    lines.append("")

    total_epochs = config.get("epochs") or (heartbeat or {}).get("epochs_total") or 0
    done = len(history)

    # Прогресс внутри текущей эпохи: из пульса, иначе оценка по времени.
    fresh = heartbeat and (time.time() - heartbeat.get("updated", 0) < 180)
    if fresh and heartbeat["epoch"] > done:
        inner = heartbeat["step"] / max(1, heartbeat["steps_total"])
        inner_text = f"шаг {heartbeat['step']}/{heartbeat['steps_total']}"
    elif history:
        typical = sorted(r["seconds"] for r in history)[len(history) // 2]
        elapsed = time.time() - (run / "history.json").stat().st_mtime
        inner = min(0.99, elapsed / max(1.0, typical))
        inner_text = f"~{human_time(elapsed)} из ~{human_time(typical)}"
    else:
        inner, inner_text = 0.0, "первая эпоха"

    overall = (done + inner) / total_epochs if total_epochs else 0.0
    lines.append(f"{BOLD}Всего{RESET}          {bar(overall, width - 30)} "
                 f"{overall * 100:5.1f}%   эпоха {done + 1}/{total_epochs}")
    lines.append(f"{BOLD}Текущая эпоха{RESET}  {bar(inner, width - 30, YELLOW)} "
                 f"{inner * 100:5.1f}%   {inner_text}")

    if history and total_epochs:
        typical = sorted(r["seconds"] for r in history)[len(history) // 2]
        remaining = (total_epochs - done - inner) * typical
        lines.append(f"{GREY}осталось примерно {human_time(remaining)}, "
                     f"эпоха идёт ~{human_time(typical)}{RESET}")
    lines.append("")

    # Качество
    scored = [(r["epoch"], r["validation"]["mAP@10"]) for r in history if r.get("validation")]
    if scored:
        best_epoch, best = max(scored, key=lambda x: x[1])
        trend = sparkline([v for _, v in scored])
        lines.append(f"{BOLD}Качество (mAP@10){RESET}")
        lines.append(f"  сейчас {GREEN}{scored[-1][1]:.4f}{RESET}   "
                     f"лучшее {GREEN}{best:.4f}{RESET} на эпохе {best_epoch}")
        lines.append(f"  динамика {CYAN}{trend}{RESET}  "
                     f"{GREY}({scored[0][1]:.3f} → {scored[-1][1]:.3f}){RESET}")
        lines.append(f"  {GREY}это {best * 45:.1f} балла из 45 за точность{RESET}")
    elif history:
        lines.append(f"{GREY}Первая проверка качества ещё не проводилась{RESET}")
    lines.append("")

    # Потери
    if history:
        last = history[-1]
        current_loss = (heartbeat or {}).get("loss", last.get("total"))
        current_acc = (heartbeat or {}).get("accuracy", last.get("accuracy"))
        lines.append(f"{BOLD}Обучение{RESET}   ошибка {current_loss:.3f}   "
                     f"узнаёт машины на {current_acc * 100:.0f}%   "
                     f"батч {config.get('identities_per_batch', 0) * config.get('shots_per_identity', 0)}")
    lines.append("")

    # Железо
    gpu = gpu_state()
    if gpu:
        share = gpu["used"] / max(1, gpu["total"])
        colour = RED if share > 0.93 else GREEN
        lines.append(f"{BOLD}Видеокарта{RESET} загрузка {gpu['util']:3d}%   "
                     f"память {colour}{gpu['used']}/{gpu['total']} МБ{RESET}   "
                     f"{gpu['temp']}°C")
    lines.append("")

    # Здоровье
    if history:
        age = time.time() - (run / "history.json").stat().st_mtime
        typical = sorted(r["seconds"] for r in history)[len(history) // 2]
        if age > max(600, typical * 6):
            busy = python_cpu_busy()
            if not busy:
                lines.append(f"{RED}{BOLD}ЗАВИСЛО{RESET} {RED}процессы не считают уже "
                             f"{human_time(age)}. Скажи Клоду.{RESET}")
            else:
                lines.append(f"{YELLOW}Эпоха идёт дольше обычного ({human_time(age)}), "
                             f"но процессы работают{RESET}")
        else:
            lines.append(f"{GREEN}Идёт нормально{RESET}")

    if summary:
        lines.append("")
        lines.append(f"{GREEN}{BOLD}ПАЙПЛАЙН ЗАВЕРШЁН{RESET}  "
                     f"mAP@10 {summary['validation']['mAP@10']:.4f}, "
                     f"порог {summary['threshold']:.3f}")

    lines.append("")
    lines.append(f"{GREY}Ctrl+C — выход. Окно можно закрыть, обучение продолжится.{RESET}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Живой индикатор обучения ФАЛЬКОН")
    parser.add_argument("run", type=Path, nargs="?", help="Каталог прогона")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    try:
        while True:
            run = args.run or find_active_run(root)
            width = min(100, shutil.get_terminal_size((90, 30)).columns)
            if run is None:
                print(CLEAR + f"{YELLOW}Активных прогонов не найдено.{RESET}\n"
                      f"{GREY}Жду появления runs/…/history.json{RESET}", flush=True)
            else:
                night = night_root(run)
                screen = render(run, width)
                if night is not None:
                    # Сверху — вся ночь, ниже — подробности текущей модели.
                    divider = "─" * min(width, 60)
                    screen = f"{render_night(night, width)}\n\n{divider}\n{screen}"
                print(CLEAR + screen, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nвыход")


if __name__ == "__main__":
    main()
