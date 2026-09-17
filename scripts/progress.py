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
import sys
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
                print(CLEAR + render(run, width), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nвыход")


if __name__ == "__main__":
    main()
