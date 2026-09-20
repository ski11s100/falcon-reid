"""Проверка готовности решения к сдаче.

Одна команда проверяет всё, что можно проверить автоматически, и печатает
понятный отчёт с оценкой в баллах конкурса:

    python scripts/check_ready.py

Проверяются семь групп: код и тесты, веса модели, измеренные метрики, файлы
сдачи, контейнеризация, работающий сервис, требования ТЗ к документации.

Скрипт ничего не меняет и не запускает обучение. Его задача — поймать
расхождение между тем, что мы думаем о решении, и тем, что есть на диске.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "[ OK ]", "[ НЕТ ]", "[  ! ]"


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.warnings: list[str] = []
        self.points: dict[str, float] = {}

    def check(self, condition: bool, label: str, detail: str = "", fatal: bool = True) -> bool:
        print(f"{OK if condition else (BAD if fatal else WARN)} {label}"
              + (f"  {detail}" if detail else ""))
        if not condition:
            (self.problems if fatal else self.warnings).append(label)
        return condition

    def note(self, label: str, detail: str = "") -> None:
        print(f"       {label}" + (f"  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * 62}")


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def check_code(report: Report, run_tests: bool) -> None:
    section("1. КОД И ТЕСТЫ")
    for name in ("falcon", "service", "scripts", "tests"):
        report.check((ROOT / name).is_dir(), f"каталог {name}/")

    modules = sorted(p.name for p in (ROOT / "falcon").glob("*.py"))
    report.check(len(modules) >= 10, "модули конкурсного ядра", f"{len(modules)} файлов")

    if not run_tests:
        report.note("тесты пропущены (--skip-tests)")
        return

    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    tail = (result.stderr or "").strip().splitlines()
    summary = next((line for line in reversed(tail) if line.startswith("Ran ")), "не определено")
    report.check(result.returncode == 0, "автоматические тесты", summary)


def check_weights(report: Report) -> None:
    section("2. ВЕСА МОДЕЛИ")
    models = sorted((ROOT / "models").glob("*.pt"))
    if not report.check(bool(models), "файлы весов в models/",
                        f"{len(models)} шт." if models else "не найдены"):
        return

    total_mb = sum(p.stat().st_size for p in models) / 1024 / 1024
    for path in models:
        report.note(path.name, f"{path.stat().st_size / 1024 / 1024:.0f} МБ")
    # Раздел 7 ТЗ: суммарный размер всех весов инференса не более 2 ГБ.
    report.check(total_mb <= 2048, "лимит 2 ГБ на веса", f"{total_mb:.0f} МБ")

    try:
        from falcon.extract import load_checkpoint
        model, meta = load_checkpoint(models[0])
        report.check(True, "чекпоинт читается", f"{meta.get('preprocessing', {})}")
        del model
    except Exception as exc:
        report.check(False, "чекпоинт читается", str(exc)[:80])


def check_metrics(report: Report) -> None:
    """Метрики берутся из отчётов, которые лежат в репозитории и идут в сдачу:
    docs/official_check.json (эталонный скрипт организаторов),
    docs/benchmark_container.json (замер в собранном образе),
    docs/plate_masking_check.json (проверка на опору на номер)."""
    section("3. ИЗМЕРЕННЫЕ МЕТРИКИ")
    official = load(ROOT / "docs" / "official_check.json")
    if report.check(official is not None, "отчёт docs/official_check.json"):
        ranking = official["ranking"]["эталонный скрипт"]
        accuracy = 45.0 * ranking["mAP@10"]
        report.points["точность"] = accuracy
        report.note("mAP@10", f"{ranking['mAP@10']:.4f}  ->  {accuracy:.1f} балла из 45")
        report.note("Rank-1 / Rank-5", f"{ranking['Rank-1']:.4f} / {ranking['Rank-5']:.4f}")
        report.check(official["ranking"]["совпадает"], "наш код совпадает с эталонным скриптом")

        mode = official["режим кандидатов"]
        candidate = 10.0 * mode["балл 0.7·F1+0.3·TNR"]
        report.points["кандидаты"] = candidate
        metrics = mode["эталонный скрипт"]
        report.note("порог отказа", f"{mode['порог']}")
        report.note("F1 / TNR", f"{metrics['F1']:.4f} / {metrics['TNR']:.4f}"
                                f"  ->  {candidate:.1f} балла из 10")
        report.check(mode["наш код совпадает"], "режим кандидатов совпадает с эталонным")
        naive = mode["всегда отвечать"]["балл"]
        report.check(mode["балл 0.7·F1+0.3·TNR"] > naive, "порог выигрывает у наивной стратегии",
                     f"{mode['балл 0.7·F1+0.3·TNR']:.4f} против {naive:.4f}")

    benchmark = load(ROOT / "docs" / "benchmark_container.json")
    if report.check(benchmark is not None, "замер в образе docs/benchmark_container.json"):
        scoring = benchmark["scoring"]
        report.points["скорость"] = scoring["performance_points_of_20"]
        latency = benchmark["latency_ms_b1_median"]
        fps = benchmark["best_throughput_fps"]
        report.check(latency <= 40, "задержка в пределах порога", f"{latency:.1f} мс при 40")
        report.check(fps >= 100, "пропускная способность", f"{fps:.0f} FPS при 100")
        if latency > 34:
            report.warnings.append(f"запас по задержке мал: {40 - latency:.1f} мс")
            print(f"{WARN} запас по задержке всего {40 - latency:.1f} мс")

    plate = load(ROOT / "docs" / "plate_masking_check.json")
    if plate:
        drops = plate["падение mAP@10"]
        narrow = plate["узкая рамка номера против худшей узкой контрольной зоны"]
        # Закраска зоны номера не должна ронять метрику сильнее, чем закраска
        # соседнего участка кузова той же площади (ответ 48).
        report.check(narrow["номер"] <= narrow["худшая контрольная"] + 0.002,
                     "модель не опирается на зону номера",
                     f"номер {narrow['номер']:.4f} против контрольной {narrow['худшая контрольная']:.4f}")
        report.note("падение при закраске", ", ".join(f"{k} {v:.3f}" for k, v in drops.items()))


def check_submission(report: Report) -> None:
    """Файлы сдачи в репозитории: формат, размерность, доля отказов, порог."""
    section("4. ФАЙЛЫ СДАЧИ")
    folder = ROOT / "submission"
    if not report.check(folder.is_dir(), "каталог submission/"):
        return

    from falcon.submit import CALIBRATED_THRESHOLD, validate_submission

    manifest = load(folder / "manifest.json")
    if not report.check(manifest is not None, "manifest.json"):
        return

    result = validate_submission(folder, manifest["queries"], manifest["gallery"])
    report.check(result["valid"], "формат прошёл самопроверку",
                 "; ".join(result["problems"])[:80] if result["problems"] else "")
    report.note("запросов / галерея", f"{manifest['queries']} / {manifest['gallery']}")
    report.note("размерность вектора", str(manifest["embedding_dim"]))
    report.note("отказов", f"{manifest['refused_queries']} "
                           f"({manifest['refusal_rate'] * 100:.1f}%)")
    report.check(manifest["threshold"] is not None, "порог задан",
                 f"{manifest['threshold']}" if manifest["threshold"] else "НЕТ: откажет во всём")
    report.check(manifest["threshold"] == CALIBRATED_THRESHOLD,
                 "порог в файлах сдачи совпадает с откалиброванным",
                 f"{manifest['threshold']} против {CALIBRATED_THRESHOLD}")


def check_docker(report: Report) -> None:
    section("5. КОНТЕЙНЕРИЗАЦИЯ")
    for name in ("Dockerfile", "docker-compose.yml", ".dockerignore", "requirements.txt"):
        report.check((ROOT / name).is_file(), name)

    # Раздел 7 ТЗ и ответ 39: версии зависимостей фиксируются точно.
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    loose = [line.strip() for line in requirements.splitlines()
             if line.strip() and not line.startswith("#") and "==" not in line]
    report.check(not loose, "версии зависимостей зафиксированы точно",
                 f"без ==: {loose}" if loose else "")

    try:
        images = subprocess.run(["docker", "images", "falcon-api", "--format", "{{.Size}}"],
                                capture_output=True, text=True, timeout=60).stdout.strip()
        report.check(bool(images), "образ falcon-api собран", images or "не найден", fatal=False)
    except Exception:
        report.check(False, "docker доступен", "не установлен или не запущен", fatal=False)


def check_service(report: Report, url: str) -> None:
    section("6. РАБОТАЮЩИЙ СЕРВИС")
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=10) as response:
            health = json.loads(response.read())
    except (urllib.error.URLError, OSError):
        report.check(False, "сервис отвечает", f"{url} недоступен — запустите docker compose up",
                     fatal=False)
        return

    report.check(health["status"] == "ok", "состояние", health["status"])
    report.note("модель", health["model"])
    report.note("устройство", health["device"])
    report.note("хранилище", f"{health['storage']}, поиск {health['search']}")
    report.check(health["threshold_calibrated"], "порог отказа задан в сервисе",
                 fatal=False)

    try:
        with urllib.request.urlopen(f"{url}/openapi.json", timeout=10) as response:
            spec = json.loads(response.read())
        report.check(len(spec["paths"]) >= 5, "спецификация OpenAPI",
                     f"{len(spec['paths'])} маршрутов")
    except Exception:
        report.check(False, "спецификация OpenAPI", fatal=False)


def check_documentation(report: Report) -> None:
    section("7. ДОКУМЕНТАЦИЯ (раздел 12 ТЗ)")
    readme = ROOT / "README.md"
    if not report.check(readme.is_file(), "README.md"):
        return

    text = readme.read_text(encoding="utf-8")
    required = {
        "описание архитектуры": "архитектур",
        "методы и алгоритмы": "алгоритм",
        "инструкция по запуску": "docker compose",
        "метрики на валидации": "mAP@10",
        "обоснование порога": "порог",
        "внешние источники": "внешн",
        "список библиотек с версиями": "requirements.txt",
    }
    for label, marker in required.items():
        report.check(marker.lower() in text.lower(), label)

    report.check((ROOT / ".git").is_dir(), "git-репозиторий (раздел 13 ТЗ)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка готовности решения к сдаче")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("ПРОВЕРКА ГОТОВНОСТИ РЕШЕНИЯ «ФАЛЬКОН»")
    print("=" * 62)

    report = Report()
    check_code(report, not args.skip_tests)
    check_weights(report)
    check_metrics(report)
    check_submission(report)
    check_docker(report)
    check_service(report, args.url)
    check_documentation(report)

    section("ИТОГ")
    if report.points:
        total = sum(report.points.values())
        for name, value in report.points.items():
            print(f"       {name:<12} {value:5.1f}")
        print(f"       {'—' * 18}")
        print(f"       {'измеримо':<12} {total:5.1f} из 75")
        print("       инженерное качество (15) и защита (10) оценивает жюри")

    print()
    if report.problems:
        print(f"{BAD} ПРОБЛЕМ: {len(report.problems)}")
        for index, problem in enumerate(report.problems, 1):
            print(f"  {index}. {problem}")
    else:
        print(f"{OK} КРИТИЧНЫХ ПРОБЛЕМ НЕТ")

    if report.warnings:
        print(f"\n{WARN} Замечания ({len(report.warnings)}):")
        for warning in report.warnings:
            print(f"  · {warning}")

    raise SystemExit(1 if report.problems else 0)


if __name__ == "__main__":
    main()
