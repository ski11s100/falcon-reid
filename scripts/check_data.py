"""Проверка конкурсного датасета: всё ли на месте и пригодно ли к работе.

Запускается сразу после распаковки архива. Ничего не меняет и не портит, только
читает и печатает понятный отчёт:

    python scripts/check_data.py D:/falcon-data

Проверяет структуру каталога, целостность CSV, наличие всех кадров, корректность
рамок BBox и пригодность данных для кросс-камерного протокола оценки.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK = "[ OK ]"
BAD = "[ НЕТ ]"
WARN = "[ ! ]"


def human(number: int) -> str:
    return f"{number:,}".replace(",", " ")


def check_structure(root: Path) -> tuple[bool, list[str]]:
    """Есть ли все файлы, которые ожидает конвейер."""
    print("\n1. СТРУКТУРА КАТАЛОГА")
    print("-" * 60)
    problems: list[str] = []

    if not root.is_dir():
        print(f"{BAD} Каталог {root} не существует")
        return False, [f"Каталог {root} не найден. Создайте его и распакуйте туда архив."]

    expected = ["train.csv", "test_query.csv", "test_gallery.csv"]
    for name in expected:
        path = root / name
        if path.is_file():
            print(f"{OK} {name}  ({path.stat().st_size // 1024} КБ)")
        else:
            print(f"{BAD} {name} — не найден")
            problems.append(f"Не хватает файла {name}")

    images = root / "images"
    if images.is_dir():
        count = sum(1 for _ in images.iterdir())
        print(f"{OK} images/  ({human(count)} файлов)")
    else:
        print(f"{BAD} images/ — каталог не найден")
        problems.append("Не найден каталог images/ с кадрами")
        # Частая причина: архив распакован с лишним уровнем вложенности.
        nested = [d for d in root.iterdir() if d.is_dir() and (d / "images").is_dir()]
        if nested:
            problems.append(
                f"Похоже, данные лежат на уровень глубже: в {nested[0].name}/. "
                f"Передайте этот путь: {nested[0]}")

    for extra, note in (("README.md", "описание датасета от организаторов"),
                        ("evaluate.py", "эталонный скрипт расчёта метрики")):
        if (root / extra).is_file():
            print(f"{OK} {extra}  ({note})")
        else:
            print(f"{WARN} {extra} — нет ({note}); не критично, но лучше забрать")

    return not problems, problems


def check_csv(root: Path) -> tuple[bool, list[str]]:
    """Колонки, количество строк, уникальность image_id."""
    print("\n2. ТАБЛИЦЫ РАЗМЕТКИ")
    print("-" * 60)
    problems: list[str] = []
    totals: dict[str, int] = {}

    for name, needs_labels in (("train.csv", True), ("test_query.csv", False),
                               ("test_gallery.csv", False)):
        path = root / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            rows = list(reader)

        totals[name] = len(rows)
        required = {"image_id", "x", "y"} | ({"w", "h"} if "w" in fields else {"width", "height"})
        if needs_labels:
            required.add("vehicle_id")
        missing = required - fields
        ids = [r["image_id"] for r in rows]
        duplicates = len(ids) - len(set(ids))

        status = OK if not missing and not duplicates else BAD
        print(f"{status} {name}: {human(len(rows))} строк, колонки: {', '.join(sorted(fields))}")
        if missing:
            problems.append(f"{name}: не хватает колонок {sorted(missing)}")
        if duplicates:
            problems.append(f"{name}: {duplicates} повторяющихся image_id")

        if name == "train.csv" and "camera_id" in fields:
            vehicles = len({r["vehicle_id"] for r in rows})
            cameras = len({r["camera_id"] for r in rows})
            print(f"       {human(vehicles)} автомобилей, {cameras} камер")
        elif name == "train.csv":
            print(f"{WARN}   нет колонки camera_id — кросс-камерную оценку не построить")
            problems.append("В train.csv нет camera_id: локальная метрика будет неверной")

    # Ожидания из ответов организаторов: 9556 / 1110 / 750.
    expected = {"train.csv": 9556, "test_query.csv": 1110, "test_gallery.csv": 750}
    for name, count in expected.items():
        actual = totals.get(name)
        if actual is not None and actual != count:
            print(f"{WARN} {name}: {human(actual)} строк, организаторы называли {human(count)}. "
                  f"Возможно, датасет обновили — это не ошибка, просто учтём.")

    return not problems, problems


def check_images(root: Path, sample_size: int) -> tuple[bool, list[str]]:
    """Все ли кадры на месте и корректны ли рамки BBox."""
    print("\n3. КАДРЫ И РАМКИ")
    print("-" * 60)
    problems: list[str] = []

    try:
        from falcon.data import load_crop, read_manifest
    except ImportError as exc:
        print(f"{BAD} Не загружаются модули проекта: {exc}")
        return False, ["Запускайте скрипт из корня проекта"]

    for name, needs_labels in (("train.csv", True), ("test_query.csv", False),
                               ("test_gallery.csv", False)):
        if not (root / name).is_file():
            continue
        try:
            rows = read_manifest(root / name, root / "images", require_labels=needs_labels)
        except FileNotFoundError as exc:
            print(f"{BAD} {name}: {exc}")
            problems.append(f"{name}: часть кадров отсутствует в images/. "
                            f"Скачайте архив заново — организаторы отвечали, что файлы на месте.")
            continue
        except ValueError as exc:
            print(f"{BAD} {name}: {exc}")
            problems.append(f"{name}: {exc}")
            continue

        # Кропы проверяются выборочно: читать 11 тысяч кадров целиком долго.
        step = max(1, len(rows) // sample_size)
        checked = broken = 0
        sizes: list[int] = []
        for row in rows[::step]:
            checked += 1
            try:
                crop = load_crop(row, target=(256, 256))
                sizes.append(min(crop.size))
            except Exception:
                broken += 1

        status = OK if not broken else BAD
        print(f"{status} {name}: проверено {checked} кадров из {human(len(rows))}, "
              f"повреждённых {broken}")
        if sizes:
            sizes.sort()
            tiny = sum(1 for s in sizes if s < 48)
            print(f"       меньшая сторона кропа: минимум {sizes[0]}, "
                  f"медиана {sizes[len(sizes) // 2]}, максимум {sizes[-1]} пикс.")
            if tiny:
                print(f"{WARN}   {tiny} из {len(sizes)} кропов мельче 48 пикселей — "
                      f"на них различающих деталей почти нет")
        if broken:
            problems.append(f"{name}: {broken} кадров не читаются")

    return not problems, problems


def check_protocol(root: Path) -> tuple[bool, list[str]]:
    """Хватит ли данных для честной локальной валидации."""
    print("\n4. ПРИГОДНОСТЬ ДЛЯ ОЦЕНКИ")
    print("-" * 60)
    problems: list[str] = []

    from falcon.data import audit, build_local_split, read_manifest

    rows = read_manifest(root / "train.csv", root / "images", require_labels=True)
    report = audit(rows)

    print(f"       наблюдений: {human(report['rows'])}")
    print(f"       автомобилей: {human(report['identities'])}")
    print(f"       камер: {report['cameras']}")
    print(f"       снимков на автомобиль: от {report['shots_per_identity_min']} "
          f"до {report['shots_per_identity_max']}, в среднем {report['shots_per_identity_mean']}")

    multi = report["identities_with_multiple_cameras"]
    share = multi / max(1, report["identities"])
    status = OK if share > 0.5 else WARN
    print(f"{status} снято двумя и более камерами: {human(multi)} "
          f"({share:.0%} от всех автомобилей)")
    if share < 0.5:
        problems.append("Меньше половины автомобилей сняты разными камерами — "
                        "кросс-камерная валидация будет на малой выборке")

    if report["singleton_identities"]:
        print(f"{WARN} автомобилей с единственным снимком: "
              f"{human(report['singleton_identities'])} — для обучения пар они бесполезны")

    try:
        split = build_local_split(rows, seed=42)
        summary = split.summary()
        print(f"{OK} локальный сплит строится:")
        print(f"       обучение — {human(summary['train_rows'])} снимков, "
              f"{human(summary['train_identities'])} автомобилей")
        print(f"       галерея — {human(summary['gallery_rows'])} снимков")
        print(f"       запросы — {human(summary['query_rows'])}, из них "
              f"{human(summary['open_set_queries'])} без пары "
              f"({summary['open_set_share']:.0%}, у организаторов около 20%)")
    except ValueError as exc:
        print(f"{BAD} локальный сплит построить нельзя: {exc}")
        problems.append(f"Локальный сплит: {exc}")

    return not problems, problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка конкурсного датасета ФАЛЬКОН")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("D:/falcon-data"))
    parser.add_argument("--sample", type=int, default=300,
                        help="Сколько кадров открыть для проверки целостности")
    args = parser.parse_args()

    print("=" * 60)
    print(f"ПРОВЕРКА ДАТАСЕТА: {args.dataset}")
    print("=" * 60)

    all_problems: list[str] = []
    ok, problems = check_structure(args.dataset)
    all_problems += problems

    if ok:
        for check in (lambda: check_csv(args.dataset),
                      lambda: check_images(args.dataset, args.sample),
                      lambda: check_protocol(args.dataset)):
            try:
                _, problems = check()
                all_problems += problems
            except Exception as exc:
                print(f"{BAD} проверка прервалась: {exc}")
                all_problems.append(str(exc))
                break

    print("\n" + "=" * 60)
    if all_problems:
        print("ИТОГ: есть проблемы, нужно поправить")
        print("=" * 60)
        for index, problem in enumerate(all_problems, start=1):
            print(f"  {index}. {problem}")
        raise SystemExit(1)

    print("ИТОГ: датасет в порядке, можно запускать обучение")
    print("=" * 60)
    print(f"\n  python scripts/run_pipeline.py {args.dataset} --output runs/v1\n")


if __name__ == "__main__":
    main()
