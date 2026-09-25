"""Двойники: когда над порогом оказываются две разные машины.

    python scripts/twin_check.py <каталог данных> --extra-train-share 0.75

Самая неприятная ошибка визуального поиска — двойник: другая машина той же
модели и окраски (каршеринг с одинаковой оклейкой, фургоны одной фирмы).
Номер их различил бы, но он размыт. Сервис не может знать наверняка, какая из
двух почти одинаковых машин настоящая, зато может заметить саму ситуацию:
над порогом две разные машины, и сходство у них почти равное. Тогда честнее
сказать «нужна проверка», чем уверенно назвать первую.

Скрипт подбирает зазор m для этого правила на машинах, которых модели не
видели: сколько ложных совпадений правило ловит и сколько верных ответов
зря отправляет на проверку. Отчёт — docs/twin_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.calibrate import find_twin  # noqa: E402
from falcon.data import build_local_split, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import l2_normalize  # noqa: E402
from falcon.submit import (  # noqa: E402
    CALIBRATED_THRESHOLD,
    SUBMISSION_CHECKPOINTS,
    SUBMISSION_FLIP_TTA,
    SUBMISSION_PROJECTION,
)


def best_by_vehicle(scores: np.ndarray, vehicles: list[str]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for score, vehicle in zip(scores, vehicles):
        if score > best.get(vehicle, -2.0):
            best[vehicle] = float(score)
    return sorted(best.items(), key=lambda item: -item[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Правило двойников: подбор зазора")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--extra-train-share", type=float, default=0.75)
    parser.add_argument("--threshold", type=float, default=CALIBRATED_THRESHOLD)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "twin_check.json")
    args = parser.parse_args()

    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42, extra_train_share=args.extra_train_share)
    extractor = build_extractor([ROOT / p for p in SUBMISSION_CHECKPOINTS],
                                ExtractorConfig(num_workers=4, threads=True, flip_tta=SUBMISSION_FLIP_TTA),
                                projection=ROOT / SUBMISSION_PROJECTION)
    query = l2_normalize(extractor.extract(split.query, progress=False))
    gallery = l2_normalize(extractor.extract(split.gallery, progress=False))
    scores = query @ gallery.T

    records = []
    for i, row in enumerate(split.query):
        # Протокол организаторов: снимки той же машины с той же камеры не считаются.
        keep = [j for j, g in enumerate(split.gallery)
                if not (g.vehicle_id == row.vehicle_id and g.camera_id == row.camera_id)]
        ranked = best_by_vehicle(scores[i, keep], [split.gallery[j].vehicle_id for j in keep])
        records.append({"ranked": ranked, "s1": ranked[0][1],
                        "correct": ranked[0][0] == row.vehicle_id})

    t = args.threshold
    accepted = [r for r in records if r["s1"] >= t]
    wrong = [r for r in accepted if not r["correct"]]
    right = [r for r in accepted if r["correct"]]
    report = {
        "что проверялось": "правило «над порогом две разные машины с почти равным "
                           "сходством → нужна проверка»",
        "выборка": f"{len(split.query)} запросов на машинах вне обучения "
                   f"(extra_train_share={args.extra_train_share})",
        "порог": t,
        "принято сервисом": len(accepted),
        "из них ошибочных совпадений": len(wrong),
        "варианты зазора": [],
    }
    for margin in (0.02, 0.03, 0.05, 0.08, 0.10):
        caught = sum(find_twin(r["ranked"], t, margin) is not None for r in wrong)
        false_alarms = sum(find_twin(r["ranked"], t, margin) is not None for r in right)
        report["варианты зазора"].append({
            "зазор": margin,
            "ошибок поймано": f"{caught} из {len(wrong)}",
            "верных ответов отправлено на проверку": f"{false_alarms} из {len(right)}",
            "доля проверок среди принятых": round((caught + false_alarms) / max(1, len(accepted)), 3),
        })
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
