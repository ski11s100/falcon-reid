"""Проверка на опору на номер: mAP@10 при закрашенной зоне номера.

    python scripts/plate_masking_check.py D:/falcon-data

Ответ 48 организаторов: финалистов прогоняют на контрольной версии теста, где
зона номера дополнительно закрашена сплошной заливкой, и заметное падение
метрики — основание для дисквалификации (раздел 9 ТЗ). Здесь это повторяется на
локальной отложенной выборке для ансамбля из сдачи: на ВСЕХ снимках запросов и
галереи закрашивается зона номера, и mAP@10 сравнивается с исходным и с
закраской контрольных зон той же площади (верх, левый и правый бок). Если зона
номера роняет метрику не сильнее контрольных зон, модель на номер не опирается.

Два набора зон одинаковой площади внутри набора, заливка чёрная:
  широкие — как в проверке одной пары в интерфейсе (plate_masking_boxes):
            центр низа кропа против крыши и боков. Центр низа занимает не только
            номер, но и решётку радиатора, эмблему и бампер — самое характерное
            в облике машины, так что сравнение с крышей и боками строгое;
  узкие   — номер отдельно от решётки: рамка номера против решётки над ним и
            низа слева и справа на той же высоте.
Итог — docs/plate_masking_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.data import build_local_split, read_manifest  # noqa: E402
from falcon.explain import plate_masking_boxes  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import Identity, evaluate_ranking, l2_normalize, rank_from_embeddings  # noqa: E402
from falcon.submit import SUBMISSION_CHECKPOINTS, SUBMISSION_FLIP_TTA  # noqa: E402


def masked(transform, box):
    """Та же предобработка, но сначала зона кропа закрашивается сплошным цветом."""
    def apply(image):
        if box is not None:
            image = image.copy()
            ImageDraw.Draw(image).rectangle(box, fill=(0, 0, 0))
        return transform(image)
    return apply


def main() -> None:
    parser = argparse.ArgumentParser(description="mAP@10 при закрашенной зоне номера")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        default=[ROOT / p for p in SUBMISSION_CHECKPOINTS])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "plate_masking_check.json")
    parser.add_argument("--flip-tta", action="store_true", default=SUBMISSION_FLIP_TTA)
    args = parser.parse_args()

    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42)
    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    q_labels = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.query}
    g_labels = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.gallery}

    extractor = build_extractor(args.checkpoints, ExtractorConfig(num_workers=8, threads=True,
                                                                  flip_tta=args.flip_tta))
    original = extractor.transform
    width, height = extractor.config.size[1], extractor.config.size[0]
    def box(x0, y0, x1, y1):
        return (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))

    zones = {"без закраски": None, **plate_masking_boxes(width, height)}
    tight = {
        "номер (узко)": box(0.38, 0.68, 0.62, 0.88),
        "решётка над номером": box(0.38, 0.48, 0.62, 0.68),
        "низ слева": box(0.08, 0.68, 0.32, 0.88),
        "низ справа": box(0.68, 0.68, 0.92, 0.88),
    }
    zones.update(tight)

    results = {}
    for name, box in zones.items():
        extractor.transform = masked(original, box)
        query = l2_normalize(extractor.extract(split.query, progress=False))
        gallery = l2_normalize(extractor.extract(split.gallery, progress=False))
        report = evaluate_ranking(rank_from_embeddings(query, gallery, qids, gids), q_labels, g_labels)
        results[name] = {"mAP@10": round(report.mAP, 4), "Rank-1": round(report.rank_1, 4)}
        print(name, results[name], flush=True)
    extractor.transform = original

    base = results["без закраски"]["mAP@10"]
    drops = {name: round(base - r["mAP@10"], 4) for name, r in results.items() if name != "без закраски"}
    tight_controls = [drops[name] for name in tight if name != "номер (узко)"]
    verdict = {
        "ансамбль": [p.name for p in args.checkpoints],
        "отражение кадра": args.flip_tta,
        "метрики": results,
        "падение mAP@10": drops,
        "узкая рамка номера против худшей узкой контрольной зоны": {
            "номер": drops["номер (узко)"], "худшая контрольная": max(tight_controls)},
        "вывод": ("узкая закраска номера роняет mAP@10 не сильнее контрольных зон той же площади"
                  if drops["номер (узко)"] <= max(tight_controls) else
                  "узкая закраска номера роняет mAP@10 сильнее контрольных зон той же площади"),
        "метод": ("зона закрашивается чёрным на всех снимках запросов и галереи локальной "
                  "отложенной выборки; контрольные зоны той же площади"),
    }
    args.output.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
