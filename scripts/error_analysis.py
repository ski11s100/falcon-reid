"""Разбор ошибок модели на отложенной выборке (разделы 10 и 11 ТЗ).

    python scripts/error_analysis.py D:/falcon-data --output docs/error_analysis

Считает, как точность зависит от условий съёмки, и собирает листы примеров:
  * качество по размеру рамки, яркости, резкости и смене ракурса;
  * «двойники» — первой оказалась другая машина, похожая на искомую;
  * провалы при смене ракурса — перед/зад против бока;
  * ложные совпадения — у машины нет пары в галерее, а сходство выше порога.

Ракурс оценивается по пропорциям рамки: бок машины вытянут (ширина/высота
около 2), перед и зад почти квадратные. Разметки ракурса в данных нет, поэтому
это приближение, и в отчёте оно так и названо.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.data import build_local_split, load_crop, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import l2_normalize  # noqa: E402
from falcon.submit import CALIBRATED_THRESHOLD, SUBMISSION_CHECKPOINTS, SUBMISSION_FLIP_TTA, SUBMISSION_PROJECTION  # noqa: E402

THRESHOLD = CALIBRATED_THRESHOLD
TILE = (220, 150)


def font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tile(row, label: str, colour: str) -> Image.Image:
    crop = load_crop(row, target=None)
    crop.thumbnail(TILE, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TILE[0], TILE[1] + 26), (14, 18, 27))
    canvas.paste(crop, ((TILE[0] - crop.width) // 2, (TILE[1] - crop.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, TILE[0] - 1, TILE[1] + 25], outline=colour, width=2)
    draw.text((8, TILE[1] + 4), label, fill=colour, font=font(14))
    return canvas


def sheet(rows_of_tiles: list[list[Image.Image]], title: str, path: Path) -> None:
    columns = max(len(r) for r in rows_of_tiles)
    title_width = int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(title, font=font(18)))
    width = max(columns * (TILE[0] + 10) + 10, title_width + 20)
    height = 44 + len(rows_of_tiles) * (TILE[1] + 36)
    canvas = Image.new("RGB", (width, height), (9, 12, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill=(238, 241, 246), font=font(18))
    for r, tiles in enumerate(rows_of_tiles):
        for c, t in enumerate(tiles):
            canvas.paste(t, (10 + c * (TILE[0] + 10), 44 + r * (TILE[1] + 36)))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=88)


def one_per_vehicle(records: list[dict], rows, limit: int) -> list[dict]:
    """По одному примеру на машину: несколько снимков одной машины подряд
    ничего не добавляют к разбору."""
    seen, out = set(), []
    for record in records:
        vehicle = rows[record["qi"]].vehicle_id
        if vehicle in seen:
            continue
        seen.add(vehicle)
        out.append(record)
        if len(out) == limit:
            break
    return out


def brightness_and_sharpness(row) -> tuple[float, float]:
    grey = np.asarray(load_crop(row, target=None).convert("L").resize((128, 96)), dtype=np.float32) / 255
    dy, dx = np.gradient(grey)
    return float(grey.mean()), float(np.var(np.hypot(dx, dy)))


def by_quartile(values: np.ndarray, hits: np.ndarray, labels: list[str]) -> list[dict]:
    edges = np.quantile(values, [0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for i in range(4):
        mask = (values >= edges[i]) & (values <= edges[i + 1] if i == 3 else values < edges[i + 1])
        out.append({"группа": labels[i], "запросов": int(mask.sum()),
                    "Rank-1": round(float(hits[mask].mean()), 3) if mask.any() else None})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Разбор ошибок ФАЛЬКОН")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "error_analysis")
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        default=[ROOT / p for p in SUBMISSION_CHECKPOINTS])
    parser.add_argument("--projection", type=Path, default=ROOT / SUBMISSION_PROJECTION,
                        help="PCA-проекция вектора ансамбля; --projection none — без неё")
    args = parser.parse_args()

    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42)
    extractor = build_extractor(args.checkpoints, ExtractorConfig(num_workers=8, threads=True,
                                                                  flip_tta=SUBMISSION_FLIP_TTA),
                                projection=None if str(args.projection).lower() == 'none' else args.projection)
    query = l2_normalize(extractor.extract(split.query, progress=False))
    gallery = l2_normalize(extractor.extract(split.gallery, progress=False))
    scores = query @ gallery.T

    g_vid = np.array([r.vehicle_id for r in split.gallery])
    g_cam = np.array([str(r.camera_id) for r in split.gallery])
    records = []
    for qi, q in enumerate(split.query):
        # Протокол организаторов: снимки той же машины с той же камеры — junk.
        junk = (g_vid == q.vehicle_id) & (g_cam == str(q.camera_id))
        order = [j for j in np.argsort(-scores[qi]) if not junk[j]]
        positives = [j for j in order if g_vid[j] == q.vehicle_id]
        top = order[0]
        records.append({
            "qi": qi, "top": top, "top_score": float(scores[qi, top]),
            "open_set": not positives,
            "correct": bool(positives) and top == positives[0],
            "first_pos": positives[0] if positives else None,
            "pos_rank": order.index(positives[0]) + 1 if positives else None,
        })

    scored = [r for r in records if not r["open_set"]]
    hits = np.array([r["correct"] for r in scored])
    q_rows = [split.query[r["qi"]] for r in scored]
    area = np.array([r.bbox[2] * r.bbox[3] for r in q_rows])
    light = np.array([brightness_and_sharpness(r) for r in q_rows])
    view_change = np.array([
        abs(math.log((q.bbox[2] / q.bbox[3]) /
                     (split.gallery[r["first_pos"]].bbox[2] / split.gallery[r["first_pos"]].bbox[3])))
        for q, r in zip(q_rows, scored)])

    stats = {
        "запросов с парой": len(scored),
        "Rank-1": round(float(hits.mean()), 4),
        "по размеру рамки": by_quartile(area, hits, ["самые мелкие", "мелкие", "крупные", "самые крупные"]),
        "по яркости": by_quartile(light[:, 0], hits, ["самые тёмные", "тёмные", "светлые", "самые светлые"]),
        "по резкости": by_quartile(light[:, 1], hits, ["самые смазанные", "смазанные", "резкие", "самые резкие"]),
        "по смене ракурса": by_quartile(view_change, hits,
                                        ["ракурс тот же", "почти тот же", "заметная смена", "перед/зад против бока"]),
    }

    # Двойники: неверный первый кандидат с высоким сходством.
    twins = one_per_vehicle(sorted([r for r in scored if not r["correct"]],
                                   key=lambda r: -r["top_score"]), split.query, 6)
    rows_twins = [[
        tile(split.query[r["qi"]], "запрос", "#7cc4ff"),
        tile(split.gallery[r["top"]], f"1-й: другая, {r['top_score']:.2f}", "#ff6b5b"),
        tile(split.gallery[r["first_pos"]], f"верная: место {r['pos_rank']}, "
             f"{scores[r['qi'], r['first_pos']]:.2f}", "#34c98b"),
    ] for r in twins]
    sheet(rows_twins, "Двойники: первой оказалась другая машина, похожая на искомую",
          args.output / "twins.jpg")

    # Смена ракурса: провалы с наибольшей разницей пропорций рамки.
    order_view = np.argsort(-view_change)
    view_fail = one_per_vehicle([scored[i] for i in order_view if not scored[i]["correct"]],
                                split.query, 6)
    rows_view = [[
        tile(split.query[r["qi"]], "запрос", "#7cc4ff"),
        tile(split.gallery[r["first_pos"]], f"та же машина: место {r['pos_rank']}", "#f5b53d"),
        tile(split.gallery[r["top"]], f"1-й: {r['top_score']:.2f}", "#ff6b5b"),
    ] for r in view_fail]
    sheet(rows_view, "Смена ракурса: перед или зад против бока — самый трудный случай",
          args.output / "viewpoint.jpg")

    # Ложные совпадения: у машины нет пары, а сходство выше порога.
    false_accepts = one_per_vehicle(sorted([r for r in records
                                            if r["open_set"] and r["top_score"] >= THRESHOLD],
                                           key=lambda r: -r["top_score"]), split.query, 6)
    rows_fa = [[
        tile(split.query[r["qi"]], "запрос: пары нет", "#7cc4ff"),
        tile(split.gallery[r["top"]], f"ложное совпадение {r['top_score']:.2f}", "#ff6b5b"),
    ] for r in false_accepts]
    if rows_fa:
        sheet(rows_fa, "Ложные совпадения: машины нет в галерее, сходство выше порога",
              args.output / "false_accepts.jpg")

    open_set = [r for r in records if r["open_set"]]
    stats["open-set"] = {
        "запросов без пары": len(open_set),
        "ложных совпадений выше порога": len([r for r in open_set if r["top_score"] >= THRESHOLD]),
    }
    stats["двойники: сходство неверного первого"] = [round(r["top_score"], 3) for r in twins]

    # Публичный тест: разметки нет, но у каждого его запроса есть пара в
    # галерее (ответ 17 организаторов). Значит, каждый отказ там — наша ошибка,
    # ложный отказ, и её можно показать на настоящих тестовых кадрах.
    if (args.dataset / "test_query.csv").is_file():
        t_query = read_manifest(args.dataset / "test_query.csv", args.dataset / "images")
        t_gallery = read_manifest(args.dataset / "test_gallery.csv", args.dataset / "images")
        t_scores = (l2_normalize(extractor.extract(t_query, progress=False))
                    @ l2_normalize(extractor.extract(t_gallery, progress=False)).T)
        t_top = t_scores.argmax(1)
        t_best = t_scores[np.arange(len(t_query)), t_top]
        answered = t_best >= THRESHOLD
        t_light = np.array([brightness_and_sharpness(r) for r in t_query])
        t_area = np.array([r.bbox[2] * r.bbox[3] / 1000 for r in t_query])

        def medians(values):
            return [round(float(np.median(values[~answered])), 3), round(float(np.median(values[answered])), 3)]

        stats["публичный тест: ложные отказы"] = {
            "запросов": len(t_query),
            "отказов (все — ошибки: пара есть у каждого)": int((~answered).sum()),
            "доля": round(float((~answered).mean()), 4),
            "медиана яркости: отказ / ответ": medians(t_light[:, 0]),
            "медиана резкости: отказ / ответ": medians(t_light[:, 1]),
            "медиана площади рамки, тыс. пикс.: отказ / ответ": medians(t_area),
        }
        refused = np.where(~answered)[0]
        worst = refused[np.argsort(t_best[refused])][:6]
        rows_refused = [[
            tile(t_query[i], "запрос: пара есть", "#7cc4ff"),
            tile(t_gallery[t_top[i]], f"1-й: {t_best[i]:.2f} < {THRESHOLD:g}", "#f5b53d"),
        ] for i in worst]
        if rows_refused:
            sheet(rows_refused, "Публичный тест: отказ там, где пара есть (ложный отказ)",
                  args.output / "public_refusals.jpg")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
