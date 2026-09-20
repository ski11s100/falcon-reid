"""Порог отказа, устойчивый и к жеребьёвке, и к плотности галереи.

    python scripts/calibrate_threshold.py D:/falcon-data

Балл режима кандидатов `0.7·F1 + 0.3·TNR` считается эталонным скриптом
организаторов на НЕСКОЛЬКИХ прогонах, и прогоны различаются двумя способами.

1. Жеребьёвка отложенной части. Обучающие машины везде одни и те же, меняется
   только деление остального: кто попал в галерею, кто в запросы, кто остался
   без пары (falcon/data.build_local_split, partition_seed). Оптимум одного
   разбиения оказывается подогнанным под него: порог 0.595, найденный на
   основном сплите, давал там 0.872, а на других — 0.827–0.861.

2. Плотность галереи. У нас на 890 запросов приходится 1541 снимок галереи, а
   в публичном тесте на 1110 запросов — всего 750. Чем реже галерея, тем ниже
   сходство с лучшим кандидатом и тем ниже оптимальный порог: на нашей галерее
   он около 0.67, на прореженной до 750 снимков — около 0.56. Поэтому считается
   и такой вариант: галерея прорежена до 750 снимков, а доля запросов без пары
   возвращена к 20% (иначе прореживание само плодит запросы без пары и завышает
   вес TNR).

Сдаётся максимум среднего по обеим плотностям: на закрытом тесте будет ровно
одна жеребьёвка и одна плотность, и какие — неизвестно.

Пишет docs/threshold_choice.json (кривые и выбор) и docs/threshold_curves.svg
(картинка для защиты).
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.data import build_local_split, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import l2_normalize  # noqa: E402
from falcon.submit import (  # noqa: E402
    CALIBRATED_THRESHOLD,
    SUBMISSION_CHECKPOINTS,
    SUBMISSION_FLIP_TTA,
    SUBMISSION_PROJECTION,
    SubmissionConfig,
    enrich_vectors,
)

COLOURS = ["#7cc4ff", "#34c98b", "#f5b53d", "#ff6b5b", "#b98cff"]


def load_official():
    spec = importlib.util.spec_from_file_location("official", ROOT / "organizers" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shrink(split, size: int, rng: np.random.Generator):
    """Галерея как в тесте по размеру, доля запросов без пары — прежние 20%."""
    keep = sorted(rng.choice(len(split.gallery), size=size, replace=False))
    gallery = [split.gallery[i] for i in keep]
    cameras: dict[str, set[str]] = {}
    for row in gallery:
        cameras.setdefault(str(row.vehicle_id), set()).add(str(row.camera_id))

    def paired(row) -> bool:
        seen = cameras.get(str(row.vehicle_id))
        return bool(seen) and any(camera != str(row.camera_id) for camera in seen)

    closed = [row for row in split.query if paired(row)]
    orphans = [row for row in split.query if not paired(row)]
    rng.shuffle(orphans)
    wanted = int(round(0.2 / 0.8 * len(closed)))
    return SimpleNamespace(query=closed + orphans[:wanted], gallery=gallery)


def curve(official, split, vectors: dict[str, np.ndarray], grid: np.ndarray) -> np.ndarray:
    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    query = np.stack([vectors[i] for i in qids])
    gallery = np.stack([vectors[i] for i in gids])
    plain = query @ gallery.T
    enriched_query, enriched_gallery = enrich_vectors(query, gallery, SubmissionConfig())
    top = (enriched_query @ enriched_gallery.T).argmax(1)
    confidence = plain[np.arange(len(qids)), top]

    truth = pd.DataFrame([{"image_id": r.image_id, "vehicle_id": str(r.vehicle_id),
                           "camera_id": str(r.camera_id), "split": part}
                          for part, items in (("query", split.query), ("gallery", split.gallery))
                          for r in items])
    q_df = truth[truth.split == "query"].set_index("image_id")
    g_df = truth[truth.split == "gallery"].set_index("image_id")

    values = []
    for threshold in grid:
        chosen = {qids[i]: [(gids[top[i]], float(confidence[i]))]
                  for i in range(len(qids)) if confidence[i] >= threshold}
        with redirect_stdout(io.StringIO()):
            metrics = official.candidate_metrics(q_df, g_df, chosen)
        values.append(0.7 * metrics["F1"] + 0.3 * metrics["TNR"])
    return np.array(values)


def draw(grid: np.ndarray, curves: dict[str, np.ndarray], mean: np.ndarray,
         chosen: float, path: Path) -> None:
    """График кривых балла: видно и разброс между разбиениями, и выбор порога."""
    width, height, pad = 900, 430, 60
    x = lambda t: pad + (t - grid[0]) / (grid[-1] - grid[0]) * (width - 2 * pad)
    lo, hi = 0.55, 0.92
    y = lambda v: height - pad - (v - lo) / (hi - lo) * (height - 2 * pad)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
             f'height="{height}" font-family="Manrope, Segoe UI, Arial, sans-serif">',
             f'<rect width="{width}" height="{height}" fill="#0b0e14"/>',
             '<text x="24" y="34" fill="#eef1f6" font-size="17" font-weight="700">'
             'Балл режима кандидатов 0.7·F1 + 0.3·TNR в зависимости от порога</text>',
             '<text x="24" y="54" fill="#6f7688" font-size="12">Каждая кривая — среднее по четырём '
             'жеребьёвкам; белая — среднее плотностей</text>']
    for value in (0.6, 0.7, 0.8, 0.9):
        parts.append(f'<line x1="{pad}" y1="{y(value):.1f}" x2="{width - pad}" y2="{y(value):.1f}" '
                     f'stroke="rgba(255,255,255,0.08)"/>')
        parts.append(f'<text x="{pad - 34}" y="{y(value) + 4:.1f}" fill="#6f7688" font-size="11">{value:.1f}</text>')
    for tick in np.arange(grid[0], grid[-1] + 0.001, 0.05):
        parts.append(f'<text x="{x(tick) - 12:.1f}" y="{height - pad + 20}" fill="#6f7688" '
                     f'font-size="11">{tick:.2f}</text>')
    for colour, (label, values) in zip(COLOURS, curves.items()):
        points = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(grid, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="1.6" opacity="0.75"/>')
    points = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(grid, mean))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#ffffff" stroke-width="3"/>')
    parts.append(f'<line x1="{x(chosen):.1f}" y1="{pad - 10}" x2="{x(chosen):.1f}" y2="{height - pad}" '
                 f'stroke="#ffffff" stroke-dasharray="5 5" opacity="0.8"/>')
    parts.append(f'<text x="{x(chosen) + 8:.1f}" y="{pad + 6}" fill="#ffffff" font-size="13" '
                 f'font-weight="700">выбран {chosen:.3f}</text>')
    legend = list(curves) + ["среднее"]
    for i, (label, colour) in enumerate(zip(legend, COLOURS[:len(curves)] + ["#ffffff"])):
        parts.append(f'<rect x="{width - pad - 150}" y="{pad + i * 20 - 8}" width="12" height="3" fill="{colour}"/>')
        parts.append(f'<text x="{width - pad - 132}" y="{pad + i * 20 - 2}" fill="#a3aab8" '
                     f'font-size="12">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Выбор порога отказа по нескольким разбиениям")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--partitions", type=int, nargs="+", default=[7, 13, 21],
                        help="Дополнительные жеребьёвки отложенной выборки")
    parser.add_argument("--test-gallery", type=int, default=750,
                        help="Размер прореженной галереи — как в публичном тесте")
    parser.add_argument("--draws", type=int, default=2,
                        help="Сколько раз прореживать галерею в каждой жеребьёвке")
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        default=[ROOT / p for p in SUBMISSION_CHECKPOINTS])
    parser.add_argument("--projection", type=Path, default=ROOT / SUBMISSION_PROJECTION)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "threshold_choice.json")
    args = parser.parse_args()

    official = load_official()
    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    base = build_local_split(rows, seed=42)

    # Векторы считаются один раз: разбиения тасуют одни и те же снимки.
    extractor = build_extractor(args.checkpoints, ExtractorConfig(num_workers=8, threads=True,
                                                                  flip_tta=SUBMISSION_FLIP_TTA),
                                projection=args.projection)
    held_out = base.query + base.gallery
    matrix = l2_normalize(extractor.extract(held_out, progress=False))
    vectors = {row.image_id: matrix[i] for i, row in enumerate(held_out)}

    grid = np.round(np.arange(0.45, 0.85, 0.005), 4)
    splits = [("основное", base)]
    for partition in args.partitions:
        splits.append((f"жеребьёвка {partition}",
                       build_local_split(rows, seed=42, partition_seed=partition)))

    dense_name = "наша галерея (1541 снимок)"
    sparse_name = f"как в тесте ({args.test_gallery} снимков)"
    per_run: dict[str, np.ndarray] = {}
    families: dict[str, list[np.ndarray]] = {dense_name: [], sparse_name: []}
    for label, split in splits:
        dense = curve(official, split, vectors, grid)
        per_run[label] = dense
        families[dense_name].append(dense)
        for attempt in range(args.draws):
            rng = np.random.default_rng(1000 + attempt)
            sparse = curve(official, shrink(split, args.test_gallery, rng), vectors, grid)
            per_run[f"{label}, галерея {args.test_gallery}"] = sparse
            families[sparse_name].append(sparse)

    curves = {name: np.vstack(values).mean(0) for name, values in families.items()}
    # Среднее по семействам, а не по прогонам: обе плотности весят одинаково.
    mean = np.vstack(list(curves.values())).mean(0)
    smooth = np.convolve(mean, np.ones(9) / 9, mode="same")
    best = int(np.argmax(smooth[4:-4])) + 4
    chosen = float(grid[best])

    def summary(values: np.ndarray) -> dict:
        return {"собственный оптимум": float(grid[int(np.argmax(values))]),
                "балл в нём": round(float(values.max()), 4),
                "балл при выбранном пороге": round(float(values[best]), 4)}

    report = {
        "выбранный порог": chosen,
        "порог в falcon/submit.py": CALIBRATED_THRESHOLD,
        "метод": ("максимум среднего по двум плотностям галереи (наша и как в тесте), "
                  "каждая усреднена по жеребьёвкам, после сглаживания окном ±0.01"),
        "по плотностям": {name: summary(values) for name, values in curves.items()},
        "по прогонам": {label: summary(values) for label, values in per_run.items()},
        "средний балл при выбранном пороге": round(float(mean[best]), 4),
        "кривые": {label: [round(float(v), 4) for v in values] for label, values in curves.items()},
        "сетка": grid.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    draw(grid, curves, mean, chosen, args.output.with_name("threshold_curves.svg"))
    print(json.dumps({k: v for k, v in report.items() if k not in ("кривые", "сетка")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
