"""Выбор состава моделей для сдачи: точность, значимость, порог и скорость.

    python scripts/compare_ensembles.py D:/falcon-data

Для каждого варианта ансамбля на общей отложенной выборке (split seed 42,
обучающие машины у всех моделей одни и те же):
  1. mAP@10 и Rank-1 эталонным скриптом организаторов;
  2. парный бутстреп разницы mAP@10 с текущей сдачей (5000 выборок запросов);
  3. порог отказа — максимум 0.7·F1 + 0.3·TNR после сглаживания окном ±0.01
     (как в verify_official.py, но на широкой сетке: у CLIP другое
     распределение сходства);
  4. задержка batch=1 и пропускная способность по методике организаторов
     (FeatureExtractor.benchmark) на свободной видеокарте.

Векторы каждой модели считаются один раз в fp16, с отражением кадра и без
него, и кешируются. Итог — docs/ensemble_choice.json и таблица в консоли.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from falcon.data import build_local_split, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import (  # noqa: E402
    Identity,
    evaluate_ranking,
    l2_normalize,
    performance_score,
    rank_from_embeddings,
)

MODELS = {
    "v1": ROOT / "runs" / "v1" / "model" / "best.pt",
    "v2-veri": ROOT / "runs" / "v2-veri" / "model" / "best.pt",
    "clip-ours": ROOT / "runs" / "clip" / "clip-ours" / "best.pt",
    "clip-veri-ours": ROOT / "runs" / "clip" / "clip-veri-ours" / "best.pt",
}
# Вариант: модели и отражение кадра (flip TTA). Отражение удваивает вычисления,
# поэтому тяжёлые ансамбли проверяются и без него.
VARIANTS = {
    "сдача сейчас: v1 + v2-veri": (["v1", "v2-veri"], True),
    "CLIP": (["clip-ours"], True),
    "CLIP + v2-veri": (["clip-ours", "v2-veri"], True),
    "2×CLIP": (["clip-ours", "clip-veri-ours"], True),
    "2×CLIP + v2-veri": (["clip-ours", "clip-veri-ours", "v2-veri"], True),
    "CLIP + v2-veri, без отражения": (["clip-ours", "v2-veri"], False),
    "2×CLIP, без отражения": (["clip-ours", "clip-veri-ours"], False),
    "2×CLIP + v2-veri, без отражения": (["clip-ours", "clip-veri-ours", "v2-veri"], False),
}
BASELINE = "сдача сейчас: v1 + v2-veri"


def load_official():
    spec = importlib.util.spec_from_file_location("official", ROOT / "organizers" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def combine(vectors: dict, names: list[str], flip: bool, part: int) -> np.ndarray:
    """Склейка нормированных векторов с равными весами — как EnsembleExtractor."""
    weight = 1.0 / len(names)
    return l2_normalize(np.hstack([vectors[(n, flip)][part] * weight for n in names]))


def calibrate(official, q_df, g_df, qids, gids, query, gallery) -> dict:
    scores = query @ gallery.T
    top = scores.argmax(1)
    best = scores[np.arange(len(qids)), top]

    def metrics(threshold: float) -> dict:
        chosen = {qids[i]: [(gids[top[i]], float(best[i]))]
                  for i in range(len(qids)) if best[i] >= threshold}
        with redirect_stdout(io.StringIO()):
            return official.candidate_metrics(q_df, g_df, chosen)

    grid = np.round(np.arange(0.20, 0.80, 0.0025), 4)
    curve = np.array([0.7 * m["F1"] + 0.3 * m["TNR"] for m in map(metrics, grid)])
    window = 9
    smooth = np.convolve(curve, np.ones(window) / window, mode="same")
    inner = slice(window // 2, len(grid) - window // 2)
    index = int(np.argmax(smooth[inner])) + window // 2
    threshold = float(grid[index])
    chosen = metrics(threshold)
    return {"порог": threshold, "F1": round(chosen["F1"], 4), "TNR": round(chosen["TNR"], 4),
            "балл 0.7·F1+0.3·TNR": round(0.7 * chosen["F1"] + 0.3 * chosen["TNR"], 4),
            "сглаженный балл": round(float(smooth[index]), 4)}


def bootstrap(per_query: dict, reference: dict, draws: int = 5000, seed: int = 0) -> dict:
    keys = sorted(reference)
    diff = np.array([per_query[k] - reference[k] for k in keys])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"разница mAP@10": round(float(diff.mean()), 4),
            "95% интервал": [round(float(low), 4), round(float(high), 4)],
            "доля выборок с разницей <= 0": round(float((means <= 0).mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Выбор состава моделей для сдачи")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("D:/falcon-cache/ensemble-vectors"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "ensemble_choice.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    official = load_official()
    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42)
    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    q_labels = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.query}
    g_labels = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.gallery}
    truth = pd.DataFrame([
        {"image_id": r.image_id, "vehicle_id": str(r.vehicle_id), "camera_id": str(r.camera_id), "split": part}
        for part, items in (("query", split.query), ("gallery", split.gallery)) for r in items])
    q_df = truth[truth.split == "query"].set_index("image_id")
    g_df = truth[truth.split == "gallery"].set_index("image_id")

    args.cache.mkdir(parents=True, exist_ok=True)
    vectors = {}
    needed = {(name, flip) for names, flip in VARIANTS.values() for name in names}
    for name, flip in sorted(needed):
        cached = args.cache / f"{name}{'' if flip else '-noflip'}.npz"
        if cached.is_file():
            data = np.load(cached)
            vectors[(name, flip)] = (data["query"], data["gallery"])
            continue
        config = ExtractorConfig(batch_size=32, num_workers=args.workers, threads=True, flip_tta=flip)
        extractor = build_extractor([MODELS[name]], config)
        vectors[(name, flip)] = (l2_normalize(extractor.extract(split.query, progress=False)),
                                 l2_normalize(extractor.extract(split.gallery, progress=False)))
        np.savez(cached, query=vectors[(name, flip)][0], gallery=vectors[(name, flip)][1])
        del extractor
        torch.cuda.empty_cache()
        print(f"векторы: {name}, отражение {flip}", flush=True)

    per_query, report = {}, {}
    for variant, (names, flip) in VARIANTS.items():
        query, gallery = combine(vectors, names, flip, 0), combine(vectors, names, flip, 1)
        ranking = rank_from_embeddings(query, gallery, qids, gids)
        with redirect_stdout(io.StringIO()):
            ranking_official = official.ranking_metrics(q_df, g_df, ranking)
        ours = evaluate_ranking(ranking, q_labels, g_labels)
        assert abs(ours.mAP - ranking_official["mAP@10"]) < 1e-9, variant
        per_query[variant] = ours.per_query
        report[variant] = {
            "модели": names,
            "отражение кадра": flip,
            "mAP@10": round(ranking_official["mAP@10"], 4),
            "Rank-1": round(ours.rank_1, 4),
            "кандидаты": calibrate(official, q_df, g_df, qids, gids, query, gallery),
            "веса, МБ": round(sum(MODELS[n].stat().st_size for n in names) / 1e6, 1),
        }
        print(variant, json.dumps(report[variant], ensure_ascii=False), flush=True)

    for variant in VARIANTS:
        if variant != BASELINE:
            report[variant]["против сдачи"] = bootstrap(per_query[variant], per_query[BASELINE])
    report["CLIP + v2-veri"]["против CLIP"] = bootstrap(per_query["CLIP + v2-veri"], per_query["CLIP"])
    report["2×CLIP + v2-veri"]["против CLIP + v2-veri"] = bootstrap(
        per_query["2×CLIP + v2-veri"], per_query["CLIP + v2-veri"])
    report["2×CLIP + v2-veri, без отражения"]["против 2×CLIP без отражения"] = bootstrap(
        per_query["2×CLIP + v2-veri, без отражения"], per_query["2×CLIP, без отражения"])
    for variant in VARIANTS:
        if variant.endswith(", без отражения"):
            full = variant.removesuffix(", без отражения")
            report[variant]["против варианта с отражением"] = bootstrap(per_query[variant], per_query[full])

    if not args.skip_benchmark:
        test_queries = read_manifest(args.dataset / "test_query.csv", args.dataset / "images")[:256]
        for variant, (names, flip) in VARIANTS.items():
            extractor = build_extractor([MODELS[n] for n in names],
                                        ExtractorConfig(batch_size=32, num_workers=args.workers,
                                                        threads=True, flip_tta=flip))
            bench = extractor.benchmark(test_queries)
            report[variant]["скорость"] = {
                **performance_score(bench["latency_ms_b1_median"], bench["best_throughput_fps"]),
                "latency_p95": bench["latency_ms_b1_p95"], "peak_vram_mb": bench["peak_vram_mb"],
                "fps_по_батчам": bench["throughput_fps"]}
            print(variant, json.dumps(report[variant]["скорость"], ensure_ascii=False), flush=True)
            del extractor
            torch.cuda.empty_cache()

    for entry in report.values():
        points = 45 * entry["mAP@10"] + 10 * entry["кандидаты"]["балл 0.7·F1+0.3·TNR"]
        if "скорость" in entry:
            points += entry["скорость"]["performance_points_of_20"]
        entry["оценка баллов из 75 (точность+кандидаты+скорость)"] = round(points, 2)

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nвариант | mAP@10 | порог | балл кандидатов | задержка, мс | FPS | баллы из 75")
    for variant, entry in report.items():
        speed = entry.get("скорость", {})
        print(f"{variant} | {entry['mAP@10']} | {entry['кандидаты']['порог']} | "
              f"{entry['кандидаты']['балл 0.7·F1+0.3·TNR']} | {speed.get('latency_ms_b1', '-')} | "
              f"{speed.get('throughput_fps', '-')} | {entry['оценка баллов из 75 (точность+кандидаты+скорость)']}")


if __name__ == "__main__":
    main()
