"""Сравнение нынешней сдачи с новым кандидатом и готовый вердикт: менять или нет.

    python scripts/compare_release.py D:/falcon-data \\
        --candidate runs/res288/models/clip-ours.pt runs/res288/models/clip-veri-ours.pt \\
                    runs/res288/models/resnet-v2-veri.pt \\
        --candidate-projection runs/res288/projection.pt \\
        --candidate-threshold-report runs/res288/threshold_choice.json

Зачем отдельный скрипт. После ночного дообучения соблазн переключить сдачу по
одному числу — mAP@10 стал выше. Этого мало: у нового ансамбля своя шкала
сходства (значит, и свой порог), свои задержка и пропускная способность, и
прирост точности может оказаться в пределах шума. Скрипт считает всё сразу на
одной и той же отложенной выборке и печатает вердикт по правилам:

  * точность растёт, и 95% интервал парного бутстрэпа не задевает нуля;
  * балл режима кандидатов (каждый со СВОИМ порогом) не падает больше чем на
    0.005 — иначе выигрыш в mAP съедается отказами;
  * задержка batch=1 <= 40 мс и пропускная способность >= 100 кадров/с, иначе
    теряется полный балл за производительность (ответ 34).

Числа для ранжирования считает эталонный скрипт организаторов
(organizers/evaluate.py), а не наш код.
"""

from __future__ import annotations

import argparse
import gc
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
    TOP_K,
    Identity,
    evaluate_ranking,
    l2_normalize,
    performance_score,
)
from falcon.submit import (  # noqa: E402
    CALIBRATED_THRESHOLD,
    SUBMISSION_CHECKPOINTS,
    SUBMISSION_FLIP_TTA,
    SUBMISSION_PROJECTION,
    SubmissionConfig,
    enrich_vectors,
)

# Насколько балл кандидатов может просесть ради прироста точности.
CANDIDATE_TOLERANCE = 0.005
LATENCY_LIMIT_MS = 40.0
THROUGHPUT_LIMIT_FPS = 100.0


def load_official():
    spec = importlib.util.spec_from_file_location("official", ROOT / "organizers" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def truth_frames(split):
    rows = [{"image_id": r.image_id, "vehicle_id": str(r.vehicle_id),
             "camera_id": str(r.camera_id), "split": part}
            for part, items in (("query", split.query), ("gallery", split.gallery))
            for r in items]
    table = pd.DataFrame(rows)
    return (table[table.split == "query"].set_index("image_id"),
            table[table.split == "gallery"].set_index("image_id"))


def evaluate(official, split, query, gallery, threshold):
    """Ранжирование и режим кандидатов одного варианта на общей выборке."""
    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    config = SubmissionConfig()
    enriched_query, enriched_gallery = enrich_vectors(query, gallery, config)
    similarity = enriched_query @ enriched_gallery.T
    order = np.argsort(-similarity, axis=1)[:, :TOP_K]
    ranking = {qids[i]: [gids[j] for j in order[i]] for i in range(len(qids))}

    q_df, g_df = truth_frames(split)
    with redirect_stdout(io.StringIO()):
        ranking_metrics = official.ranking_metrics(q_df, g_df, ranking)

    top = similarity.argmax(1)
    confidence = (query @ gallery.T)[np.arange(len(qids)), top]
    chosen = {qids[i]: [(gids[top[i]], float(confidence[i]))]
              for i in range(len(qids)) if confidence[i] >= threshold}
    with redirect_stdout(io.StringIO()):
        candidate_metrics = official.candidate_metrics(q_df, g_df, chosen)

    ours = evaluate_ranking(
        ranking,
        {r.image_id: Identity(str(r.vehicle_id), str(r.camera_id)) for r in split.query},
        {r.image_id: Identity(str(r.vehicle_id), str(r.camera_id)) for r in split.gallery})
    return ranking_metrics, candidate_metrics, ours.per_query


def bootstrap(before: dict[str, float], after: dict[str, float], draws: int = 5000) -> dict:
    """Парный бутстрэп по запросам: сравниваем на одних и тех же запросах."""
    shared = sorted(set(before) & set(after))
    a = np.array([before[k] for k in shared])
    b = np.array([after[k] for k in shared])
    rng = np.random.default_rng(0)
    samples = rng.integers(0, len(shared), size=(draws, len(shared)))
    differences = np.array([(b[s] - a[s]).mean() for s in samples])
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"разница mAP@10": round(float(b.mean() - a.mean()), 4),
            "95% интервал": [round(float(low), 4), round(float(high), 4)],
            "запросов": len(shared)}


def verdict(report: dict, with_speed: bool = True) -> tuple[str, list[str]]:
    """Менять ли сдачу. Возвращает вердикт и список причин отказаться.

    Правила намеренно строгие: замена ансамбля стоит пересчёта порога, сдачи и
    всех отчётов, поэтому она оправдана только уверенным приростом точности,
    который ничего не ломает.
    """
    reasons = []
    if report["бутстрэп"]["95% интервал"][0] <= 0:
        reasons.append("прирост mAP@10 неотличим от нуля")
    drop = report["сдача"]["балл кандидатов"] - report["кандидат"]["балл кандидатов"]
    if drop > CANDIDATE_TOLERANCE:
        reasons.append(f"балл кандидатов падает на {drop:.4f}")
    if with_speed:
        if report["кандидат"]["задержка batch=1, мс"] > LATENCY_LIMIT_MS:
            reasons.append("задержка выше 40 мс")
        if report["кандидат"]["пропускная способность, кадр/с"] < THROUGHPUT_LIMIT_FPS:
            reasons.append("пропускная способность ниже 100 кадров/с")
    return ("менять сдачу" if not reasons else "оставить как есть"), reasons


def vectors_for(checkpoints, projection, split, workers):
    extractor = build_extractor(
        [Path(c) for c in checkpoints],
        ExtractorConfig(num_workers=workers, threads=True, half=True, flip_tta=SUBMISSION_FLIP_TTA),
        projection=None if projection is None else Path(projection))
    query = l2_normalize(extractor.extract(split.query, progress=False))
    gallery = l2_normalize(extractor.extract(split.gallery, progress=False))
    return extractor, query, gallery


def threshold_from(report: Path | None, default: float) -> float:
    if report is None:
        return default
    data = json.loads(Path(report).read_text(encoding="utf-8"))
    return float(data["выбранный порог"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Сдача против кандидата: менять или нет")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--extra-train-share", type=float, default=0.0,
                        help="Доля отложенных машин, отданная в обучение (build_local_split); "
                             "оценка идёт на остальных")
    parser.add_argument("--candidate", type=Path, nargs="+", required=True,
                        help="Чекпойнты кандидата (например, runs/res288/models/*.pt)")
    parser.add_argument("--candidate-projection", type=Path, default=None)
    parser.add_argument("--candidate-threshold", type=float, default=None,
                        help="Порог кандидата, если отчёт калибровки не передан")
    parser.add_argument("--candidate-threshold-report", type=Path, default=None,
                        help="threshold_choice.json кандидата (scripts/calibrate_threshold.py)")
    parser.add_argument("--current", type=Path, nargs="+",
                        default=[ROOT / p for p in SUBMISSION_CHECKPOINTS])
    parser.add_argument("--current-projection", type=Path, default=ROOT / SUBMISSION_PROJECTION)
    parser.add_argument("--current-threshold-report", type=Path, default=None,
                        help="threshold_choice.json сдачи, посчитанный на той же проверочной "
                             "выборке; без него — CALIBRATED_THRESHOLD")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "release_choice.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    official = load_official()
    rows = read_manifest(args.dataset / "train.csv", args.dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=42, extra_train_share=args.extra_train_share)

    candidate_threshold = args.candidate_threshold
    if candidate_threshold is None:
        candidate_threshold = threshold_from(args.candidate_threshold_report, CALIBRATED_THRESHOLD)

    current_threshold = threshold_from(args.current_threshold_report, CALIBRATED_THRESHOLD)
    report = {"порог сдачи": current_threshold, "порог кандидата": candidate_threshold}
    per_query = {}
    for name, checkpoints, projection, threshold in (
        ("сдача", args.current, args.current_projection, current_threshold),
        ("кандидат", args.candidate, args.candidate_projection, candidate_threshold),
    ):
        extractor, query, gallery = vectors_for(checkpoints, projection, split, args.workers)
        ranking_metrics, candidate_metrics, aps = evaluate(official, split, query, gallery, threshold)
        per_query[name] = aps
        entry = {
            "чекпойнты": [str(Path(c).name) for c in checkpoints],
            "mAP@10": round(ranking_metrics["mAP@10"], 4),
            "Rank-1": round(ranking_metrics["Rank-1"], 4),
            "F1": round(candidate_metrics["F1"], 4),
            "TNR": round(candidate_metrics["TNR"], 4),
            "балл кандидатов": round(0.7 * candidate_metrics["F1"] + 0.3 * candidate_metrics["TNR"], 4),
        }
        if not args.skip_benchmark:
            speed = extractor.benchmark(split.query[:256])
            entry["задержка batch=1, мс"] = round(speed["latency_ms_b1_median"], 2)
            entry["пропускная способность, кадр/с"] = round(speed["best_throughput_fps"], 1)
            entry["баллы за скорость из 20"] = performance_score(
                speed["latency_ms_b1_median"], speed["best_throughput_fps"])["performance_points_of_20"]
        report[name] = entry
        print(json.dumps({name: entry}, ensure_ascii=False), flush=True)
        # Второй ансамбль грузится в ту же видеокарту, поэтому первый нужно
        # отпустить явно: у 6 ГБ запаса на два набора весов нет.
        del extractor, query, gallery
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report["бутстрэп"] = bootstrap(per_query["сдача"], per_query["кандидат"])
    report["вердикт"], report["причины"] = verdict(report, with_speed=not args.skip_benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"бутстрэп": report["бутстрэп"], "вердикт": report["вердикт"],
                      "причины": report["причины"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
