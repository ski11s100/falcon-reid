"""Полный цикл: аудит данных, обучение, калибровка порога, файлы сдачи.

Одна команда от сырого датасета до готовой сдачи:

    python scripts/run_pipeline.py D:/falcon-data --output runs/v1

Этапы:
  1. аудит датасета и проверка пригодности для кросс-камерного протокола;
  2. обучение с локальной валидацией по официальной метрике mAP@10;
  3. калибровка порога отказа под формулу 0.7*F1 + 0.3*TNR;
  4. A/B-проверка k-reciprocal re-ranking на локальном сплите;
  5. проверка на остаточные признаки номера (защита от дисквалификации);
  6. генерация submission.csv, embeddings.npy, candidates.csv;
  7. замер производительности по методике организаторов.

Каждый этап пишет свой отчёт в каталог результата и может быть пропущен, если
результат предыдущего запуска уже есть.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.calibrate import calibrate  # noqa: E402
from falcon.data import audit, build_local_split, read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import (  # noqa: E402
    Identity,
    evaluate_ranking,
    l2_normalize,
    mean_inverse_negative_penalty,
    performance_score,
    rank_from_embeddings,
)
from falcon.submit import (  # noqa: E402
    SubmissionConfig,
    build_ranking,
    validate_submission,
    write_submission,
)
from falcon.train import TrainConfig, train  # noqa: E402


def stage(name: str) -> None:
    print(f"\n{'=' * 64}\n{name}\n{'=' * 64}", flush=True)


def step_audit(dataset: Path, output: Path) -> dict:
    stage("1. Аудит датасета")
    rows = read_manifest(dataset / "train.csv", dataset / "images", require_labels=True)
    report = audit(rows)

    problems: list[str] = []
    if not report["camera_id_available"]:
        problems.append("нет camera_id: кросс-камерную валидацию построить нельзя")
    if report["identities_with_multiple_cameras"] < report["identities"] * 0.5:
        problems.append("меньше половины ТС сняты двумя камерами: валидация будет слабой")
    if report["singleton_identities"] > report["identities"] * 0.1:
        problems.append("много ТС с единственным снимком: они бесполезны для triplet")
    report["problems"] = problems

    (output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if problems:
        print("\nВНИМАНИЕ:", "; ".join(problems), flush=True)
    return report


def step_train(dataset: Path, output: Path, config: TrainConfig, device: str,
               resume: Path | None = None) -> Path:
    stage("2. Обучение")
    checkpoint = output / "model" / "best.pt"
    if checkpoint.is_file():
        print(f"Чекпоинт уже есть: {checkpoint}. Обучение пропущено.", flush=True)
        return checkpoint
    if resume is not None:
        print(json.dumps({"resume_from": str(resume)}, ensure_ascii=False), flush=True)
    result = train(dataset, output / "model", config, device=device, resume=resume)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return checkpoint


def local_vectors(checkpoints: list[Path], dataset: Path, seed: int, device: str, workers: int):
    """Эмбеддинги локального сплита: нужны и для калибровки, и для A/B re-rank."""
    rows = read_manifest(dataset / "train.csv", dataset / "images", require_labels=True)
    split = build_local_split(rows, seed=seed)
    extractor = build_extractor(
        checkpoints,
        ExtractorConfig(batch_size=64, num_workers=workers, device=device,
                        half=True, flip_tta=True),
    )
    gallery = l2_normalize(extractor.extract(split.gallery, progress=False))
    query = l2_normalize(extractor.extract(split.query, progress=False))
    return split, query, gallery, extractor


def step_validate(split, query, gallery, output: Path) -> dict:
    stage("3. Локальная оценка по официальному протоколу")
    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    ql = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.query}
    gl = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.gallery}

    report = evaluate_ranking(rank_from_embeddings(query, gallery, qids, gids), ql, gl)
    result = report.as_dict()
    result["mINP"] = round(
        mean_inverse_negative_penalty(query, gallery, list(ql.values()), list(gl.values())), 6)
    result["accuracy_points_of_45"] = round(45.0 * report.mAP, 2)

    (output / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def step_calibrate(split, query, gallery, output: Path) -> dict:
    stage("4. Калибровка порога отказа")
    gids = [r.image_id for r in split.gallery]
    similarity = query @ gallery.T
    top_ids: dict[str, str] = {}
    top_scores: dict[str, float] = {}
    for i, row in enumerate(split.query):
        best = int(np.argmax(similarity[i]))
        top_ids[row.image_id] = gids[best]
        top_scores[row.image_id] = float(similarity[i][best])

    ql = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.query}
    gl = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.gallery}
    report = calibrate(top_ids, top_scores, ql, gl)

    (output / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("selected", "baselines", "gain_over_always_answer", "pr_auc")},
                     ensure_ascii=False, indent=2), flush=True)
    return report


def step_rerank_ab(split, query, gallery, output: Path) -> dict:
    stage("5. A/B-проверка k-reciprocal re-ranking")
    qids = [r.image_id for r in split.query]
    gids = [r.image_id for r in split.gallery]
    ql = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.query}
    gl = {r.image_id: Identity(r.vehicle_id, r.camera_id) for r in split.gallery}

    baseline = evaluate_ranking(rank_from_embeddings(query, gallery, qids, gids), ql, gl)
    results = {"без re-ranking": round(baseline.mAP, 6)}
    best_config, best_map = None, baseline.mAP

    for k1 in (12, 20):
        for lam in (0.3, 0.5, 0.7):
            config = SubmissionConfig(use_rerank=True, k1=k1, k2=6, lambda_value=lam,
                                      rerank_pool=min(100, len(gallery)))
            indices, _ = build_ranking(query, gallery, config, progress=False)
            ranking = {qids[i]: [gids[j] for j in indices[i]] for i in range(len(qids))}
            value = evaluate_ranking(ranking, ql, gl).mAP
            results[f"k1={k1}, lambda={lam}"] = round(value, 6)
            # Порог значимости: прирост меньше 1% относительного — шум одного
            # сплита, а не реальное улучшение. Переранжирование усложняет
            # пайплайн и его стоит включать только за ощутимую прибавку.
            if value > best_map * 1.01:
                best_map, best_config = value, {"k1": k1, "k2": 6, "lambda": lam}

    verdict = {
        "results": results,
        "baseline_mAP": round(baseline.mAP, 6),
        "best_mAP": round(best_map, 6),
        "recommended": best_config,
        "enable_rerank": best_config is not None,
        "note": ("re-ranking включается только при приросте больше 1% относительного: "
                 "на малой галерее он систематически ухудшает метрику, а мелкий "
                 "плюс неотличим от шума одного сплита"),
    }
    (output / "rerank_ab.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)
    return verdict


def step_plate_check(split, extractor, output: Path) -> dict:
    stage("6. Проверка на остаточные признаки номера")
    from falcon.data import load_crop
    from falcon.explain import control_masks, masking_robustness

    model = extractor.explain_model
    sample = split.query[: min(64, len(split.query))]
    batch = torch.stack([extractor.transform(load_crop(r, target=extractor.config.size))
                         for r in sample]).to(extractor.device)

    size = extractor.config.size
    result = masking_robustness(model, batch, control_masks(size[0], size[1]))

    plate = result.get("зона номера", 1.0)
    controls = [v for k, v in result.items() if k != "зона номера"]
    margin = float(np.mean(controls)) - plate

    verdict = {
        "similarity_after_masking": {k: round(v, 4) for k, v in result.items()},
        "plate_minus_controls": round(-margin, 4),
        "verdict": (
            "зона номера не выделяется среди контрольных областей"
            if margin < 0.03 else
            "закрашивание зоны номера бьёт по признаку сильнее контрольных областей — "
            "разобраться до защиты, это основание для дисквалификации по разделу 9 ТЗ"
        ),
        "method": ("сравнивается косинус между исходным признаком и признаком после "
                   "закрашивания области; контрольные области имеют ту же площадь"),
    }
    (output / "plate_check.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)
    return verdict


def step_submit(dataset: Path, checkpoints: list[Path], output: Path, threshold: float | None,
                rerank: dict, device: str, workers: int, benchmark: bool) -> dict:
    stage("7. Файлы сдачи")
    queries = read_manifest(dataset / "test_query.csv", dataset / "images")
    gallery = read_manifest(dataset / "test_gallery.csv", dataset / "images")

    extractor = build_extractor(
        checkpoints,
        ExtractorConfig(batch_size=32, num_workers=workers, device=device,
                        half=True, flip_tta=True),
    )
    query_vectors = extractor.extract(queries)
    gallery_vectors = extractor.extract(gallery)
    embeddings = np.vstack([query_vectors, gallery_vectors]).astype(np.float32)

    recommended = rerank.get("recommended") or {}
    config = SubmissionConfig(
        threshold=threshold,
        use_rerank=bool(rerank.get("enable_rerank")),
        k1=recommended.get("k1", 20),
        k2=recommended.get("k2", 6),
        lambda_value=recommended.get("lambda", 0.3),
    )
    indices, scores = build_ranking(query_vectors, gallery_vectors, config)
    submission_dir = output / "submission"
    if submission_dir.exists():
        shutil.rmtree(submission_dir)

    manifest = write_submission(submission_dir, queries, gallery, embeddings, indices,
                                scores, config, model_version=" + ".join(c.name for c in checkpoints))
    report = validate_submission(submission_dir, len(queries), len(gallery))
    print(json.dumps({"manifest": manifest, "validation": report},
                     ensure_ascii=False, indent=2), flush=True)

    if benchmark:
        stage("8. Замер производительности")
        result = extractor.benchmark(queries[: min(256, len(queries))])
        result["scoring"] = performance_score(result["latency_ms_b1_median"],
                                              result["best_throughput_fps"])
        (output / "benchmark.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    return {"manifest": manifest, "validation": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный цикл решения ФАЛЬКОН")
    parser.add_argument("dataset", type=Path, help="Каталог с CSV и images/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        help="Готовые веса: обучение пропускается. Несколько — ансамбль")
    parser.add_argument("--resume", type=Path,
                        help="Чекпоинт предобучения; классификатор не переносится")
    args = parser.parse_args()

    for name in ("train.csv", "test_query.csv", "test_gallery.csv"):
        if not (args.dataset / name).is_file():
            parser.error(f"Не найден {args.dataset / name}")
    if not (args.dataset / "images").is_dir():
        parser.error(f"Не найден каталог {args.dataset / 'images'}")

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    step_audit(args.dataset, args.output)
    if args.checkpoints:
        stage("2. Обучение пропущено, используются готовые веса")
        for path in args.checkpoints:
            if not path.is_file():
                parser.error(f"Не найден чекпоинт {path}")
        print(json.dumps({"checkpoints": [str(c) for c in args.checkpoints],
                          "ensemble": len(args.checkpoints) > 1}, ensure_ascii=False), flush=True)
        checkpoints = list(args.checkpoints)
    else:
        config = TrainConfig(epochs=args.epochs, num_workers=args.workers, seed=args.seed)
        checkpoints = [step_train(args.dataset, args.output, config, args.device,
                                  resume=args.resume)]

    split, query, gallery, extractor = local_vectors(
        checkpoints, args.dataset, args.seed, args.device, args.workers)

    validation = step_validate(split, query, gallery, args.output)
    calibration = step_calibrate(split, query, gallery, args.output)
    rerank = step_rerank_ab(split, query, gallery, args.output)
    step_plate_check(split, extractor, args.output)

    threshold = calibration["selected"]["threshold"]
    result = step_submit(args.dataset, checkpoints, args.output, threshold, rerank,
                         args.device, args.workers, not args.skip_benchmark)

    stage("Итог")
    summary = {
        "validation": validation,
        "threshold": threshold,
        "candidate_mode_score": calibration["selected"]["combined_score"],
        "rerank_enabled": rerank["enable_rerank"],
        "submission_valid": result["validation"]["valid"],
        "output": str(args.output),
        "total_minutes": round((time.perf_counter() - started) / 60, 1),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
