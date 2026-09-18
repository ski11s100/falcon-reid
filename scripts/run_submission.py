"""Пакетная генерация файлов сдачи. Точка входа для проверки организаторами.

Ответ 40: организаторы передают каталог изображений и CSV целиком, решение
обрабатывает весь набор и возвращает три файла. API с вызовом по одному запросу
не требуется.

    python scripts/run_submission.py /data --output /data/submission

Ожидаемая структура входного каталога:
    /data/test_query.csv
    /data/test_gallery.csv
    /data/images/

На выходе: submission.csv, embeddings.npy, candidates.csv, manifest.json
и benchmark.json с замером производительности по методике организаторов.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.data import read_manifest  # noqa: E402
from falcon.extract import ExtractorConfig, build_extractor  # noqa: E402
from falcon.metrics import performance_score  # noqa: E402
from falcon.submit import (  # noqa: E402
    CALIBRATED_THRESHOLD,
    SubmissionConfig,
    build_ranking,
    validate_submission,
    write_submission,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация файлов сдачи ФАЛЬКОН")
    parser.add_argument("dataset", type=Path, help="Каталог с test_query.csv, test_gallery.csv, images/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        default=[Path("models/model-a.pt"), Path("models/model-b.pt")],
                        help="Веса модели. Несколько файлов — ансамбль")
    # По умолчанию — откалиброванный порог, а не «отказ на всём»: если жюри
    # запустит сдачу без флага, режим кандидатов не должен обнулиться.
    parser.add_argument("--threshold", type=float,
                        default=float(os.environ.get("FALCON_MATCH_THRESHOLD", CALIBRATED_THRESHOLD)),
                        help=f"Порог отказа (по умолчанию {CALIBRATED_THRESHOLD}, "
                             "откалиброван под ансамбль из сдачи)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument("--rerank", action="store_true",
                        help="Включить k-reciprocal. Только после проверки на валидации")
    parser.add_argument("--benchmark", action="store_true",
                        help="Дополнительно замерить latency и throughput")
    args = parser.parse_args()

    for path in args.checkpoints:
        if not path.is_file():
            parser.error(f"Не найден чекпоинт {path}")

    started = time.perf_counter()
    queries = read_manifest(args.dataset / "test_query.csv", args.dataset / "images")
    gallery = read_manifest(args.dataset / "test_gallery.csv", args.dataset / "images")
    print(json.dumps({"queries": len(queries), "gallery": len(gallery)}), flush=True)

    extractor = build_extractor(
        args.checkpoints,
        ExtractorConfig(batch_size=args.batch_size, num_workers=args.workers,
                        device=args.device, half=True,
                        flip_tta=not args.no_flip_tta),
    )

    # Порядок строго фиксирован: сначала все query в порядке test_query.csv,
    # затем вся gallery в порядке test_gallery.csv (ответ 27). Никакой
    # дополнительной сортировки ни на одном шаге.
    query_vectors = extractor.extract(queries)
    gallery_vectors = extractor.extract(gallery)
    embeddings = np.vstack([query_vectors, gallery_vectors]).astype(np.float32)

    config = SubmissionConfig(threshold=args.threshold, use_rerank=args.rerank)
    top_indices, top_scores = build_ranking(query_vectors, gallery_vectors, config)

    manifest = write_submission(args.output, queries, gallery, embeddings,
                                top_indices, top_scores, config,
                                model_version=extractor.metadata.get("architecture", "falcon"))
    report = validate_submission(args.output, len(queries), len(gallery))

    if args.benchmark:
        benchmark = extractor.benchmark(queries[:256])
        benchmark["scoring"] = performance_score(
            benchmark["latency_ms_b1_median"], benchmark["best_throughput_fps"])
        (args.output / "benchmark.json").write_text(
            json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(benchmark, ensure_ascii=False), flush=True)

    print(json.dumps({
        "manifest": {k: manifest[k] for k in
                     ("queries", "gallery", "embedding_dim", "accepted_queries",
                      "refused_queries", "refusal_rate")},
        "validation": report,
        "total_seconds": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False, indent=2))

    if not report["valid"]:
        raise SystemExit("Файлы сдачи не прошли самопроверку формата")


if __name__ == "__main__":
    main()
