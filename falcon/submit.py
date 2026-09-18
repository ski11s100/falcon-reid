"""Формирование трёх файлов сдачи в точном формате организаторов.

Формат подтверждён ответами 20, 21, 22, 24, 27, 29:

  embeddings.npy   float32, shape (len(test_query) + len(test_gallery), D).
                   Сначала ВСЕ query в порядке строк файла, затем ВСЯ gallery.
                   Никакой дополнительной сортировки. L2-нормализация не
                   обязательна, но сдаём нормированные — так честнее и дешевле.

  submission.csv   query_id, gallery_id_1 ... gallery_id_10.
                   РОВНО десять кандидатов на каждый запрос, БЕЗ ИСКЛЮЧЕНИЙ,
                   включая те, где мы уверены в отказе. Отказ здесь никак не
                   выражается — это файл только про ранжирование.

  candidates.csv   query_id, gallery_id, confidence.
                   Отказ = ПОЛНОЕ ОТСУТСТВИЕ строк для этого query_id.
                   Не пустая строка, не спецзначение. confidence — произвольная
                   шкала, важна только монотонность по уверенности.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Observation
from .metrics import TOP_K, l2_normalize
from .rerank import build_gallery_index, rerank_query


@dataclass
class SubmissionConfig:
    """Параметры сдачи.

    use_rerank выключен намеренно. На реалистичном локальном сплите лучшая
    конфигурация (k1=12, lambda=0.7) дала +0.0057 mAP@10, но парный бутстрэп по
    запросам даёт 95% интервал [-0.010; +0.022]: прирост неотличим от шума, а
    на предыдущем сплите та же конфигурация теряла 4.7%. Выигрыш сомнителен,
    потеря возможна — в сдаче метод не используется.

    Параметры по умолчанию — лучшие из проверенных, на случай ручного включения:
    прежние (k1=20, lambda=0.3) были худшими в сетке, mAP 0.487.
    """

    threshold: float | None = None
    use_rerank: bool = False
    rerank_pool: int = 100
    k1: int = 12
    k2: int = 6
    lambda_value: float = 0.7
    candidates_per_query: int = 1


def build_ranking(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    config: SubmissionConfig,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (индексы топ-10 для каждого запроса, их оценки уверенности)."""
    queries = l2_normalize(query_vectors)
    gallery = l2_normalize(gallery_vectors)
    if len(gallery) < TOP_K:
        raise ValueError(f"В галерее {len(gallery)} объектов, а сдавать нужно топ-{TOP_K}")

    top_indices = np.zeros((len(queries), TOP_K), dtype=np.int64)
    top_scores = np.zeros((len(queries), TOP_K), dtype=np.float32)

    if config.use_rerank:
        # Индекс соседства галереи строится один раз: галерея статична (ответ 38).
        index = build_gallery_index(gallery, k=max(config.k1 + 10, 30))
        for i, vector in enumerate(queries):
            order, _ = rerank_query(
                vector, index,
                k1=config.k1, k2=config.k2, lambda_value=config.lambda_value,
                candidate_pool=config.rerank_pool,
            )
            top_indices[i] = order[:TOP_K]
            # Уверенность — ВСЕГДА исходная косинусная близость, даже когда
            # порядок определён переранжированием. Оценки re-ranking лежат в
            # другой шкале (смесь расстояний со знаком минус), и порог, найденный
            # калибровкой на косинусе, к ним неприменим: однажды это отклонило
            # все 1110 запросов и обнулило режим кандидатов.
            # Монотонность по уверенности — единственное требование к confidence
            # (ответ 26), и косинус ему удовлетворяет.
            top_scores[i] = gallery[top_indices[i]] @ vector
            if progress and i % 200 == 0:
                print(json.dumps({"reranked": i + 1, "total": len(queries)}), flush=True)
    else:
        for start in range(0, len(queries), 256):
            block = queries[start : start + 256] @ gallery.T
            for offset, scores in enumerate(block):
                order = np.argsort(-scores, kind="stable")[:TOP_K]
                top_indices[start + offset] = order
                top_scores[start + offset] = scores[order]

    return top_indices, top_scores


def write_submission(
    output_dir: Path,
    queries: list[Observation],
    gallery: list[Observation],
    embeddings: np.ndarray,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    config: SubmissionConfig,
    model_version: str = "unknown",
) -> dict:
    """Пишет три файла и манифест с контрольными суммами."""
    output_dir.mkdir(parents=True, exist_ok=True)
    query_ids = [row.image_id for row in queries]
    gallery_ids = [row.image_id for row in gallery]

    if len(embeddings) != len(queries) + len(gallery):
        raise ValueError(
            f"embeddings.npy должен содержать {len(queries)} + {len(gallery)} = "
            f"{len(queries) + len(gallery)} строк, получено {len(embeddings)}"
        )
    if set(query_ids) & set(gallery_ids):
        raise ValueError("query_id и gallery_id пересекаются — это ошибка формата")

    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32, copy=False))

    with (output_dir / "submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, TOP_K + 1)])
        for i, qid in enumerate(query_ids):
            writer.writerow([qid] + [gallery_ids[j] for j in top_indices[i]])

    accepted = 0
    with (output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(query_ids):
            if config.threshold is None:
                # Без порога честнее отказать во всём, чем выдать необоснованные пары.
                continue
            keep = [j for j in range(TOP_K) if float(top_scores[i][j]) >= config.threshold]
            keep = keep[: max(1, config.candidates_per_query)] if keep else []
            if keep:
                accepted += 1
            for j in keep:
                # Никаких строк для отклонённого запроса не пишем вообще (ответ 20).
                writer.writerow([qid, gallery_ids[top_indices[i][j]],
                                 format(float(top_scores[i][j]), ".9g")])

    manifest = {
        "format": "falcon-submission-v1",
        "model": model_version,
        "queries": len(queries),
        "gallery": len(gallery),
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_dtype": str(embeddings.dtype),
        "threshold": config.threshold,
        "accepted_queries": accepted,
        "refused_queries": len(queries) - accepted,
        "refusal_rate": round((len(queries) - accepted) / max(1, len(queries)), 4),
        "reranking": {
            "enabled": config.use_rerank,
            "method": "k-reciprocal, потоково-совместимый (без связей query-query)",
            "k1": config.k1, "k2": config.k2, "lambda": config.lambda_value,
            "pool": config.rerank_pool,
        },
        "confidence_definition": ("косинусное сходство эмбеддингов, не калиброванная вероятность; "
                                  "переранжирование, если включено, меняет только порядок"),
        "refusal_encoding": "отсутствие строк для query_id в candidates.csv",
        "checksums": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in ("submission.csv", "candidates.csv", "embeddings.npy")
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_submission(output_dir: Path, expected_queries: int, expected_gallery: int) -> dict:
    """Самопроверка перед сдачей: расхождение формата сразу считается ошибкой."""
    output_dir = Path(output_dir)
    problems: list[str] = []

    embeddings = np.load(output_dir / "embeddings.npy")
    if embeddings.shape[0] != expected_queries + expected_gallery:
        problems.append(
            f"embeddings.npy: {embeddings.shape[0]} строк вместо "
            f"{expected_queries + expected_gallery}"
        )
    if embeddings.dtype != np.float32:
        problems.append(f"embeddings.npy: dtype {embeddings.dtype}, ожидается float32")
    if not np.isfinite(embeddings).all():
        problems.append("embeddings.npy: есть NaN или inf")

    with (output_dir / "submission.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != ["query_id"] + [f"gallery_id_{i}" for i in range(1, TOP_K + 1)]:
            problems.append(f"submission.csv: неверный заголовок {header}")
        submission_ids = set()
        rows = 0
        for row in reader:
            rows += 1
            if len(row) != TOP_K + 1:
                problems.append(f"submission.csv: строка {rows} содержит {len(row)} полей")
            if len(set(row[1:])) != TOP_K:
                problems.append(f"submission.csv: строка {rows} содержит повторяющиеся gallery_id")
            submission_ids.add(row[0])
        if rows != expected_queries:
            problems.append(f"submission.csv: {rows} строк вместо {expected_queries}")

    with (output_dir / "candidates.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != ["query_id", "gallery_id", "confidence"]:
            problems.append(f"candidates.csv: неверный заголовок {header}")
        answered = set()
        for row in reader:
            if len(row) != 3 or not row[1].strip() or not row[2].strip():
                problems.append(f"candidates.csv: пустые поля в строке {row} — отказ кодируется "
                                "отсутствием строки, а не пустыми значениями")
                continue
            answered.add(row[0])
        unknown = answered - submission_ids
        if unknown:
            problems.append(f"candidates.csv: query_id, которых нет в submission.csv: {sorted(unknown)[:3]}")

    return {
        "valid": not problems,
        "problems": problems,
        "queries": expected_queries,
        "gallery": expected_gallery,
        "answered_queries": len(answered),
        "refused_queries": expected_queries - len(answered),
    }
