"""Официальный протокол оценки, восстановленный по ответам организаторов.

Источник — сводная таблица вопросов участников (ответы 10-29). Ключевые правила,
которые отличают этот расчёт от «обычного» ReID-протокола:

1. Основная метрика — mAP@10 по submission.csv, а не по полному ранжированию.
2. Junk-фильтр убирает из галереи объекты, у которых ОДНОВРЕМЕННО совпадают
   vehicle_id и camera_id с запросом. Объекты той же камеры с другим ТС остаются:
   это валидные (и самые сложные) негативы.
3. Junk-фильтр применяется ДО усечения до десяти позиций, поэтому junk-объект
   не занимает слот в топ-10 и не штрафуется.
4. AP нормируется на min(n_pos, 10), иначе ТС, снятое восемью камерами, не может
   получить AP = 1 даже при идеальной выдаче.
5. Запросы, у которых после фильтрации не осталось позитивов, ИСКЛЮЧАЮТСЯ из
   mAP/Rank (не получают AP = 0) и оцениваются только в режиме отказа.
6. F1 и TNR — micro, на уровне запроса, по верхнему кандидату из candidates.csv.
   Отказ кодируется отсутствием строк для query_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOP_K = 10


@dataclass(frozen=True)
class Identity:
    """Скрытая разметка объекта: кто это и с какой камеры снят."""

    vehicle_id: str
    camera_id: str | None = None


def _valid_positives(query: Identity, gallery: list[Identity]) -> np.ndarray:
    """Маска валидных позитивов в галерее (то же ТС, но не та же камера)."""
    return np.array(
        [g.vehicle_id == query.vehicle_id and g.camera_id != query.camera_id for g in gallery],
        dtype=bool,
    )


def _junk(query: Identity, gallery: list[Identity]) -> np.ndarray:
    """Маска junk-объектов: то же ТС И та же камера — выбрасываются из ранжирования."""
    return np.array(
        [g.vehicle_id == query.vehicle_id and g.camera_id == query.camera_id for g in gallery],
        dtype=bool,
    )


def average_precision_at_k(hits: np.ndarray, n_pos: int, k: int = TOP_K) -> float:
    """AP@k по булевой маске попаданий в уже отфильтрованном топ-k.

    Нормировка на min(n_pos, k) — ровно как в официальном скрипте.
    """
    if n_pos <= 0:
        raise ValueError("AP не определён для запроса без валидных позитивов")
    hits = np.asarray(hits, dtype=bool)[:k]
    if not hits.any():
        return 0.0
    ranks = np.flatnonzero(hits) + 1
    precisions = np.arange(1, len(ranks) + 1) / ranks
    return float(precisions.sum() / min(n_pos, k))


@dataclass
class RankingReport:
    """Результат оценки ранжирования."""

    mAP: float
    rank_1: float
    rank_5: float
    scored_queries: int
    excluded_queries: int
    per_query: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "mAP@10": round(self.mAP, 6),
            "Rank-1": round(self.rank_1, 6),
            "Rank-5": round(self.rank_5, 6),
            "scored_queries": self.scored_queries,
            "excluded_queries": self.excluded_queries,
        }


def evaluate_ranking(
    ranking: dict[str, list[str]],
    query_labels: dict[str, Identity],
    gallery_labels: dict[str, Identity],
    k: int = TOP_K,
) -> RankingReport:
    """Считает mAP@10, Rank-1 и Rank-5 по готовому ранжированию.

    ranking: query_id -> упорядоченный список gallery_id (то, что лежит в
    submission.csv). Junk организаторы выбрасывают сами, поэтому здесь список
    фильтруется и только потом усекается до k.
    """
    gallery_ids = list(gallery_labels)
    index = {gid: i for i, gid in enumerate(gallery_ids)}
    gallery = [gallery_labels[gid] for gid in gallery_ids]

    aps: list[float] = []
    top1: list[bool] = []
    top5: list[bool] = []
    per_query: dict[str, float] = {}
    excluded = 0

    for qid, identity in query_labels.items():
        positives = _valid_positives(identity, gallery)
        n_pos = int(positives.sum())
        if n_pos == 0:
            # Open-set запрос либо единственный позитив оказался junk: не в mAP.
            excluded += 1
            continue

        junk = _junk(identity, gallery)
        candidates = ranking.get(qid, [])
        kept = [gid for gid in candidates if gid in index and not junk[index[gid]]][:k]
        hits = np.array([positives[index[gid]] for gid in kept], dtype=bool)

        ap = average_precision_at_k(hits, n_pos, k)
        aps.append(ap)
        per_query[qid] = ap
        top1.append(bool(hits[:1].any()))
        top5.append(bool(hits[:5].any()))

    if not aps:
        raise ValueError("Ни одного запроса с валидным позитивом — оценка невозможна")

    return RankingReport(
        mAP=float(np.mean(aps)),
        rank_1=float(np.mean(top1)),
        rank_5=float(np.mean(top5)),
        scored_queries=len(aps),
        excluded_queries=excluded,
        per_query=per_query,
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Ожидается матрица эмбеддингов формы (N, D)")
    if not np.isfinite(matrix).all():
        raise ValueError("В эмбеддингах есть NaN или inf")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def rank_from_embeddings(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    query_ids: list[str],
    gallery_ids: list[str],
    k: int = TOP_K,
) -> dict[str, list[str]]:
    """Базовое ранжирование косинусом. Ничьи разрешаются порядком test_gallery.csv."""
    q = l2_normalize(query_vectors)
    g = l2_normalize(gallery_vectors)
    ranking: dict[str, list[str]] = {}
    # Батчами, чтобы не аллоцировать матрицу «запросы x галерея» целиком.
    for start in range(0, len(q), 512):
        chunk = q[start : start + 512] @ g.T
        for offset, scores in enumerate(chunk):
            order = np.argsort(-scores, kind="stable")[:k]
            ranking[query_ids[start + offset]] = [gallery_ids[j] for j in order]
    return ranking


def mean_inverse_negative_penalty(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    query_labels: list[Identity],
    gallery_labels: list[Identity],
) -> float:
    """mINP по полному ранжированию: насколько глубоко лежит последний верный ответ.

    Публикуется организаторами справочно, в баллы не идёт, но честно показывает
    качество самого представления без ограничения топ-10.
    """
    q = l2_normalize(query_vectors)
    g = l2_normalize(gallery_vectors)
    values: list[float] = []
    for i, identity in enumerate(query_labels):
        positives = _valid_positives(identity, gallery_labels)
        n_pos = int(positives.sum())
        if n_pos == 0:
            continue
        keep = ~_junk(identity, gallery_labels)
        scores = q[i] @ g[keep].T
        order = np.argsort(-scores, kind="stable")
        hit_positions = np.flatnonzero(positives[keep][order]) + 1
        values.append(n_pos / float(hit_positions[-1]))
    if not values:
        raise ValueError("Нет запросов с валидными позитивами для mINP")
    return float(np.mean(values))


@dataclass
class RefusalReport:
    """Результат оценки режима предложения кандидатов."""

    f1: float
    tnr: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    open_set_queries: int

    @property
    def score_share(self) -> float:
        """Доля от 10% за режим кандидатов: 0.7 * F1 + 0.3 * TNR."""
        return 0.7 * self.f1 + 0.3 * self.tnr

    def as_dict(self) -> dict:
        return {
            "F1": round(self.f1, 6),
            "TNR": round(self.tnr, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "open_set_queries": self.open_set_queries,
            "candidate_mode_share": round(self.score_share, 6),
        }


def evaluate_refusal(
    candidates: dict[str, list[tuple[str, float]]],
    query_labels: dict[str, Identity],
    gallery_labels: dict[str, Identity],
) -> RefusalReport:
    """F1 и TNR режима кандидатов. Micro, на уровне запроса, по верхнему кандидату.

    candidates: query_id -> [(gallery_id, confidence), ...]. Запрос, которого нет
    в словаре (или с пустым списком), считается отказом.

    Совпадает с эталонным скриптом организаторов (organizers/evaluate.py,
    candidate_metrics): верхний кандидат верен, если это тот же vehicle_id —
    камера НЕ проверяется. Пара у запроса есть, если в галерее остался хотя бы
    один позитив с другой камеры. Раньше здесь требовалась и другая камера
    («строгое» толкование, до выхода эталонного скрипта); это занижало F1
    с 0.82 до 0.34, потому что у 44% запросов ближайший кандидат — та же
    машина с той же камеры.
    """
    gallery_ids = list(gallery_labels)
    index = {gid: i for i, gid in enumerate(gallery_ids)}
    gallery = [gallery_labels[gid] for gid in gallery_ids]

    tp = fp = fn = tn = 0
    open_set = 0

    for qid, identity in query_labels.items():
        has_pair = bool(_valid_positives(identity, gallery).any())
        if not has_pair:
            open_set += 1

        rows = candidates.get(qid) or []
        if not rows:
            # Отказ = отсутствие строк для query_id.
            if has_pair:
                fn += 1
            else:
                tn += 1
            continue

        best_gid, _ = max(rows, key=lambda row: row[1])
        correct = (
            has_pair
            and best_gid in index
            and gallery[index[best_gid]].vehicle_id == identity.vehicle_id
        )
        if correct:
            tp += 1
        else:
            # Верхний кандидат неверен, либо у запроса вообще нет пары.
            fp += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    # TNR считается только по запросам без пары: FP среди них — это open_set - tn.
    tnr = tn / float(open_set) if open_set else 0.0

    return RefusalReport(
        f1=f1,
        tnr=tnr,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        open_set_queries=open_set,
    )


def performance_score(latency_ms_b1: float, throughput_fps: float) -> dict:
    """20% за производительность по формуле организаторов (ответ 34).

    Latency (batch=1, полный цикл): <=40 мс — полный балл, 40-80 — линейно, >80 — 0.
    Throughput (лучший FPS): >=100 — полный балл, 50-100 — линейно, <50 — 0.
    """
    latency_share = float(np.clip((80.0 - latency_ms_b1) / 40.0, 0.0, 1.0))
    throughput_share = float(np.clip((throughput_fps - 50.0) / 50.0, 0.0, 1.0))
    return {
        "latency_ms_b1": round(latency_ms_b1, 3),
        "throughput_fps": round(throughput_fps, 2),
        "latency_points_of_10": round(10.0 * latency_share, 2),
        "throughput_points_of_10": round(10.0 * throughput_share, 2),
        "performance_points_of_20": round(10.0 * latency_share + 10.0 * throughput_share, 2),
    }


def scoreboard(ranking: RankingReport, refusal: RefusalReport, performance: dict | None = None) -> dict:
    """Пересчёт метрик в баллы конкурса: 45% mAP + 10% кандидаты + 20% скорость."""
    board = {
        "accuracy_points_of_45": round(45.0 * ranking.mAP, 2),
        "candidate_points_of_10": round(10.0 * refusal.score_share, 2),
        "ranking": ranking.as_dict(),
        "refusal": refusal.as_dict(),
    }
    if performance is not None:
        board["performance"] = performance
    return board
