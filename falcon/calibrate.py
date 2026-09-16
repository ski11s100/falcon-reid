"""Подбор порога отказа под формулу организаторов.

Балл за режим кандидатов (ответ 15): 10% x (0.7 x F1 + 0.3 x TNR).
Оптимизировать надо именно эту величину, а не F1 отдельно: они тянут в разные
стороны. Низкий порог отвечает на всё и обнуляет TNR; высокий порог отказывает
на всём и обнуляет F1. Максимум комбинации лежит между ними, и его положение
зависит от качества модели, поэтому переносить порог между моделями нельзя.

Дополнительно считается PR-AUC — порогонезависимая мера разделимости классов
«есть пара / нет пары». Организаторы публикуют её справочно, но для защиты она
полезнее самого порога: показывает, есть ли вообще сигнал для отказа.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import Identity, RefusalReport, evaluate_refusal


@dataclass
class CalibrationPoint:
    threshold: float
    f1: float
    tnr: float
    precision: float
    recall: float
    score: float
    answered: int
    refused: int

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 6),
            "F1": round(self.f1, 4),
            "TNR": round(self.tnr, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "combined_score": round(self.score, 4),
            "answered": self.answered,
            "refused": self.refused,
        }


def sweep_threshold(
    top_gallery_ids: dict[str, str],
    top_scores: dict[str, float],
    query_labels: dict[str, Identity],
    gallery_labels: dict[str, Identity],
    *,
    steps: int = 200,
) -> list[CalibrationPoint]:
    """Прогоняет пороги по всему диапазону оценок и считает метрики в каждом.

    top_gallery_ids / top_scores — верхний кандидат и его оценка для каждого
    запроса. Порог решает, попадёт ли запрос в candidates.csv вообще.
    """
    values = np.array(list(top_scores.values()), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Нет оценок для калибровки")

    lo, hi = float(values.min()), float(values.max())
    grid = np.linspace(lo, hi, steps) if hi > lo else np.array([lo])
    # Точка выше максимума добавляется всегда, в том числе когда все оценки
    # совпали: «отказать на всём» — валидная стратегия с гарантированным
    # TNR = 1, и её балл нужно знать, чтобы понимать, что дают ответы.
    thresholds = np.concatenate([grid, [np.nextafter(hi, np.inf)]])

    points: list[CalibrationPoint] = []
    for threshold in thresholds:
        candidates = {
            qid: [(top_gallery_ids[qid], score)]
            for qid, score in top_scores.items()
            if score >= threshold
        }
        report: RefusalReport = evaluate_refusal(candidates, query_labels, gallery_labels)
        points.append(CalibrationPoint(
            threshold=float(threshold), f1=report.f1, tnr=report.tnr,
            precision=report.precision, recall=report.recall,
            score=report.score_share, answered=len(candidates),
            refused=len(query_labels) - len(candidates),
        ))
    return points


def best_threshold(points: list[CalibrationPoint]) -> CalibrationPoint:
    """Лучшая точка по 0.7*F1 + 0.3*TNR; при равенстве — более осторожная."""
    return max(points, key=lambda p: (p.score, p.threshold))


def precision_recall_auc(
    top_scores: dict[str, float],
    query_labels: dict[str, Identity],
    gallery_labels: dict[str, Identity],
) -> float:
    """PR-AUC задачи «есть ли у запроса пара в галерее».

    Позитив — запрос с валидной кросс-камерной парой, негатив — open-set запрос.
    Оценка — уверенность верхнего кандидата.
    """
    gallery = list(gallery_labels.values())
    labels: list[int] = []
    scores: list[float] = []
    for qid, identity in query_labels.items():
        if qid not in top_scores:
            continue
        has_pair = any(
            g.vehicle_id == identity.vehicle_id and g.camera_id != identity.camera_id
            for g in gallery
        )
        labels.append(1 if has_pair else 0)
        scores.append(top_scores[qid])

    if not labels or len(set(labels)) < 2:
        raise ValueError("Для PR-AUC нужны и запросы с парой, и open-set запросы")

    order = np.argsort(-np.asarray(scores), kind="stable")
    y = np.asarray(labels)[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(y.sum()))
    # Ступенчатая интерполяция, как в average_precision_score.
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def calibrate(
    top_gallery_ids: dict[str, str],
    top_scores: dict[str, float],
    query_labels: dict[str, Identity],
    gallery_labels: dict[str, Identity],
) -> dict:
    """Полный отчёт калибровки: лучший порог, кривая и обоснование для защиты."""
    points = sweep_threshold(top_gallery_ids, top_scores, query_labels, gallery_labels)
    best = best_threshold(points)
    always_answer = min(points, key=lambda p: p.threshold)
    always_refuse = max(points, key=lambda p: p.threshold)

    return {
        "selected": best.as_dict(),
        "baselines": {
            "always_answer": always_answer.as_dict(),
            "always_refuse": always_refuse.as_dict(),
        },
        "gain_over_always_answer": round(best.score - always_answer.score, 4),
        "pr_auc": round(precision_recall_auc(top_scores, query_labels, gallery_labels), 4),
        "curve": [p.as_dict() for p in points[:: max(1, len(points) // 40)]],
        "rationale": (
            "Порог выбран максимизацией 0.7*F1 + 0.3*TNR — ровно той величины, по "
            "которой начисляются 10% за режим кандидатов. Калибровка выполнена на "
            "локальном сплите с долей open-set запросов около 20%, как в закрытом "
            "тесте. Метки закрытого теста не использовались."
        ),
    }
