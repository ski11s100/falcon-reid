"""Проверка официального протокола оценки против ответов организаторов.

Каждый тест проверяет конкретное правило из сводной таблицы Q&A, чтобы при
рефакторинге нельзя было случайно вернуться к «обычному» ReID-протоколу.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.metrics import (  # noqa: E402
    Identity,
    average_precision_at_k,
    evaluate_ranking,
    evaluate_refusal,
    mean_inverse_negative_penalty,
    performance_score,
    rank_from_embeddings,
)


class TestJunkFilter(unittest.TestCase):
    """Ответ 11: junk = то же ТС И та же камера, фильтр до усечения до десяти."""

    def test_same_camera_same_vehicle_is_dropped_not_penalised(self):
        query = {"q": Identity("A", "cam1")}
        gallery = {
            "junk": Identity("A", "cam1"),   # то же ТС, та же камера -> выбросить
            "pos": Identity("A", "cam2"),    # валидный позитив
            "neg": Identity("B", "cam1"),    # та же камера, другое ТС -> валидный негатив
        }
        # Модель поставила junk на первое место. Слот он занимать не должен.
        report = evaluate_ranking({"q": ["junk", "pos", "neg"]}, query, gallery)
        self.assertEqual(report.mAP, 1.0)
        self.assertEqual(report.rank_1, 1.0)

    def test_same_camera_other_vehicle_is_a_valid_negative(self):
        query = {"q": Identity("A", "cam1")}
        gallery = {"neg": Identity("B", "cam1"), "pos": Identity("A", "cam2")}
        report = evaluate_ranking({"q": ["neg", "pos"]}, query, gallery)
        # Позитив на второй позиции: AP = (1/2) / min(1, 10) = 0.5
        self.assertAlmostEqual(report.mAP, 0.5)
        self.assertEqual(report.rank_1, 0.0)
        self.assertEqual(report.rank_5, 1.0)


class TestExclusion(unittest.TestCase):
    """Ответы 13 и 22: запрос без валидного позитива исключается, а не получает AP=0."""

    def test_open_set_query_excluded_from_map(self):
        query = {"open": Identity("Z", "cam1"), "normal": Identity("A", "cam1")}
        gallery = {"pos": Identity("A", "cam2"), "other": Identity("B", "cam3")}
        report = evaluate_ranking(
            {"open": ["pos", "other"], "normal": ["pos", "other"]}, query, gallery
        )
        self.assertEqual(report.scored_queries, 1)
        self.assertEqual(report.excluded_queries, 1)
        self.assertEqual(report.mAP, 1.0)  # не размывается нулём от open-set запроса

    def test_query_whose_only_positive_is_junk_is_excluded(self):
        query = {"q": Identity("A", "cam1")}
        gallery = {"junk": Identity("A", "cam1"), "other": Identity("B", "cam2")}
        with self.assertRaises(ValueError):
            evaluate_ranking({"q": ["junk", "other"]}, query, gallery)


class TestAveragePrecision(unittest.TestCase):
    """Ответ 10: AP нормируется на min(n_pos, 10), усреднение macro."""

    def test_normalisation_by_min_npos_ten(self):
        # 20 позитивов, идеальный топ-10 -> AP должен быть ровно 1, а не 10/20.
        hits = np.ones(10, dtype=bool)
        self.assertAlmostEqual(average_precision_at_k(hits, n_pos=20), 1.0)

    def test_fewer_positives_than_ten(self):
        # 2 позитива на первых двух местах -> AP = (1/1 + 2/2) / 2 = 1.0
        hits = np.array([True, True] + [False] * 8)
        self.assertAlmostEqual(average_precision_at_k(hits, n_pos=2), 1.0)

    def test_miss_gives_zero(self):
        self.assertEqual(average_precision_at_k(np.zeros(10, dtype=bool), n_pos=3), 0.0)

    def test_partial(self):
        # Позитивы на местах 2 и 4 при n_pos=2: (1/2 + 2/4) / 2 = 0.5
        hits = np.array([False, True, False, True] + [False] * 6)
        self.assertAlmostEqual(average_precision_at_k(hits, n_pos=2), 0.5)


class TestRefusalMode(unittest.TestCase):
    """Ответы 19-26: F1/TNR micro, на уровне запроса, отказ = отсутствие строки."""

    def setUp(self):
        self.gallery = {
            "gA": Identity("A", "cam2"),
            "gB": Identity("B", "cam2"),
        }
        self.queries = {
            "q_hit": Identity("A", "cam1"),     # есть пара
            "q_miss": Identity("B", "cam1"),    # есть пара
            "q_open1": Identity("Z", "cam1"),   # пары нет
            "q_open2": Identity("Y", "cam1"),   # пары нет
        }

    def test_perfect_behaviour(self):
        candidates = {
            "q_hit": [("gA", 0.9)],
            "q_miss": [("gB", 0.8)],
            # два open-set запроса отсутствуют в файле = корректный отказ
        }
        report = evaluate_refusal(candidates, self.queries, self.gallery)
        self.assertEqual((report.tp, report.fp, report.fn, report.tn), (2, 0, 0, 2))
        self.assertEqual(report.f1, 1.0)
        self.assertEqual(report.tnr, 1.0)
        self.assertAlmostEqual(report.score_share, 1.0)

    def test_answering_open_set_query_is_a_false_positive(self):
        candidates = {
            "q_hit": [("gA", 0.9)],
            "q_miss": [("gB", 0.8)],
            "q_open1": [("gA", 0.7)],  # нельзя было отвечать
        }
        report = evaluate_refusal(candidates, self.queries, self.gallery)
        self.assertEqual((report.tp, report.fp, report.fn, report.tn), (2, 1, 0, 1))
        self.assertAlmostEqual(report.tnr, 0.5)

    def test_only_top_confidence_candidate_counts(self):
        # Верный ответ есть в списке, но не верхний по confidence -> это FP.
        candidates = {"q_hit": [("gA", 0.2), ("gB", 0.9)]}
        report = evaluate_refusal(candidates, self.queries, self.gallery)
        self.assertEqual(report.tp, 0)
        self.assertEqual(report.fp, 1)

    def test_extra_low_confidence_candidates_are_harmless(self):
        # Ответ 23: кандидаты ниже верхнего в расчёт не входят никак.
        lean = {"q_hit": [("gA", 0.9)]}
        padded = {"q_hit": [("gA", 0.9), ("gB", 0.1)]}
        self.assertEqual(
            evaluate_refusal(lean, self.queries, self.gallery).as_dict(),
            evaluate_refusal(padded, self.queries, self.gallery).as_dict(),
        )

    def test_silence_on_a_query_with_a_pair_is_a_false_negative(self):
        report = evaluate_refusal({}, self.queries, self.gallery)
        self.assertEqual((report.tp, report.fp, report.fn, report.tn), (0, 0, 2, 2))
        self.assertEqual(report.tnr, 1.0)  # отказал везде, но потерял все реальные пары
        self.assertEqual(report.f1, 0.0)

    def test_same_vehicle_same_camera_counts_as_correct(self):
        # Эталонный скрипт организаторов: верность верхнего кандидата — только
        # по vehicle_id, камера не проверяется (organizers/evaluate.py).
        gallery = {**self.gallery, "gA_same_cam": Identity("A", "cam1")}
        candidates = {"q_hit": [("gA_same_cam", 0.9)]}
        report = evaluate_refusal(candidates, self.queries, gallery)
        self.assertEqual(report.tp, 1)
        self.assertEqual(report.fp, 0)

    def test_always_answering_destroys_tnr(self):
        # Сценарий, который организаторы явно хотят наказать: всегда отдаём топ-1.
        candidates = {qid: [("gA", 0.5)] for qid in self.queries}
        report = evaluate_refusal(candidates, self.queries, self.gallery)
        self.assertEqual(report.tnr, 0.0)
        self.assertEqual(report.tp, 1)
        self.assertEqual(report.fp, 3)


class TestEmbeddingRanking(unittest.TestCase):
    def test_cosine_ranking_orders_by_similarity(self):
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        g = np.array([[0.0, 1.0], [0.9, 0.1], [1.0, 0.0]], dtype=np.float32)
        ranking = rank_from_embeddings(q, g, ["q"], ["far", "near", "exact"])
        self.assertEqual(ranking["q"], ["exact", "near", "far"])

    def test_minp_rewards_shallow_last_hit(self):
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        g = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
        labels_q = [Identity("A", "cam1")]
        labels_g = [Identity("A", "cam2"), Identity("A", "cam3"), Identity("B", "cam2")]
        # Оба позитива на первых двух местах -> INP = 2/2 = 1.0
        self.assertAlmostEqual(
            mean_inverse_negative_penalty(q, g, labels_q, labels_g), 1.0
        )


class TestPerformanceScore(unittest.TestCase):
    """Ответ 34: пороги 40/80 мс и 100/50 FPS."""

    def test_full_marks(self):
        result = performance_score(latency_ms_b1=25.0, throughput_fps=180.0)
        self.assertEqual(result["performance_points_of_20"], 20.0)

    def test_zero_marks(self):
        result = performance_score(latency_ms_b1=95.0, throughput_fps=30.0)
        self.assertEqual(result["performance_points_of_20"], 0.0)

    def test_linear_middle(self):
        result = performance_score(latency_ms_b1=60.0, throughput_fps=75.0)
        self.assertAlmostEqual(result["latency_points_of_10"], 5.0)
        self.assertAlmostEqual(result["throughput_points_of_10"], 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
