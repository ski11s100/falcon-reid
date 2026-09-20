"""Проверка формата сдачи и построения локального сплита.

Ошибка формата обнуляет результат независимо от качества модели, поэтому все
правила из ответов 20-29 закреплены тестами, а не только комментариями.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.calibrate import calibrate, sweep_threshold  # noqa: E402
from falcon.data import Observation  # noqa: E402
from falcon.metrics import TOP_K, Identity  # noqa: E402
from falcon.rerank import build_gallery_index, rerank_query  # noqa: E402
from falcon.submit import (  # noqa: E402
    SubmissionConfig,
    build_ranking,
    validate_submission,
    write_submission,
)


def fake_rows(prefix: str, count: int) -> list[Observation]:
    return [
        Observation(row_id=i, image_id=f"{prefix}_{i:04d}.jpg",
                    path=Path(f"/nonexistent/{prefix}_{i}.jpg"), bbox=(0.0, 0.0, 10.0, 10.0))
        for i in range(count)
    ]


class TestSubmissionFiles(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.queries = fake_rows("q", 12)
        self.gallery = fake_rows("g", 30)
        self.query_vectors = rng.normal(size=(12, 64)).astype(np.float32)
        self.gallery_vectors = rng.normal(size=(30, 64)).astype(np.float32)
        self.embeddings = np.vstack([self.query_vectors, self.gallery_vectors]).astype(np.float32)

    def _write(self, threshold, directory):
        config = SubmissionConfig(threshold=threshold, use_rerank=False)
        indices, scores = build_ranking(self.query_vectors, self.gallery_vectors, config, progress=False)
        manifest = write_submission(Path(directory), self.queries, self.gallery,
                                    self.embeddings, indices, scores, config)
        return manifest, validate_submission(Path(directory), len(self.queries), len(self.gallery))

    def test_submission_always_has_ten_candidates_even_when_refusing(self):
        """Ответ 21: submission.csv всегда 10 кандидатов, отказ в нём не выражается."""
        with tempfile.TemporaryDirectory() as directory:
            # Порог выше любого косинуса: отказываем на всех запросах.
            manifest, report = self._write(2.0, directory)
            self.assertTrue(report["valid"], report["problems"])
            self.assertEqual(manifest["refused_queries"], len(self.queries))

            with (Path(directory) / "submission.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            # Без заголовка: так читает эталонный скрипт организаторов.
            self.assertEqual(len(rows), len(self.queries))
            self.assertNotEqual(rows[0][0], "query_id")
            for row in rows:
                self.assertEqual(len(row) - 1, TOP_K)
                self.assertTrue(all(value.strip() for value in row))

    def test_refusal_is_absence_of_rows_not_empty_fields(self):
        """Ответ 20: отказ = отсутствие строк, не пустой gallery_id."""
        with tempfile.TemporaryDirectory() as directory:
            self._write(2.0, directory)
            with (Path(directory) / "candidates.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ["query_id", "gallery_id", "confidence"])
            self.assertEqual(len(rows), 1, "При полном отказе не должно быть ни одной строки данных")

    def test_embeddings_order_is_queries_then_gallery(self):
        """Ответ 27: сначала все query, затем вся gallery, без сортировки."""
        with tempfile.TemporaryDirectory() as directory:
            self._write(0.0, directory)
            saved = np.load(Path(directory) / "embeddings.npy")
            self.assertEqual(saved.shape, (len(self.queries) + len(self.gallery), 64))
            self.assertEqual(saved.dtype, np.float32)
            np.testing.assert_allclose(saved[: len(self.queries)], self.query_vectors, rtol=1e-6)
            np.testing.assert_allclose(saved[len(self.queries):], self.gallery_vectors, rtol=1e-6)

    def test_manifest_declares_vector_expansion(self):
        """Манифест обязан описывать обогащение векторов: жюри должно видеть,
        что порядок кандидатов строится не по голому косинусу."""
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._write(0.0, directory)
            expansion = manifest["vector_expansion"]
            self.assertEqual(expansion["dba_k"], SubmissionConfig().dba_k)
            self.assertEqual(expansion["qe_k"], SubmissionConfig().qe_k)
            self.assertIn("до обогащения", manifest["confidence_definition"])

    def test_embedding_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SubmissionConfig(use_rerank=False)
            indices, scores = build_ranking(self.query_vectors, self.gallery_vectors, config, progress=False)
            with self.assertRaises(ValueError):
                write_submission(Path(directory), self.queries, self.gallery,
                                 self.embeddings[:-1], indices, scores, config)

    def test_overlapping_query_and_gallery_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SubmissionConfig(use_rerank=False)
            indices, scores = build_ranking(self.query_vectors, self.gallery_vectors, config, progress=False)
            with self.assertRaises(ValueError):
                write_submission(Path(directory), self.queries, self.queries[:30] + self.gallery[:0],
                                 self.embeddings[: 2 * len(self.queries)], indices, scores, config)

    def test_gallery_smaller_than_ten_is_rejected(self):
        config = SubmissionConfig(use_rerank=False)
        with self.assertRaises(ValueError):
            build_ranking(self.query_vectors, self.gallery_vectors[:5], config, progress=False)


class TestRerankCompliance(unittest.TestCase):
    """Ответ 38: выдача запроса не должна зависеть от других запросов."""

    def test_result_independent_of_other_queries(self):
        rng = np.random.default_rng(7)
        gallery = rng.normal(size=(120, 32)).astype(np.float32)
        queries = rng.normal(size=(5, 32)).astype(np.float32)
        index = build_gallery_index(gallery, k=20)

        alone, _ = rerank_query(queries[0], index, candidate_pool=50)
        # Тот же запрос, но индекс построен заново — и другие запросы никак
        # не участвуют в вычислении. Результат обязан совпасть побитово.
        again, _ = rerank_query(queries[0], build_gallery_index(gallery, k=20), candidate_pool=50)
        np.testing.assert_array_equal(alone, again)

    def test_returns_full_ordering(self):
        rng = np.random.default_rng(11)
        gallery = rng.normal(size=(60, 16)).astype(np.float32)
        index = build_gallery_index(gallery, k=15)
        order, scores = rerank_query(rng.normal(size=16).astype(np.float32), index)
        self.assertEqual(len(order), 60)
        self.assertEqual(len(set(order.tolist())), 60)
        self.assertTrue(np.all(np.diff(scores) <= 1e-6), "Оценки должны убывать")


class TestConfidenceScale(unittest.TestCase):
    """Уверенность обязана быть в одной шкале с калибровкой порога.

    Переранжирование выдаёт оценки в другой шкале (смесь расстояний со знаком
    минус). Однажды оно включилось автоматически, и порог, найденный на
    косинусе, отклонил все 1110 запросов — режим кандидатов обнулился.
    """

    def setUp(self):
        rng = np.random.default_rng(3)
        self.queries = rng.normal(size=(20, 48)).astype(np.float32)
        self.gallery = rng.normal(size=(200, 48)).astype(np.float32)

    def test_scores_are_cosine_with_and_without_rerank(self):
        plain, _ = build_ranking(self.queries, self.gallery,
                                 SubmissionConfig(use_rerank=False), progress=False)
        _, plain_scores = build_ranking(self.queries, self.gallery,
                                        SubmissionConfig(use_rerank=False), progress=False)
        _, rerank_scores = build_ranking(self.queries, self.gallery,
                                         SubmissionConfig(use_rerank=True, rerank_pool=80),
                                         progress=False)
        for name, scores in (("без re-rank", plain_scores), ("с re-rank", rerank_scores)):
            self.assertTrue((scores >= -1.0001).all() and (scores <= 1.0001).all(),
                            f"{name}: оценки вне диапазона косинуса: "
                            f"[{scores.min():.3f}, {scores.max():.3f}]")
            self.assertGreater(scores.max(), 0.0,
                               f"{name}: все оценки неположительные — шкала не косинусная")

    def test_rerank_scores_match_recomputed_cosine(self):
        indices, scores = build_ranking(self.queries, self.gallery,
                                        SubmissionConfig(use_rerank=True, rerank_pool=80),
                                        progress=False)
        from falcon.metrics import l2_normalize
        q = l2_normalize(self.queries)
        g = l2_normalize(self.gallery)
        for i in range(len(q)):
            expected = g[indices[i]] @ q[i]
            np.testing.assert_allclose(scores[i], expected, atol=1e-5)


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.gallery_labels = {"g1": Identity("A", "c2"), "g2": Identity("B", "c2")}
        self.query_labels = {
            "q1": Identity("A", "c1"),   # есть пара
            "q2": Identity("B", "c1"),   # есть пара
            "q3": Identity("Z", "c1"),   # open-set
            "q4": Identity("Y", "c1"),   # open-set
        }

    def test_threshold_separates_open_set_when_scores_allow(self):
        top_ids = {"q1": "g1", "q2": "g2", "q3": "g1", "q4": "g2"}
        # Настоящие пары уверенно выше, open-set уверенно ниже.
        top_scores = {"q1": 0.92, "q2": 0.88, "q3": 0.31, "q4": 0.27}
        report = calibrate(top_ids, top_scores, self.query_labels, self.gallery_labels)
        self.assertEqual(report["selected"]["F1"], 1.0)
        self.assertEqual(report["selected"]["TNR"], 1.0)
        self.assertGreater(report["selected"]["threshold"], 0.31)
        self.assertLessEqual(report["selected"]["threshold"], 0.88)
        self.assertGreater(report["pr_auc"], 0.99)

    def test_always_answering_is_penalised(self):
        top_ids = {"q1": "g1", "q2": "g2", "q3": "g1", "q4": "g2"}
        top_scores = {"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.5}
        points = sweep_threshold(top_ids, top_scores, self.query_labels, self.gallery_labels)
        answering = min(points, key=lambda p: p.threshold)
        self.assertEqual(answering.tnr, 0.0)
        # Сплошной отказ даёт 0.3 * 1.0; ответ на всё — 0.7 * F1 при TNR = 0.
        refusing = max(points, key=lambda p: p.threshold)
        self.assertAlmostEqual(refusing.score, 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
