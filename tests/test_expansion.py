"""Обогащение векторов соседями: правила протокола и поведение.

Ответ 38 запрещает связи «запрос — запрос»: выдача одного запроса не должна
зависеть от того, какие ещё запросы попали в пачку. alpha-QE обогащает запрос
только векторами галереи, DBA — галерею её же соседями, поэтому правило
соблюдается; тест это закрепляет.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.submit import SubmissionConfig, build_ranking, enrich_vectors  # noqa: E402


def sample(seed=0, queries=12, gallery=48, dim=8):
    rng = np.random.default_rng(seed)
    # Галерея из групп похожих векторов: у «машины» несколько снимков.
    centres = rng.standard_normal((gallery // 4, dim))
    g = np.repeat(centres, 4, axis=0) + 0.25 * rng.standard_normal((gallery, dim))
    q = centres[np.arange(queries) % len(centres)] + 0.3 * rng.standard_normal((queries, dim))
    normalise = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    return normalise(q).astype(np.float32), normalise(g).astype(np.float32)


class TestEnrichment(unittest.TestCase):
    def test_result_of_a_query_does_not_depend_on_other_queries(self):
        q, g = sample()
        config = SubmissionConfig()
        full, full_scores = build_ranking(q, g, config, progress=False)
        for i in (0, 5, 11):
            single, single_scores = build_ranking(q[i : i + 1], g, config, progress=False)
            self.assertEqual(list(single[0]), list(full[i]))
            self.assertTrue(np.allclose(single_scores[0], full_scores[i], atol=1e-6))

    def test_gallery_enrichment_does_not_depend_on_queries(self):
        q, g = sample()
        config = SubmissionConfig()
        _, gallery_a = enrich_vectors(q, g, config)
        _, gallery_b = enrich_vectors(q[:3], g, config)
        self.assertTrue(np.allclose(gallery_a, gallery_b, atol=1e-6))

    def test_vectors_stay_unit_length(self):
        q, g = sample()
        query, gallery = enrich_vectors(q, g, SubmissionConfig())
        for vectors in (query, gallery):
            self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5))

    def test_high_neighbour_threshold_keeps_vectors_unchanged(self):
        # Соседей выше порога нет — обогащать нечем, векторы прежние.
        q, g = sample()
        config = SubmissionConfig(neighbour_min=1.1)
        query, gallery = enrich_vectors(q, g, config)
        self.assertTrue(np.allclose(query, q, atol=1e-6))
        self.assertTrue(np.allclose(gallery, g, atol=1e-6))

    def test_distant_neighbour_pulls_less_than_a_close_one(self):
        """Вес соседа — сходство в кубе: дальний сосед почти не влияет.

        Проверяем на двух галерейных снимках: близкий сосед и далёкий. При
        линейном весе далёкий тянет вектор заметно сильнее, чем при кубе.
        """
        base = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        close = np.array([0.95, 0.312, 0.0], dtype=np.float32)
        far = np.array([0.55, 0.835, 0.0], dtype=np.float32)
        gallery = np.vstack([base[0], close, far]).astype(np.float32)
        shifts = {}
        for power in (1.0, 3.0):
            config = SubmissionConfig(dba_k=2, qe_k=0, neighbour_min=0.4, expansion_power=power)
            _, enriched = enrich_vectors(base, gallery, config)
            shifts[power] = float(enriched[0] @ far)
        self.assertLess(shifts[3.0], shifts[1.0])

    def test_blocks_give_the_same_result_as_one_matrix(self):
        """Счёт блоками — только про память: результат обязан совпасть до бита."""
        import falcon.submit as submit
        q, g = sample()
        saved = submit.MERGE_BLOCK
        try:
            submit.MERGE_BLOCK = 10 ** 6
            whole = enrich_vectors(q, g, SubmissionConfig())
            submit.MERGE_BLOCK = 7            # меньше галереи и не делит её нацело
            blocks = enrich_vectors(q, g, SubmissionConfig())
        finally:
            submit.MERGE_BLOCK = saved
        for a, b in zip(whole, blocks):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    def test_switched_off_by_zero_neighbours(self):
        q, g = sample()
        query, gallery = enrich_vectors(q, g, SubmissionConfig(dba_k=0, qe_k=0))
        self.assertTrue(np.allclose(query, q, atol=1e-6))
        self.assertTrue(np.allclose(gallery, g, atol=1e-6))

    def test_confidence_is_the_plain_cosine(self):
        # Порог откалиброван по исходному косинусу, поэтому и в выдаче он же.
        q, g = sample()
        indices, scores = build_ranking(q, g, SubmissionConfig(), progress=False)
        for i in range(len(q)):
            self.assertTrue(np.allclose(scores[i], g[indices[i]] @ q[i], atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
