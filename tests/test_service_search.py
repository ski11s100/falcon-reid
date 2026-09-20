"""Поиск через API: снимок-запрос из галереи не находит сам себя.

Интерфейс ищет машину «на других камерах» по её снимку из галереи и передаёт
этот снимок в exclude_image_ids. Без исключения первым кандидатом всегда был
бы он сам со сходством 1.0, и пример в интерфейсе ничего бы не показывал.
Модель подменяется детерминированной заглушкой, хранилище — SQLite в памяти.
"""

import base64
import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from service.app import app, require_model, require_repository  # noqa: E402
from service.repository import GalleryItem, SQLiteRepository  # noqa: E402


class FakeExtractor:
    """Вектор — средний цвет кропа: одинаковые картинки дают одинаковый вектор."""

    def encode_image(self, crop):
        mean = np.asarray(crop.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) + 1.0
        return mean / np.linalg.norm(mean)


def png(colour) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), colour).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class TestSearchExclude(unittest.TestCase):
    def setUp(self):
        self.repository = SQLiteRepository(":memory:")
        self.repository.initialise(3)
        extractor = FakeExtractor()
        items = []
        for image_id, vehicle, colour in [("red-a", "ТС-1", (200, 30, 30)), ("red-b", "ТС-1", (190, 40, 30)),
                                          ("blue", "ТС-2", (30, 30, 200))]:
            crop = Image.new("RGB", (32, 24), colour)
            items.append(GalleryItem(image_id=image_id, vehicle_id=vehicle,
                                     embedding=extractor.encode_image(crop), metadata={}))
        self.repository.upsert(items)
        app.dependency_overrides[require_model] = lambda: extractor
        app.dependency_overrides[require_repository] = lambda: self.repository
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def search(self, **extra):
        response = self.client.post("/api/search", json={"image_base64": png((200, 30, 30)),
                                                         "top_k": 2, **extra})
        self.assertEqual(response.status_code, 200, response.text)
        return [c["image_id"] for c in response.json()["candidates"]]

    def test_without_exclusion_the_same_shot_comes_first(self):
        self.assertEqual(self.search()[0], "red-a")

    def test_excluded_shot_is_skipped_and_top_k_is_kept(self):
        found = self.search(exclude_image_ids=["red-a"])
        self.assertNotIn("red-a", found)
        self.assertEqual(found, ["red-b", "blue"])

    def test_confidence_stays_the_plain_cosine(self):
        """Обогащение меняет порядок, но не шкалу: порог откалиброван по косинусу.

        Проверяем, что число рядом со снимком — именно косинус исходных
        векторов, а не пересчитанное сходство обогащённого запроса.
        """
        extractor = FakeExtractor()
        query = extractor.encode_image(Image.new("RGB", (32, 24), (200, 30, 30)))
        response = self.client.post("/api/search", json={"image_base64": png((200, 30, 30)),
                                                         "top_k": 3})
        candidates = response.json()["candidates"]
        for candidate in candidates:
            reference = self.repository.embedding_of(candidate["image_id"])
            expected = float(np.asarray(query, dtype=np.float32) @ reference)
            self.assertAlmostEqual(candidate["score"], expected, places=5)

    def test_batch_embeddings_match_single_reads(self):
        ids = ["red-a", "red-b", "blue"]
        batch = self.repository.embeddings_of(ids)
        self.assertEqual(set(batch), set(ids))
        for image_id in ids:
            np.testing.assert_allclose(batch[image_id], self.repository.embedding_of(image_id))
        self.assertEqual(self.repository.embeddings_of([]), {})

    def test_gallery_summary_names_the_thumbnail_shot(self):
        vehicles = self.repository.list_vehicles(10)
        self.assertEqual({v["vehicle_id"] for v in vehicles}, {"ТС-1", "ТС-2"})
        self.assertTrue(all("image_id" in v for v in vehicles))


if __name__ == "__main__":
    unittest.main(verbosity=2)
