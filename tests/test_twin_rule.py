"""Двойники: над порогом две разные машины почти вровень — «требуется проверка».

Без номера две машины одной модели и окраски почти неразличимы. Сервис не
должен уверенно называть первую из них: он отвечает «требуется проверка» и
показывает обе. Проверяется сама функция правила и ответ API целиком.
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

from falcon.calibrate import find_twin  # noqa: E402
from service.app import app, require_model, require_repository  # noqa: E402
from service.config import Settings, get_settings  # noqa: E402
from service.repository import GalleryItem, SQLiteRepository  # noqa: E402


class TestFindTwin(unittest.TestCase):
    def test_close_rival_over_threshold_is_a_twin(self):
        ranked = [("ТС-1", 0.91), ("ТС-2", 0.88), ("ТС-3", 0.40)]
        self.assertEqual(find_twin(ranked, 0.69, 0.05), ("ТС-2", 0.88))

    def test_clear_leader_is_not_a_twin(self):
        self.assertIsNone(find_twin([("ТС-1", 0.95), ("ТС-2", 0.80)], 0.69, 0.05))

    def test_rival_below_threshold_is_not_a_twin(self):
        """Под порогом соперник уже отвергнут — сомнения нет."""
        self.assertIsNone(find_twin([("ТС-1", 0.71), ("ТС-2", 0.68)], 0.69, 0.05))

    def test_leader_below_threshold_is_a_refusal_not_a_twin(self):
        self.assertIsNone(find_twin([("ТС-1", 0.60), ("ТС-2", 0.59)], 0.69, 0.05))

    def test_more_shots_of_the_same_car_are_not_rivals(self):
        ranked = [("ТС-1", 0.91), ("ТС-1", 0.90), ("ТС-2", 0.70)]
        self.assertIsNone(find_twin(ranked, 0.69, 0.05))

    def test_leader_is_scored_by_its_best_shot(self):
        """Порядок выдачи идёт по обогащённым векторам и может разойтись с
        числами: сравнивается лучший снимок первой машины, а не первый."""
        ranked = [("ТС-1", 0.80), ("ТС-2", 0.83), ("ТС-1", 0.95)]
        self.assertIsNone(find_twin(ranked, 0.69, 0.05))

    def test_rival_ahead_of_the_leader_is_a_twin(self):
        ranked = [("ТС-1", 0.80), ("ТС-2", 0.83)]
        self.assertEqual(find_twin(ranked, 0.69, 0.05), ("ТС-2", 0.83))

    def test_shots_without_vehicle_are_ignored(self):
        """Про снимок без машины нельзя сказать, что это другая машина."""
        self.assertIsNone(find_twin([("ТС-1", 0.90), (None, 0.89)], 0.69, 0.05))
        self.assertIsNone(find_twin([(None, 0.90), ("ТС-2", 0.89)], 0.69, 0.05))

    def test_empty_list(self):
        self.assertIsNone(find_twin([], 0.69, 0.05))


class FakeExtractor:
    """Вектор — средний цвет кропа: близкие цвета дают близкие векторы."""

    def encode_image(self, crop):
        mean = np.asarray(crop.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) + 1.0
        return mean / np.linalg.norm(mean)


def png(colour) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), colour).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class TestTwinVerdict(unittest.TestCase):
    """Красная машина и её двойник чуть другого оттенка, синяя — чужая."""

    def setUp(self):
        self.repository = SQLiteRepository(":memory:")
        self.repository.initialise(3)
        extractor = FakeExtractor()
        self.gallery = {"red": ("ТС-1", (200, 30, 30)), "red-twin": ("ТС-2", (196, 36, 30)),
                        "blue": ("ТС-3", (30, 30, 200))}
        self.repository.upsert([
            GalleryItem(image_id=image_id, vehicle_id=vehicle,
                        embedding=extractor.encode_image(Image.new("RGB", (32, 24), colour)), metadata={})
            for image_id, (vehicle, colour) in self.gallery.items()])
        self.settings = Settings(match_threshold=0.99, twin_margin=0.05)
        app.dependency_overrides[require_model] = lambda: extractor
        app.dependency_overrides[require_repository] = lambda: self.repository
        app.dependency_overrides[get_settings] = lambda: self.settings
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def search(self, **extra):
        response = self.client.post("/api/search", json={"image_base64": png((200, 30, 30)),
                                                         "top_k": 3, **extra})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_twin_goes_to_review_with_both_cars_named(self):
        data = self.search()
        self.assertFalse(data["accepted"])
        self.assertEqual(data["verdict"], "требуется проверка")
        self.assertEqual(data["matches"], [])
        self.assertIn("ТС-1", data["refusal_reason"])
        self.assertIn("ТС-2", data["refusal_reason"])
        # Кандидаты остаются: оператору нужны оба снимка для сравнения.
        self.assertEqual({c["vehicle_id"] for c in data["candidates"]}, {"ТС-1", "ТС-2", "ТС-3"})

    def test_without_the_twin_the_match_is_accepted(self):
        data = self.search(exclude_image_ids=["red-twin"])
        self.assertTrue(data["accepted"])
        self.assertEqual(data["verdict"], "совпадение")

    def test_rule_is_off_without_margin(self):
        self.settings = Settings(match_threshold=0.99, twin_margin=None)
        self.assertEqual(self.search()["verdict"], "совпадение")


if __name__ == "__main__":
    unittest.main()
