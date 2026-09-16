import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from app.fingerprint import InvalidImage, classify_match, cosine_similarity, create_fingerprint, crop_bbox
from app.storage import VehicleStore
from app.quality import analyze_quality


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        self.base = Image.new("RGB", (320, 200), (220, 220, 215))
        pixels = np.asarray(self.base).copy()
        pixels[50:150, 60:260] = (40, 90, 160)
        self.base = Image.fromarray(pixels)

    def test_vector_is_normalized_and_stable_on_resize(self):
        left = create_fingerprint(self.base)
        right = create_fingerprint(self.base.resize((640, 400)))
        self.assertAlmostEqual(float(np.linalg.norm(left)), 1.0, places=5)
        self.assertGreater(cosine_similarity(left, right), 0.99)

    def test_brightness_change_is_closer_than_different_color(self):
        base = create_fingerprint(self.base)
        darker = create_fingerprint(ImageEnhance.Brightness(self.base).enhance(0.8))
        red = create_fingerprint(Image.new("RGB", (320, 200), (180, 35, 40)))
        self.assertGreater(cosine_similarity(base, darker), cosine_similarity(base, red))
        self.assertGreater(cosine_similarity(base, darker), 0.94)

    def test_decision_thresholds(self):
        self.assertIn("высокая", classify_match(0.97).verdict)
        self.assertIn("проверка", classify_match(0.90).verdict)
        self.assertIn("маловероятно", classify_match(0.4).verdict)

    def test_bbox_crop_uses_xywh_and_clips_to_frame(self):
        cropped = crop_bbox(self.base, {"x": 50, "y": 40, "w": 300, "h": 190})
        self.assertEqual(cropped.size, (270, 160))
        with self.assertRaises(InvalidImage):
            crop_bbox(self.base, [10, 10, 0, 20])


class StorageTests(unittest.TestCase):
    def test_upsert_and_search(self):
        with tempfile.TemporaryDirectory() as folder:
            store = VehicleStore(Path(folder) / "test.sqlite3")
            vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            store.upsert("car-a", vector, {"color": "white"})
            profile = store.upsert("car-a", vector, {"color": "white"})
            store.upsert("car-b", np.asarray([0.0, 1.0, 0.0]), {"color": "red"})
            matches = store.search(vector, top_k=2)
            self.assertEqual(matches[0]["vehicle_id"], "car-a")
            self.assertEqual(len(store.list()), 2)
            self.assertEqual(profile["sample_count"], 2)


class QualityTests(unittest.TestCase):
    def test_tiny_dark_image_reports_issues(self):
        result = analyze_quality(Image.new("RGB", (80, 60), (5, 5, 5)))
        self.assertEqual(result["status"], "poor")
        self.assertIn("низкое разрешение", result["issues"])
        self.assertIn("кадр слишком тёмный", result["issues"])

    def test_normal_textured_image_is_usable(self):
        values = np.indices((200, 320)).sum(axis=0) % 2
        image = Image.fromarray(np.uint8(values * 180 + 30)).convert("RGB")
        result = analyze_quality(image)
        self.assertGreaterEqual(result["score"], 75)


if __name__ == "__main__":
    unittest.main()
