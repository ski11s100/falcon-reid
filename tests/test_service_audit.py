"""Журнал обращений и срок хранения галереи.

Для системы, которая ищет конкретные машины по кадрам камер, важно не только
найти, но и уметь ответить «кто и что искал» — и при этом не превратить журнал
во вторую копию галереи. А снимки не должны лежать дольше срока, который
задаёт регламент заказчика (152-ФЗ: не дольше, чем требует цель обработки).
"""

import base64
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from service.app import app, purge_expired, require_audit, require_model, require_repository  # noqa: E402
from service.audit import AuditLog, key_fingerprint  # noqa: E402
from service.repository import GalleryItem, SQLiteRepository  # noqa: E402


class FakeExtractor:
    def encode_image(self, crop):
        mean = np.asarray(crop.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) + 1.0
        return mean / np.linalg.norm(mean)


def png(colour) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), colour).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class TestAuditLog(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "audit.jsonl"
        self.audit = AuditLog(self.path)
        self.repository = SQLiteRepository(":memory:")
        self.repository.initialise(3)
        extractor = FakeExtractor()
        self.repository.upsert([GalleryItem(
            image_id="red", vehicle_id="ТС-1",
            embedding=extractor.encode_image(Image.new("RGB", (32, 24), (200, 30, 30))), metadata={})])
        app.dependency_overrides[require_model] = lambda: extractor
        app.dependency_overrides[require_repository] = lambda: self.repository
        app.dependency_overrides[require_audit] = lambda: self.audit
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.directory.cleanup()

    def entries(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_search_is_recorded_with_its_outcome(self):
        response = self.client.post("/api/search", json={"image_base64": png((200, 30, 30)), "top_k": 1},
                                    headers={"X-Forwarded-For": "10.1.2.3, 172.18.0.2"})
        self.assertEqual(response.status_code, 200, response.text)
        entry = self.entries()[-1]
        self.assertEqual(entry["action"], "search")
        self.assertEqual(entry["top_vehicle_id"], "ТС-1")
        self.assertEqual(entry["client"], "10.1.2.3")   # первый адрес — настоящий клиент
        self.assertIn("time", entry)

    def test_log_never_contains_images_or_vectors(self):
        """Журнал не должен становиться второй галереей: ни кадров, ни векторов."""
        image = png((200, 30, 30))
        self.client.post("/api/search", json={"image_base64": image, "top_k": 1})
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn(image[:40], text)
        for field in ("image_base64", "embedding", "thumbnail", "fingerprint"):
            self.assertNotIn(field, text)

    def test_api_key_is_fingerprinted_not_stored(self):
        self.client.post("/api/search", json={"image_base64": png((200, 30, 30)), "top_k": 1},
                         headers={"X-API-Key": "very-secret-key-42"})
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("very-secret-key-42", text)
        self.assertEqual(self.entries()[-1]["key"], key_fingerprint("very-secret-key-42"))

    def test_registration_and_deletion_are_recorded(self):
        self.client.post("/api/gallery/register",
                         json={"image_id": "blue", "vehicle_id": "ТС-2", "image_base64": png((30, 30, 200))})
        self.client.delete("/api/gallery/blue")
        actions = [e["action"] for e in self.entries()]
        self.assertEqual(actions[-2:], ["register", "delete"])

    def test_unwritable_log_does_not_break_search(self):
        """Сбой записи журнала виден в /api/health, но запрос оператора не падает."""
        broken = AuditLog(Path(self.directory.name))           # каталог вместо файла
        app.dependency_overrides[require_audit] = lambda: broken
        response = self.client.post("/api/search", json={"image_base64": png((200, 30, 30)), "top_k": 1})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(broken.healthy)


class TestRetention(unittest.TestCase):
    def setUp(self):
        self.repository = SQLiteRepository(":memory:")
        self.repository.initialise(3)
        vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.repository.upsert([GalleryItem(image_id=name, vehicle_id="ТС", embedding=vector, metadata={})
                                for name in ("old", "fresh")])
        # «old» добавлен сорок дней назад.
        with self.repository._lock:
            self.repository.connection.execute(
                "UPDATE gallery SET created_at = ? WHERE image_id = 'old'", (time.time() - 40 * 86400,))
            self.repository.connection.commit()

    def test_only_expired_shots_are_removed_and_logged(self):
        audit = AuditLog(None)
        removed = purge_expired(self.repository, 30, audit)
        self.assertEqual(removed, 1)
        self.assertIsNone(self.repository.embedding_of("old"))
        self.assertIsNotNone(self.repository.embedding_of("fresh"))

    def test_nothing_to_remove_is_not_an_error(self):
        self.assertEqual(purge_expired(self.repository, 100, AuditLog(None)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
