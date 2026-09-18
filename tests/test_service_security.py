"""Защита HTTP-слоя: лимиты, заголовки, ключ API, опасные картинки.

Тесты не поднимают модель: TestClient без контекстного менеджера не запускает
lifespan. Проверяемые рубежи срабатывают раньше, чем нужна модель, — именно
так и должно быть: отказ обязан стоить дешевле обработки.
"""

import base64
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from service.app import app, decode_image  # noqa: E402
from service.config import get_settings  # noqa: E402


def png_bytes(size=(8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestBodyLimit(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_oversized_body_is_rejected_before_parsing(self):
        huge = "A" * (25 * 1024 * 1024)  # больше лимита одиночного запроса
        response = self.client.post("/api/search", content=huge,
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413)

    def test_chunked_body_without_length_is_counted(self):
        def chunks():
            for _ in range(30):
                yield b"A" * (1024 * 1024)
        response = self.client.post("/api/search", content=chunks(),
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413)


class TestSecurityHeaders(unittest.TestCase):
    def test_headers_on_api_and_ui(self):
        client = TestClient(app)
        for path in ("/api/health", "/"):
            response = client.get(path)
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")
            self.assertEqual(response.headers["x-frame-options"], "DENY")
            self.assertIn("script-src 'self'", response.headers["content-security-policy"])

    def test_swagger_is_not_broken_by_csp(self):
        response = TestClient(app).get("/docs")
        self.assertNotIn("content-security-policy", response.headers)


class TestApiKey(unittest.TestCase):
    def setUp(self):
        self.settings = get_settings()
        self.saved = self.settings.api_key
        self.settings.api_key = "test-key-7f3a"
        self.client = TestClient(app)

    def tearDown(self):
        self.settings.api_key = self.saved

    def test_protected_without_key(self):
        response = self.client.get("/api/gallery")
        self.assertEqual(response.status_code, 401)

    def test_wrong_key(self):
        response = self.client.get("/api/gallery", headers={"X-API-Key": "wrong-key"})
        self.assertEqual(response.status_code, 401)

    def test_health_stays_open(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_right_key_passes_the_gate(self):
        response = self.client.get("/api/gallery", headers={"X-API-Key": "test-key-7f3a"})
        # Дальше ключа запрос проходит; без поднятого хранилища ответ 503.
        self.assertNotEqual(response.status_code, 401)


class TestDangerousImages(unittest.TestCase):
    def test_decompression_bomb(self):
        saved = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = 1000  # делаем обычную картинку «бомбой»
        try:
            payload = base64.b64encode(png_bytes((200, 200))).decode()
            with self.assertRaises(HTTPException) as caught:
                decode_image(payload, 10 * 1024 * 1024)
            self.assertEqual(caught.exception.status_code, 413)
        finally:
            Image.MAX_IMAGE_PIXELS = saved

    def test_unsupported_format(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="GIF")
        with self.assertRaises(HTTPException) as caught:
            decode_image(base64.b64encode(buffer.getvalue()).decode(), 10 * 1024 * 1024)
        self.assertEqual(caught.exception.status_code, 415)

    def test_valid_png(self):
        image = decode_image(base64.b64encode(png_bytes()).decode(), 10 * 1024 * 1024)
        self.assertEqual(image.size, (8, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
