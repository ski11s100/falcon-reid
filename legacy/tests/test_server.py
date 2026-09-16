import base64
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

import app.server as server_module
from app.storage import VehicleStore


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        server_module.STORE = VehicleStore(Path(cls.temp.name) / "api.sqlite3")
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)
        cls.temp.cleanup()

    @staticmethod
    def image_base64(color=(190, 40, 50)):
        buffer = io.BytesIO()
        Image.new("RGB", (320, 200), color).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.load(response)

    def test_health_register_and_search(self):
        status, health = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")

        image = self.image_base64()
        status, registered = self.request(
            "/api/vehicles/register",
            {"vehicle_id": "api-car", "image_base64": image, "metadata": {"color": "red"}},
        )
        self.assertEqual(status, 201)
        self.assertEqual(registered["profile"]["vehicle_id"], "api-car")
        self.assertIn("quality", registered)

        status, schema = self.request("/openapi.json")
        self.assertEqual(status, 200)
        self.assertEqual(schema["openapi"], "3.0.3")

        status, result = self.request("/api/search", {"image_base64": image, "bbox": [0, 0, 320, 200], "top_k": 3})
        self.assertEqual(status, 200)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["matches"][0]["vehicle_id"], "api-car")
        self.assertGreater(result["matches"][0]["score"], 0.99)

    def test_invalid_image_returns_400(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/search", {"image_base64": "not-base64"})
        self.assertEqual(caught.exception.code, 400)
        body = json.loads(caught.exception.read())
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
