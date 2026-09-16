"""Dependency-light HTTP server for the Falcon MVP."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .fingerprint import (
    FINGERPRINT_VERSION,
    HIGH_THRESHOLD,
    REVIEW_THRESHOLD,
    InvalidImage,
    create_fingerprint,
    crop_bbox,
    decode_image,
)
from .quality import analyze_quality
from .storage import VehicleStore


ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"
DATA = Path(os.environ.get("FALCON_DATA_DIR", ROOT / "data"))
ENCODER = create_fingerprint
if os.environ.get("FALCON_CHECKPOINT"):
    from .neural import NeuralEncoder
    ENCODER = NeuralEncoder(os.environ["FALCON_CHECKPOINT"], os.environ.get("FALCON_DEVICE", "cpu"))
    FINGERPRINT_VERSION = ENCODER.version
STORE = VehicleStore(DATA / "falcon.sqlite3", version=FINGERPRINT_VERSION)
NEURAL_THRESHOLD_READY = not os.environ.get("FALCON_CHECKPOINT") or bool(os.environ.get("FALCON_REVIEW_THRESHOLD"))
MAX_BODY = 14 * 1024 * 1024

OPENAPI = {
    "openapi": "3.0.3",
    "info": {"title": "ФАЛЬКОН Vehicle ReID API", "version": "0.2.0"},
    "paths": {
        "/api/health": {"get": {"summary": "Проверка состояния", "responses": {"200": {"description": "OK"}}}},
        "/api/vehicles/register": {"post": {"summary": "Добавить наблюдение автомобиля", "responses": {"201": {"description": "Создано"}, "400": {"description": "Ошибка входных данных"}}}},
        "/api/search": {"post": {"summary": "Найти похожие автомобили или вернуть отказ", "responses": {"200": {"description": "Результат поиска"}, "400": {"description": "Ошибка входных данных"}}}},
        "/api/vehicles/{vehicle_id}": {"delete": {"summary": "Удалить профиль", "parameters": [{"name": "vehicle_id", "in": "path", "required": True, "schema": {"type": "string"}}], "responses": {"200": {"description": "Удалено"}, "404": {"description": "Не найдено"}}}},
    },
}

IMAGE_SCHEMA = {"type": "object", "required": ["image_base64"], "properties": {
    "image_base64": {"type": "string", "description": "Base64 encoded JPEG/PNG"},
    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
             "description": "x, y, width, height in encoded frame pixels"},
    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
    "vehicle_id": {"type": "string", "maxLength": 80},
    "metadata": {"type": "object"}}}
for route in ("/api/search", "/api/vehicles/register"):
    OPENAPI["paths"][route]["post"]["requestBody"] = {
        "required": True, "content": {"application/json": {"schema": IMAGE_SCHEMA}}}
OPENAPI["paths"]["/api/vehicles"] = {"get": {"summary": "List profiles", "responses": {"200": {"description": "Profiles"}}}}


class Handler(BaseHTTPRequestHandler):
    server_version = "FalconMVP/0.1"

    def _json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Некорректный Content-Length") from exc
        if length <= 0 or length > MAX_BODY:
            raise ValueError("Пустой или слишком большой запрос")
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Некорректный JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Ожидается JSON-объект")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(
                200,
                {"status": "ok", "profiles": len(STORE.list()), "fingerprint_version": FINGERPRINT_VERSION},
            )
        if path == "/api/vehicles":
            return self._json(200, STORE.list())
        if path == "/api/config":
            return self._json(
                200,
                {
                    "max_image_mb": 10,
                    "fingerprint_version": FINGERPRINT_VERSION,
                    "thresholds": {"high": HIGH_THRESHOLD, "review": REVIEW_THRESHOLD},
                },
            )
        if path == "/openapi.json":
            return self._json(200, OPENAPI)
        if path in ("/", "/index.html"):
            body = (STATIC / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        self._json(404, {"error": "Маршрут не найден"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            image = decode_image(payload.get("image_base64", ""))
            crop = crop_bbox(image, payload.get("bbox"))
            quality = analyze_quality(crop)
            fingerprint = ENCODER(crop)
            path = urlparse(self.path).path
            if path == "/api/vehicles/register":
                vehicle_id = str(payload.get("vehicle_id", "")).strip()
                if not vehicle_id or len(vehicle_id) > 80:
                    raise ValueError("Укажите vehicle_id длиной до 80 символов")
                metadata = payload.get("metadata") or {}
                if not isinstance(metadata, dict):
                    raise ValueError("metadata должен быть объектом")
                profile = STORE.upsert(vehicle_id, fingerprint, metadata)
                return self._json(
                    201,
                    {
                        "profile": profile,
                        "fingerprint_size": int(fingerprint.size),
                        "fingerprint_version": FINGERPRINT_VERSION,
                        "quality": quality,
                        "crop_size": {"width": crop.width, "height": crop.height},
                    },
                )
            if path == "/api/search":
                top_k = max(1, min(int(payload.get("top_k", 10)), 20))
                matches = STORE.search(fingerprint, top_k)
                accepted = bool(NEURAL_THRESHOLD_READY and matches and matches[0]["score"] >= REVIEW_THRESHOLD)
                return self._json(200, {
                    "accepted": accepted,
                    "matches": matches if accepted else [],
                    "candidates": matches,
                    "refusal_reason": None if accepted else ("Порог новой модели ещё не откалиброван" if not NEURAL_THRESHOLD_READY else "Надёжное совпадение не найдено"),
                    "quality": quality,
                    "crop_size": {"width": crop.width, "height": crop.height},
                })
            self._json(404, {"error": "Маршрут не найден"})
        except (ValueError, InvalidImage) as exc:
            self._json(400, {"error": str(exc)})
        except Exception:
            self._json(500, {"error": "Внутренняя ошибка сервиса"})

    def do_DELETE(self) -> None:  # noqa: N802
        prefix = "/api/vehicles/"
        path = urlparse(self.path).path
        if not path.startswith(prefix):
            return self._json(404, {"error": "Маршрут не найден"})
        vehicle_id = unquote(path[len(prefix):])
        if STORE.delete(vehicle_id):
            return self._json(200, {"deleted": vehicle_id})
        self._json(404, {"error": "Профиль не найден"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[falcon] {self.address_string()} {format % args}")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"ФАЛЬКОН запущен: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
