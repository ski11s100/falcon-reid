"""Защита HTTP-слоя сервиса.

Сервис принимает изображения из сети и хранит галерею, поэтому здесь три
рубежа, не зависящих от того, стоит ли перед ним обратный прокси:

  * ограничение размера тела запроса — до разбора JSON. Без него запрос на
    гигабайт целиком читается в память, и один клиент кладёт сервис;
  * заголовки безопасности и политика CSP для интерфейса оператора: скрипты
    только со своего источника, запрет встраивания во фрейм, запрет угадывания
    типа содержимого;
  * необязательный ключ API (FALCON_API_KEY). По умолчанию выключен, чтобы жюри
    запускало решение одной командой без настройки; в эксплуатации включается
    одной переменной среды.

Middleware написаны на чистом ASGI, без BaseHTTPMiddleware: тот буферизует
тело и ломает потоковую проверку размера.
"""

from __future__ import annotations

import json
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from .config import get_settings

# Политика для интерфейса оператора. Стили разрешены inline: интерфейс задаёт
# ширину шкал атрибутом style. Скрипты — строго со своего источника, inline-
# скриптов в интерфейсе нет. Картинки data: — миниатюры кандидатов и кадр запроса.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)

COMMON_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    (b"cross-origin-opener-policy", b"same-origin"),
]


class SecurityHeadersMiddleware:
    """Заголовки безопасности на каждый ответ, CSP — на всё, кроме Swagger UI."""

    def __init__(self, app, docs_prefixes: tuple[str, ...] = ("/docs",)):
        self.app = app
        self.docs_prefixes = docs_prefixes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        with_csp = not scope["path"].startswith(self.docs_prefixes)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                for name, value in COMMON_HEADERS:
                    if name not in present:
                        headers.append((name, value))
                if with_csp and b"content-security-policy" not in present:
                    headers.append((b"content-security-policy", CONTENT_SECURITY_POLICY.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodySizeLimitMiddleware:
    """Отказывает запросам с телом больше лимита, не читая их целиком.

    Проверяется и заявленный Content-Length, и фактически полученные байты:
    при передаче по частям (chunked) заголовка длины нет вовсе.
    """

    def __init__(self, app, limit_for_path):
        self.app = app
        self.limit_for_path = limit_for_path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return

        limit = self.limit_for_path(scope["path"])
        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await _reject(send, limit)
            return

        # Превышение при передаче по частям: отказ 413 отправляется сразу, а
        # приложению сообщается «клиент отключился». Исключение здесь не годится:
        # FastAPI перехватывает любые ошибки чтения тела и отвечает 400.
        received = 0
        responded = False

        async def limited_receive():
            nonlocal received, responded
            if responded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    await _reject(send, limit)
                    responded = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            # Ответ уже отправлен — всё, что приложение попытается сказать
            # после отключения, отбрасывается.
            if not responded:
                await send(message)

        await self.app(scope, limited_receive, guarded_send)


async def _reject(send, limit: int) -> None:
    body = json.dumps({"detail": f"Тело запроса больше {limit // (1024 * 1024)} МБ"},
                      ensure_ascii=False).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json; charset=utf-8"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def request_body_limit(path: str) -> int:
    """Лимит тела по маршруту. Base64 раздувает картинку на треть."""
    settings = get_settings()
    megabyte = 1024 * 1024
    if path.endswith("/register-batch"):
        return settings.max_batch_mb * megabyte
    return int(settings.max_upload_mb * megabyte * 1.4) + megabyte


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="Ключ API. Нужен, только если задан FALCON_API_KEY")


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    """Проверка ключа API. Сравнение за постоянное время — против подбора по таймингу."""
    expected = get_settings().api_key
    if not expected:
        return
    if not key or not secrets.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Нужен действительный ключ API (заголовок X-API-Key)",
                            headers={"WWW-Authenticate": "ApiKey"})
