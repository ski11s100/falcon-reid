"""Журнал обращений: кто, когда и что делал в системе.

Сервис ищет конкретные автомобили по кадрам городских камер. В связке с
данными заказчика (где и когда стояла камера, кому принадлежит машина) это
обработка персональных данных, и оператор такой системы обязан уметь ответить
на вопрос «кто и зачем искал эту машину». Приказ ФСТЭК России №21 к 152-ФЗ
относит регистрацию событий безопасности к базовым мерам защиты. Журнал же —
главная защита от злоупотребления самим поиском: слежки за конкретным
человеком по его машине.

Что пишется: время, действие, адрес клиента, отпечаток ключа API (не сам
ключ), идентификаторы найденного и итог. Что НЕ пишется никогда: кадры,
миниатюры и эмбеддинги — журнал не должен становиться второй галереей.

Каждая запись — одна строка JSON: в журнал контейнера (docker logs) и, если
задан FALCON_AUDIT_LOG, в файл на томе с данными. Файл только дописывается.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, UTC
from pathlib import Path

from fastapi import Request

logger = logging.getLogger("falcon.audit")


def client_address(request: Request | None) -> str | None:
    """Адрес клиента. За nginx настоящий адрес приходит в X-Forwarded-For."""
    if request is None:
        return None
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def key_fingerprint(key: str | None) -> str:
    """Отпечаток ключа API: по нему видно, чей это был доступ, но ключ не утекает."""
    if not key:
        return "нет"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class AuditLog:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.healthy = True
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.exception("каталог журнала обращений недоступен: %s", self.path)
                self.healthy = False

    def record(self, action: str, request: Request | None = None, **fields) -> dict:
        entry = {"time": datetime.now(UTC).isoformat(timespec="seconds"),
                 "action": action}
        if request is not None:
            entry["client"] = client_address(request)
            entry["key"] = key_fingerprint(request.headers.get("x-api-key"))
        entry.update({name: value for name, value in fields.items() if value is not None})
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        logger.info(line)
        if self.path is not None:
            try:
                with self._lock, self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                self.healthy = True
            except OSError:
                # Запрос оператора из-за журнала не падает, но сбой виден: в
                # журнале контейнера и в /api/health (audit_log_ok = false).
                logger.exception("не удалось записать журнал обращений: %s", self.path)
                self.healthy = False
        return entry
