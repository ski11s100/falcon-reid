"""Докачка весов, если вместо них пришли указатели Git LFS.

    python scripts/fetch_weights.py
    python scripts/fetch_weights.py --base-url https://зеркало/веса/

Веса моделей (403 МБ) хранятся в Git LFS. Если репозиторий склонирован без
установленного Git LFS, скачан ZIP-архивом с GitHub или у репозитория
исчерпана месячная квота трафика LFS, в models/ оказываются не веса, а
текстовые указатели по 130 байт. Сервис это распознаёт и падает с понятной
ошибкой, а этот скрипт исправляет ситуацию: для каждого указателя скачивает
файл из релиза GitHub (или с зеркала) и сверяет его SHA-256 с хешем, записанным
в самом указателе. Файл с неверным хешем не сохраняется.

Только стандартная библиотека: скрипт должен работать до установки
зависимостей.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://github.com/ski11s100/falcon-reid/releases/download/weights-v1/"
POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def read_pointer(path: Path) -> dict | None:
    """Поля указателя Git LFS (oid, size) или None, если это настоящий файл."""
    with path.open("rb") as stream:
        head = stream.read(512)
    if not head.startswith(POINTER_PREFIX):
        return None
    fields = {}
    for line in head.decode("utf-8", errors="replace").splitlines():
        key, _, value = line.partition(" ")
        fields[key] = value.strip()
    oid = fields.get("oid", "")
    if not oid.startswith("sha256:"):
        raise ValueError(f"{path}: указатель без sha256")
    return {"sha256": oid.split(":", 1)[1], "size": int(fields.get("size", 0))}


def download(url: str, target: Path, expected_sha256: str, expected_size: int) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    received = 0
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        while chunk := response.read(1 << 20):
            stream.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if expected_size:
                print(f"\r  {target.name}: {received / 1e6:.0f} / {expected_size / 1e6:.0f} МБ",
                      end="", flush=True)
    print()
    if digest.hexdigest() != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{target.name}: SHA-256 не совпал — файл повреждён или подменён")
    partial.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Докачка весов вместо указателей Git LFS")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="Откуда брать веса: релиз GitHub или зеркало")
    args = parser.parse_args()

    pointers = {}
    for path in sorted(args.models.glob("*.pt")):
        pointer = read_pointer(path)
        if pointer is not None:
            pointers[path] = pointer
    if not pointers:
        print("Все веса на месте, скачивать нечего.")
        return

    base = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    for path, pointer in pointers.items():
        print(f"{path.name}: вместо весов указатель Git LFS, скачиваю")
        try:
            download(base + path.name, path, pointer["sha256"], pointer["size"])
        except Exception as error:  # noqa: BLE001 — сообщение важнее типа ошибки
            sys.exit(f"Не удалось скачать {path.name}: {error}")
        print(f"{path.name}: скачан, SHA-256 совпал")


if __name__ == "__main__":
    main()
