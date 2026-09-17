"""Диагностика не должна ронять обучение.

Пульс для индикатора прогресса однажды убил прогон на девятой эпохе: os.replace()
на Windows не может заменить файл, открытый другим процессом на чтение, а читал
его как раз индикатор. Инструмент наблюдения уронил наблюдаемое.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.train import write_heartbeat  # noqa: E402


class TestHeartbeatNeverRaises(unittest.TestCase):
    def test_writes_payload(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_heartbeat(folder, {"epoch": 3, "step": 40})
            saved = json.loads((folder / "heartbeat.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["epoch"], 3)
            self.assertEqual(saved["step"], 40)

    def test_permission_error_is_swallowed(self):
        """Ровно та ошибка, что убила прогон: файл занят читателем."""
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("pathlib.Path.replace",
                            side_effect=PermissionError("[WinError 5] Отказано в доступе")):
                write_heartbeat(Path(directory), {"epoch": 1})

    def test_missing_directory_is_swallowed(self):
        write_heartbeat(Path("D:/такого/каталога/нет"), {"epoch": 1})

    def test_disk_error_is_swallowed(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("pathlib.Path.write_text",
                            side_effect=OSError("[Errno 28] No space left on device")):
                write_heartbeat(Path(directory), {"epoch": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
