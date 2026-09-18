"""Многочасовое обучение должно переживать сбои вокруг себя.

Пульс для индикатора прогресса однажды убил прогон на девятой эпохе: os.replace()
на Windows не может заменить файл, открытый другим процессом на чтение, а читал
его как раз индикатор. Инструмент наблюдения уронил наблюдаемое.

Другой прогон встал на 20-й эпохе: валидация запускала процессы-воркеры, пока
ноутбук был в режиме ожидания, и родитель навсегда застрял в передаче данных
потомку. Отсюда потоковая валидация и сторож зависаний.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.extract import ExtractorConfig, ThreadedBatchLoader, build_loader  # noqa: E402
from falcon.train import save_state, write_heartbeat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import overnight  # noqa: E402


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


class TestValidationSpawnsNoProcesses(unittest.TestCase):
    def test_threads_flag_selects_thread_loader(self):
        loader = build_loader(list(range(10)), batch_size=4, num_workers=2,
                              pin_memory=False, threads=True)
        self.assertIsInstance(loader, ThreadedBatchLoader)

    def test_validation_config_uses_threads(self):
        """validate() обязана просить потоки: иначе посреди обучения снова spawn."""
        import inspect

        from falcon import train
        self.assertIn("threads=True", inspect.getsource(train.validate))

    def test_default_is_unchanged(self):
        self.assertFalse(ExtractorConfig().threads)


class TestStateSaveNeverRaises(unittest.TestCase):
    def test_disk_error_is_swallowed(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("torch.save", side_effect=OSError("[Errno 28] No space left")):
                save_state(Path(directory) / "state.pt", {"epoch": 1})


class TestStallWatchdog(unittest.TestCase):
    def test_silent_process_is_killed(self):
        import subprocess
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        with mock.patch.object(overnight, "STALL_MINUTES", 0.01):
            code = overnight.watch(process, Path("D:/нет/heartbeat.json"))
        self.assertIsNone(code)
        self.assertIsNotNone(process.poll())

    def test_exit_code_is_passed_through(self):
        import subprocess
        process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
        self.assertEqual(overnight.watch(process, Path("D:/нет/heartbeat.json")), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
