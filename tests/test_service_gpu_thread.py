"""Вся работа сервиса с видеокартой идёт в одном потоке.

Кэш автоподбора алгоритмов cuDNN у PyTorch свой в каждом потоке. Когда поиск
попадал в свежий поток пула FastAPI, он заново подбирал алгоритмы для всех
свёрток ансамбля: около 6 секунд вместо 25 мс. Так выглядел первый поиск
сразу после загрузки страницы.
"""

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.app import on_gpu  # noqa: E402


class TestSingleGpuThread(unittest.TestCase):
    def test_calls_from_different_threads_run_in_one(self):
        seen: list[str] = []

        def request():
            seen.append(on_gpu(lambda: threading.current_thread().name))

        workers = [threading.Thread(target=request) for _ in range(6)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(len(seen), 6)
        self.assertEqual(len(set(seen)), 1, seen)

    def test_exceptions_reach_the_caller(self):
        with self.assertRaises(ZeroDivisionError):
            on_gpu(lambda: 1 / 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
