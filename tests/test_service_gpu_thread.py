"""Сервис: работа с видеокартой в одном потоке и отпечатки для показа.

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


class TestFingerprint(unittest.TestCase):
    """Отпечаток для показа: детерминирован и отражает сходство эмбеддингов."""

    def setUp(self):
        import numpy as np
        from service.app import FINGERPRINT_GROUPS, fingerprint
        self.np, self.groups, self.fingerprint = np, FINGERPRINT_GROUPS, fingerprint

    def test_shape_and_determinism(self):
        vector = self.np.random.default_rng(1).standard_normal(4096).astype("float32")
        first, second = self.fingerprint(vector), self.fingerprint(vector)
        self.assertEqual(len(first), self.groups)
        self.assertEqual(first, second)

    def test_similar_vectors_give_similar_rings(self):
        np = self.np
        rng = np.random.default_rng(2)
        base = rng.standard_normal(4096).astype("float32")
        near = base + 0.3 * rng.standard_normal(4096).astype("float32")
        far = rng.standard_normal(4096).astype("float32")
        f = [np.array(self.fingerprint(v)) for v in (base, near, far)]
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
        self.assertGreater(corr(f[0], f[1]), 0.8)
        self.assertLess(abs(corr(f[0], f[2])), 0.4)

    def test_missing_embedding(self):
        self.assertIsNone(self.fingerprint(None))


class TestExplainParts(unittest.TestCase):
    """Grad-CAM ансамбля: каждая модель получает свою часть вектора."""

    def test_ensemble_reference_is_split_by_member(self):
        import numpy as np
        from types import SimpleNamespace
        from falcon.model import VehicleReID
        from service.app import explain_parts
        # Настоящие модели не нужны: explain_parts смотрит только на их тип.
        resnet = VehicleReID.__new__(VehicleReID)
        members = [SimpleNamespace(model=resnet, explain_model="A", feature_dim=3),
                   SimpleNamespace(model=resnet, explain_model="B", feature_dim=2)]
        reference = np.arange(5, dtype=np.float32)
        parts = explain_parts(SimpleNamespace(members=members), reference)
        self.assertEqual([m for m, _ in parts], ["A", "B"])
        self.assertEqual(parts[0][1].tolist(), [0, 1, 2])
        self.assertEqual(parts[1][1].tolist(), [3, 4])

    def test_mixed_ensemble_includes_vit_member(self):
        import numpy as np
        from types import SimpleNamespace
        from falcon.model import VehicleReID, ViTReID
        from service.app import explain_parts
        # Grad-CAM строится и по ViT (сетка патчей): в карту входят все участники.
        members = [SimpleNamespace(model=ViTReID.__new__(ViTReID), explain_model="V", feature_dim=2),
                   SimpleNamespace(model=VehicleReID.__new__(VehicleReID), explain_model="R",
                                   feature_dim=3)]
        reference = np.arange(5, dtype=np.float32)
        parts = explain_parts(SimpleNamespace(members=members), reference)
        self.assertEqual([m for m, _ in parts], ["V", "R"])
        self.assertEqual(parts[0][1].tolist(), [0, 1])
        self.assertEqual(parts[1][1].tolist(), [2, 3, 4])

    def test_single_model_gets_whole_vector(self):
        import numpy as np
        from types import SimpleNamespace
        from service.app import explain_parts
        reference = np.ones(4, dtype=np.float32)
        parts = explain_parts(SimpleNamespace(explain_model="M"), reference)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][1].shape, (4,))


class TestModelSummary(unittest.TestCase):
    """Строка состава ансамбля в шапке интерфейса."""

    def test_repeated_architectures_are_grouped(self):
        from types import SimpleNamespace
        from falcon.model import VehicleReID, ViTReID
        from service.app import summarise_models
        vit, resnet = ViTReID.__new__(ViTReID), VehicleReID.__new__(VehicleReID)
        ensemble = SimpleNamespace(members=[SimpleNamespace(model=vit), SimpleNamespace(model=vit),
                                            SimpleNamespace(model=resnet)])
        self.assertEqual(summarise_models(ensemble), "2 × CLIP ViT-B/16 + ResNet50-IBN")
        self.assertEqual(summarise_models(SimpleNamespace(model=resnet)), "ResNet50-IBN")
