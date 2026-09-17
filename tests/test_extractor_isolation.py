"""Экстрактор не должен менять модель, которой не владеет.

Эти тесты закрывают конкретный класс ошибок, который дважды стоил часов отладки:
валидация во время обучения передаёт экстрактору ту самую модель, которая
обучается, и любая её мутация тихо портит обучение.

История:
  * конвертация в half — веса каждые две эпохи округлялись fp32 -> fp16 -> fp32;
  * .to(memory_format=channels_last) — после валидации веса оставались в
    channels_last, обучение подавало NCHW, cuDNN перекладывал память на каждой
    свёртке, эпоха дорожала с 55 до 400 секунд.

Оба раза симптом выглядел как проблема с памятью, а не как порча состояния.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.extract import ExtractorConfig, FeatureExtractor  # noqa: E402
from falcon.model import build_model  # noqa: E402


def make_model():
    return build_model(num_classes=8, embedding_dim=256, pretrained=False, verbose=False)


class TestBorrowedModelIsUntouched(unittest.TestCase):
    """Модель, переданная извне, обязана остаться ровно такой, какой была."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = make_model().to(self.device)

    def _snapshot(self):
        sample = self.model.backbone.layer1[0].conv2.weight
        return {
            "dtype": next(self.model.parameters()).dtype,
            "channels_last": sample.is_contiguous(memory_format=torch.channels_last),
            "device": next(self.model.parameters()).device.type,
        }

    def test_dtype_unchanged(self):
        before = self._snapshot()
        FeatureExtractor(model=self.model,
                         config=ExtractorConfig(device=self.device, half=True, num_workers=0))
        self.assertEqual(before["dtype"], self._snapshot()["dtype"],
                         "Экстрактор сконвертировал тип данных чужой модели")

    def test_memory_format_unchanged(self):
        before = self._snapshot()
        FeatureExtractor(model=self.model,
                         config=ExtractorConfig(device=self.device, half=True,
                                                channels_last=True, num_workers=0))
        self.assertEqual(before["channels_last"], self._snapshot()["channels_last"],
                         "Экстрактор переложил веса чужой модели в channels_last")

    def test_weights_bit_identical(self):
        reference = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        FeatureExtractor(model=self.model,
                         config=ExtractorConfig(device=self.device, half=True, num_workers=0))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(reference[key], value),
                            f"Экстрактор изменил веса чужой модели в {key}")

    def test_training_mode_restorable(self):
        # Экстрактор переводит модель в eval — это допустимо, но вызывающий
        # должен иметь возможность вернуть режим, и веса при этом не портятся.
        self.model.train()
        FeatureExtractor(model=self.model,
                         config=ExtractorConfig(device=self.device, half=True, num_workers=0))
        self.model.train()
        self.assertTrue(self.model.training)

    def test_borrowed_model_uses_autocast_not_precast(self):
        extractor = FeatureExtractor(
            model=self.model,
            config=ExtractorConfig(device=self.device, half=True, num_workers=0))
        if self.device == "cuda":
            self.assertTrue(extractor._autocast, "Заимствованная модель должна идти через autocast")
            self.assertFalse(extractor._pre_cast, "Заимствованную модель нельзя пре-кастовать")
            self.assertFalse(extractor._channels_last,
                             "Заимствованной модели нельзя менять раскладку памяти")


class TestOwnedModelIsOptimised(unittest.TestCase):
    """Собственную модель экстрактор оптимизирует полностью — это рабочий путь."""

    def test_checkpoint_model_is_precast(self):
        if not torch.cuda.is_available():
            self.skipTest("нужен CUDA")
        import tempfile

        from falcon.extract import save_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.pt"
            save_checkpoint(make_model(), path)
            extractor = FeatureExtractor(
                checkpoint=path,
                config=ExtractorConfig(device="cuda", half=True, channels_last=True, num_workers=0))
            self.assertTrue(extractor._pre_cast, "Своя модель должна конвертироваться в half")
            self.assertTrue(extractor._channels_last, "Своей модели channels_last разрешён")
            self.assertFalse(extractor._autocast, "При пре-касте autocast не нужен")
            self.assertEqual(next(extractor.model.parameters()).dtype, torch.float16)


class TestCameraCodeStability(unittest.TestCase):
    """Код камеры обязан совпадать между процессами.

    Батчи собираются в процессах-воркерах DataLoader. Встроенный hash() для
    строк рандомизируется на каждый процесс, поэтому одна и та же камера
    получала бы разные коды у разных воркеров, и маска кросс-камерных пар в
    triplet-потере превращалась бы в шум — молча, без единой ошибки.
    """

    def test_numeric_ids_pass_through(self):
        from falcon.extract import _camera_code
        self.assertEqual(_camera_code("90"), 90)
        self.assertEqual(_camera_code("0"), 0)

    def test_missing_id(self):
        from falcon.extract import _camera_code
        self.assertEqual(_camera_code(None), -1)

    def test_textual_ids_are_deterministic_across_processes(self):
        import subprocess
        code = (
            "import sys; sys.path.insert(0, r'%s');"
            "from falcon.extract import _camera_code;"
            "print(_camera_code('veri-c1'), _camera_code('veri-c17'))"
            % str(Path(__file__).resolve().parent.parent)
        )
        runs = {
            subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(3)
        }
        self.assertEqual(len(runs), 1,
                         f"Код камеры различается между процессами: {runs}")

    def test_different_cameras_get_different_codes(self):
        from falcon.extract import _camera_code
        codes = {_camera_code(f"veri-c{i}") for i in range(1, 21)}
        self.assertEqual(len(codes), 20, "Коды камер столкнулись")


if __name__ == "__main__":
    unittest.main(verbosity=2)
