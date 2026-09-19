"""Grad-CAM для CLIP ViT: карта строится по сетке патчей и не вырождается.

Вектор ViT берётся из CLS-токена, поэтому по патчам на выходе последнего блока
градиент нулевой, и карта с этого слоя была бы пустой. Тест ловит такую
регрессию: карта должна быть ненулевой и иметь размер входа.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.explain import GradCAM  # noqa: E402
from falcon.model import ViTReID  # noqa: E402


class TestViTGradCAM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = ViTReID(num_classes=0, img_size=256, pretrained=False).eval()

    def test_token_grid_matches_patches(self):
        cam = GradCAM(self.model)
        tokens = torch.randn(2, 1 + 16 * 16, 768)
        self.assertEqual(tuple(cam._as_grid(tokens).shape), (2, 768, 16, 16))

    def test_similarity_map_is_not_degenerate(self):
        images = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            reference = self.model(torch.randn(1, 3, 256, 256))[0]
        flags = [p.requires_grad for p in self.model.parameters()]
        with GradCAM(self.model) as cam:
            heat = cam.similarity_map(images, reference)
        self.assertEqual(heat.shape, (256, 256))
        self.assertGreater(float(heat.max()), 0.99)
        self.assertGreaterEqual(float(heat.min()), 0.0)
        # Флаги обучаемости восстановлены (смещение BNNeck заморожено и так),
        # режим модели не изменился.
        self.assertEqual([p.requires_grad for p in self.model.parameters()], flags)
        self.assertFalse(self.model.training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
