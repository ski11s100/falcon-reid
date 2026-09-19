"""Аугментация PlateZoneErase: закрашивает низ по центру кропа и больше ничего."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.transforms import PlateZoneErase, build_train_transform  # noqa: E402


class TestPlateZoneErase(unittest.TestCase):
    def test_always_fills_lower_centre_with_one_colour(self):
        torch.manual_seed(0)
        erase = PlateZoneErase(probability=1.0)
        for _ in range(50):
            image = torch.randn(3, 256, 256)
            out = erase(image)
            changed = (out != image).any(0)
            rows, cols = changed.nonzero(as_tuple=True)
            self.assertGreater(len(rows), 0)
            # Зона внизу по центру кропа: номер не бывает у крыши или у края.
            self.assertGreaterEqual(rows.min().item(), int(0.52 * 256))
            self.assertLessEqual(rows.max().item(), int(0.96 * 256))
            self.assertGreaterEqual(cols.min().item(), int(0.18 * 256))
            self.assertLessEqual(cols.max().item(), int(0.82 * 256))
            # Заливка сплошная: во всей зоне один цвет.
            filled = out[:, rows, cols]
            self.assertTrue(torch.allclose(filled, filled[:, :1].expand_as(filled)))

    def test_probability_zero_keeps_image(self):
        image = torch.randn(3, 64, 64)
        self.assertTrue(torch.equal(PlateZoneErase(probability=0.0)(image), image))

    def test_input_not_modified_in_place(self):
        image = torch.randn(3, 64, 64)
        copy = image.clone()
        PlateZoneErase(probability=1.0)(image)
        self.assertTrue(torch.equal(image, copy))

    def test_train_transform_includes_erase_only_when_asked(self):
        names = lambda t: [type(step).__name__ for step in t.transforms]
        self.assertNotIn("PlateZoneErase", names(build_train_transform()))
        self.assertIn("PlateZoneErase", names(build_train_transform(plate_erase=0.5)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
