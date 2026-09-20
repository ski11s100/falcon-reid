"""PCA-проекция вектора ансамбля: сохранение, применение, возврат для Grad-CAM."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falcon.extract import Projection  # noqa: E402


def make_projection(dim_in=6, dim_out=3, seed=0):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((dim_in, dim_in)))
    return Projection(torch.from_numpy(rng.standard_normal(dim_in).astype(np.float32) * 0.1),
                      torch.from_numpy(basis[:, :dim_out].astype(np.float32)), scale=0.8,
                      metadata={"dim": dim_out})


class TestProjection(unittest.TestCase):
    def test_apply_gives_unit_vectors_of_output_dim(self):
        projection = make_projection()
        vectors = torch.nn.functional.normalize(torch.randn(5, 6), dim=1)
        out = projection.apply(vectors)
        self.assertEqual(tuple(out.shape), (5, 3))
        self.assertTrue(torch.allclose(out.norm(dim=1), torch.ones(5), atol=1e-5))

    def test_save_and_load_round_trip(self):
        projection = make_projection()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projection.pt"
            projection.save(path)
            loaded = Projection.load(path)          # weights_only=True внутри
        self.assertEqual((loaded.input_dim, loaded.output_dim), (6, 3))
        self.assertTrue(torch.equal(loaded.components, projection.components))
        self.assertEqual(loaded.metadata, {"dim": 3})

    def test_back_restores_a_full_vector_close_in_direction(self):
        # Вектор, лежащий в подпространстве проекции, возвращается почти точно.
        projection = make_projection()
        inside = projection.mean + projection.components @ torch.tensor([0.5, -0.3, 0.2])
        projected = projection.apply(inside.unsqueeze(0))[0].numpy()
        restored = projection.back(projected)
        self.assertEqual(restored.shape, (6,))
        cosine = float(np.dot(restored - projection.mean.numpy(), (inside - projection.mean).numpy())
                       / np.linalg.norm(restored - projection.mean.numpy())
                       / np.linalg.norm((inside - projection.mean).numpy()))
        self.assertGreater(cosine, 0.999)

    def test_explain_parts_splits_back_projected_reference(self):
        from service.app import explain_parts
        from falcon.model import VehicleReID
        resnet = VehicleReID.__new__(VehicleReID)
        extractor = SimpleNamespace(
            projection=make_projection(),
            members=[SimpleNamespace(model=resnet, explain_model="A", feature_dim=4),
                     SimpleNamespace(model=resnet, explain_model="B", feature_dim=2)])
        parts = explain_parts(extractor, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual([m for m, _ in parts], ["A", "B"])
        self.assertEqual([len(p) for _, p in parts], [4, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
