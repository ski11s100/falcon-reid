"""Optional ML tests; the baseline installation does not require PyTorch."""
import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'), 'ML dependencies not installed')
class NeuralTests(unittest.TestCase):
    def test_checkpoint_preprocessing_matches_training(self):
        import numpy as np
        import torch
        from PIL import Image
        from app.neural import NeuralEncoder, ReID, preprocessing
        torch.set_num_threads(1)
        model = ReID(1).eval()
        rng = np.random.default_rng(42)
        image = Image.fromarray(rng.integers(0,256,(279,417,3),dtype=np.uint8))
        checkpoint = {'architecture':'resnet18-reid-v1','classes':1,
                      'model':model.state_dict(),'preprocessing':'pil-bicubic-256x160-v1'}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.pt'
            torch.save(checkpoint, path)
            encoder = NeuralEncoder(path)
            with torch.inference_mode():
                expected = model(preprocessing()(image.resize((256,160)))[None])[0][0].numpy()
            np.testing.assert_allclose(encoder(image), expected, atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.norm(encoder(image))),1.0,places=5)
            checkpoint['preprocessing'] = 'unknown'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'preprocessing'):
                NeuralEncoder(path)


if __name__ == '__main__':
    unittest.main()
