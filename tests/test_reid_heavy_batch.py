import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reid_backbones import ReIDExtractor


class DummyModel(torch.nn.Module):
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.mean(dim=(2, 3))


def make_extractor(tta_mode: int = 1) -> ReIDExtractor:
    ext = ReIDExtractor(backend="osnet", device="cpu")
    ext.backend = "clip_vitl14"
    ext.model = DummyModel()
    ext.device = torch.device("cpu")
    ext.tta_mode = tta_mode
    ext.pca_dim = 0

    def _preprocess_clip(self, bgr):
        rgb = bgr[:, :, ::-1]
        ten = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return ten

    ext._preprocess_clip = _preprocess_clip.__get__(ext, ReIDExtractor)
    return ext


def test_heavy_batch_matches_individual():
    ext = make_extractor(tta_mode=1)
    crops = [np.random.randint(0, 256, (64, 32, 3), dtype=np.uint8) for _ in range(5)]
    seq = np.stack([ext._embed_one(c) for c in crops], axis=0)
    bat = ext.forward(crops, batch_size=2)
    assert np.allclose(seq, bat, atol=1e-6)
