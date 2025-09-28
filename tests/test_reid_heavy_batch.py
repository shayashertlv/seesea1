import numpy as np

from reid_backbones import ReIDExtractor


def test_batch_matches_sequential() -> None:
    extractor = ReIDExtractor()
    crops = [np.random.randint(0, 255, (64, 32, 3), dtype=np.uint8) for _ in range(5)]
    sequential = np.stack([feat for feat in extractor.forward_iter(crops)], axis=0)
    batched = extractor.forward(crops)
    np.testing.assert_allclose(sequential, batched, atol=1e-6)
