import threading

import numpy as np

from reid_backbones import ReIDExtractor


def test_thread_safe_forward() -> None:
    extractor = ReIDExtractor()
    crop = np.random.randint(0, 255, (48, 24, 3), dtype=np.uint8)
    outputs = []

    def worker() -> None:
        for _ in range(10):
            feat = extractor.forward([crop])[0]
            outputs.append(feat)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(outputs) == 40
    reference = outputs[0]
    for feat in outputs:
        assert feat.shape == reference.shape
        np.testing.assert_allclose(np.linalg.norm(feat), 1.0, atol=1e-5)
        np.testing.assert_allclose(feat, reference, atol=1e-5)
