import threading
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reid_backbones import ReIDExtractor


def dummy_load_osnet(self):
    self.is_vit_square = False
    self.input_size = (256, 128)
    self.model = None


def test_pca_reduce_thread_safe(monkeypatch):
    monkeypatch.setattr(ReIDExtractor, "_load_osnet", dummy_load_osnet)
    ext = ReIDExtractor(backend="osnet", device="cpu")
    ext.pca_dim = 16
    with ReIDExtractor._pca_lock:
        ReIDExtractor._pca_ready = False
        ReIDExtractor._pca_mean = None
        ReIDExtractor._pca_comp = None
        ReIDExtractor._pca_buf.clear()
    sample = np.random.rand(64, 128).astype(np.float32)

    def worker():
        for _ in range(10):
            ext._pca_reduce_cached(sample, ext.pca_dim)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ReIDExtractor._pca_ready
    assert len(ReIDExtractor._pca_buf) <= ReIDExtractor._pca_buf.maxlen
    out = ext._pca_reduce_cached(sample[0], ext.pca_dim)
    assert out.shape == (ext.pca_dim,)
