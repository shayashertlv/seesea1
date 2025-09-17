import pytest

import models.seqtrack as seqtrack
from models.seqtrack import HAS_TORCH, SeqTrackLSTM

np = pytest.importorskip("numpy")


def test_seqtracklstm_instantiation() -> None:
    model = SeqTrackLSTM()
    assert hasattr(model, "predict")
    assert hasattr(model, "device")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_seqtracklstm_predict_with_appearance_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seqtrack, "LSTM_APPEAR_ENABLE", True, raising=False)
    model = SeqTrackLSTM(variant='B', device='cpu', fp16=False)
    monkeypatch.setattr(model, "LSTM_APPEAR_ENABLE", True, raising=False)

    track_window = {
        "centers": [(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)],
        "reid_seq": [np.ones((4,), dtype=np.float32) * 0.5, np.ones((4,), dtype=np.float32)],
        "hist_seq": [np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32) * 0.5],
    }

    out = model.predict(track_window)

    assert "app_mem" in out
    app_mem = out["app_mem"]
    assert isinstance(app_mem, np.ndarray)
    assert app_mem.ndim == 1
    assert np.linalg.norm(app_mem) > 0.0
