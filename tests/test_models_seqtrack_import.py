import pytest

np = pytest.importorskip("numpy")

import models.seqtrack as seqtrack
from models.seqtrack import HAS_TORCH, SeqTrackLSTM


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


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_seqtracklstm_predict_invokes_iou(monkeypatch: pytest.MonkeyPatch) -> None:
    model = SeqTrackLSTM(device='cpu', fp16=False)

    calls = {"count": 0}

    def fake_iou(a, b):
        calls["count"] += 1
        return 0.5

    monkeypatch.setattr(seqtrack, "_iou_xyxy", fake_iou, raising=False)

    track_window = {
        "centers": [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0)],
        "boxes": [
            (0.0, 0.0, 2.0, 2.0),
            (0.5, 0.5, 2.5, 2.5),
            (1.0, 1.0, 3.0, 3.0),
        ],
    }

    out = model.predict(track_window)

    assert calls["count"] >= 1
    assert isinstance(out, dict)
    assert any(abs(v) > 1e-6 for v in out["delta"])
