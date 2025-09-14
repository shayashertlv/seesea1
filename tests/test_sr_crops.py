import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sr_crops import CropSR


class DummyEngine:
    def predict(self, img):
        return img


def _setup_sr():
    sr = CropSR(scale=2)
    sr.ok = True
    sr._engine = DummyEngine()
    return sr


def test_expected_exception_handled(monkeypatch):
    sr = _setup_sr()
    crop = np.zeros((4, 4, 3), dtype=np.uint8)

    def raise_cv2_error(*args, **kwargs):
        raise cv2.error("mock")

    monkeypatch.setattr(cv2, "cvtColor", raise_cv2_error)
    out = sr.maybe_upscale(crop, min_side=8)
    assert out is crop


def test_unexpected_exception_propagates(monkeypatch):
    sr = _setup_sr()
    crop = np.zeros((4, 4, 3), dtype=np.uint8)

    def raise_value_error(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(cv2, "cvtColor", raise_value_error)
    with pytest.raises(ValueError):
        sr.maybe_upscale(crop, min_side=8)
