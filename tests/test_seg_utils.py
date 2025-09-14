import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from seg_utils import _grabcut_roi

def test_grabcut_roi_returns_mask_and_visibility():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (30, 30), (70, 70), (255, 255, 255), -1)
    box = (20.0, 20.0, 80.0, 80.0)
    mask, vis = _grabcut_roi(frame, box)
    assert mask is not None
    assert mask.shape == (60, 60)
    assert np.isclose(vis, mask.mean())
    assert 0.0 < vis <= 1.0

def test_grabcut_roi_invalid_box():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    # Box with no area
    box = (10.0, 10.0, 10.5, 10.5)
    mask, vis = _grabcut_roi(frame, box)
    assert mask is None
    assert vis == 0.0
