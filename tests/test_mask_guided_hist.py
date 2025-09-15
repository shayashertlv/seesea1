import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tracker import _compute_hsv_hist, _hist_similarity  # noqa: E402


def test_mask_guided_crops_improve_hist_similarity():
    # ensure mask guidance is enabled
    import tracker  # type: ignore

    tracker.MASK_GUIDED_HIST = True

    frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(frame, (25, 25), (75, 75), (0, 255, 0), -1)
    bbox = (25.0, 25.0, 75.0, 75.0)

    mask_full = np.ones((50, 50), dtype=np.uint8)
    hist_ref = _compute_hsv_hist(frame, bbox, mask_roi=mask_full)

    frame_occ = frame.copy()
    cv2.rectangle(frame_occ, (25, 25), (50, 75), (0, 0, 0), -1)
    mask_vis = np.zeros((50, 50), dtype=np.uint8)
    mask_vis[:, 25:] = 1

    hist_bbox = _compute_hsv_hist(frame_occ, bbox)
    hist_masked = _compute_hsv_hist(frame_occ, bbox, mask_roi=mask_vis)

    sim_bbox = _hist_similarity(hist_ref, hist_bbox)
    sim_masked = _hist_similarity(hist_ref, hist_masked)

    assert sim_masked is not None and sim_bbox is not None
    assert sim_masked > sim_bbox

