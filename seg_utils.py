import numpy as np
import cv2
from typing import Tuple, Optional

def _grabcut_roi(frame_bgr: np.ndarray,
                  box: Tuple[float, float, float, float]) -> Tuple[Optional[np.ndarray], float]:
    """Quick GrabCut ROI segmentation.

    Parameters
    ----------
    frame_bgr: np.ndarray
        Full frame in BGR color space.
    box: tuple(float, float, float, float)
        ROI bounding box as (x1, y1, x2, y2).

    Returns
    -------
    mask: Optional[np.ndarray]
        ROI-sized binary mask with values in {0,1}. ``None`` if the ROI is invalid.
    vis: float
        Visibility score in [0,1] computed as ``mask.mean()`` or 0.0 if ``mask`` is ``None``.
    """
    H, W = frame_bgr.shape[:2]
    try:
        x1, y1, x2, y2 = map(float, box)
        xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        xi2, yi2 = min(W, int(np.ceil(x2))), min(H, int(np.ceil(y2)))
        if xi2 - xi1 <= 1 or yi2 - yi1 <= 1:
            return None, 0.0
        roi = frame_bgr[yi1:yi2, xi1:xi2]
        m = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
        rect = (
            int(0.06 * roi.shape[1]),
            int(0.06 * roi.shape[0]),
            int(0.88 * roi.shape[1]),
            int(0.88 * roi.shape[0]),
        )
        # OpenCV's grabCut requires preallocated background/foreground models.
        # They are mutated in-place, so reinitialise them for every ROI.
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(roi, m, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            mask = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        except Exception:
            mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        return mask, float(mask.mean())
    except Exception:
        return None, 0.0
