import numpy as np
import cv2
from typing import List, Tuple, Optional

_logged = False

def infer_roi_masks(frame_bgr: np.ndarray,
                    det_xyxy: List[Tuple[float, float, float, float]]):
    """
    Heavy segmentation placeholder for Mask2Former. Fail-open:
    - If Mask2Former deps are not available, fall back to a quick GrabCut on each ROI.
    Returns (masks, boxes, visibilities) aligned one-to-one with det_xyxy.
    Each mask is ROI-sized (h,w) uint8 with values in {0,1}.
    """
    global _logged
    H, W = frame_bgr.shape[:2]
    n = len(det_xyxy)
    masks: List[Optional[np.ndarray]] = [None] * n
    boxes: List[Optional[Tuple[float,float,float,float]]] = [None] * n
    vis: List[float] = [0.0] * n
    try:
        # Attempt heavy deps (not provided here). If present, user can extend.
        have_heavy = False
        try:
            import detectron2  # noqa: F401
            have_heavy = True
        except Exception:
            have_heavy = False
        if not have_heavy and not _logged:
            try:
                print('[seg] Mask2Former not available; using GrabCut ROI fallback')
                _logged = True
            except Exception:
                pass
        for i, b in enumerate(det_xyxy):
            try:
                x1,y1,x2,y2 = map(float, b)
                xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
                xi2, yi2 = min(W, int(np.ceil(x2))), min(H, int(np.ceil(y2)))
                if xi2-xi1 <= 1 or yi2-yi1 <= 1:
                    continue
                roi = frame_bgr[yi1:yi2, xi1:xi2]
                # Quick GrabCut fallback
                m = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
                rect = (int(0.06*roi.shape[1]), int(0.06*roi.shape[0]), int(0.88*roi.shape[1]), int(0.88*roi.shape[0]))
                try:
                    cv2.grabCut(roi, m, rect, None, None, 2, cv2.GC_INIT_WITH_RECT)
                    mask = np.where((m==cv2.GC_FGD)|(m==cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
                except Exception:
                    mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                masks[i] = mask
                boxes[i] = (x1,y1,x2,y2)
                vis[i] = float(mask.mean())
            except Exception:
                pass
        return masks, boxes, vis
    except Exception:
        return [None]*n, [None]*n, [0.0]*n
