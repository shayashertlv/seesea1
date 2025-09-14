from typing import List, Tuple, Optional

from seg_common import grabcut_roi_masks

_logged = False


def infer_roi_masks(frame_bgr,
                    det_xyxy: List[Tuple[float, float, float, float]]):
    """
    Heavy segmentation placeholder for Mask2Former. Fail-open:
    - If Mask2Former deps are not available, fall back to a quick GrabCut on each ROI.
    Returns (masks, boxes, visibilities) aligned one-to-one with det_xyxy.
    Each mask is ROI-sized (h,w) uint8 with values in {0,1}.
    """
    global _logged
    n = len(det_xyxy)
    try:
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
        return grabcut_roi_masks(frame_bgr, det_xyxy)
    except Exception:
        return [None]*n, [None]*n, [0.0]*n
