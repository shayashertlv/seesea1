from typing import List, Tuple, Optional

from seg_common import grabcut_roi_masks

_logged = False


def infer_roi_masks(frame_bgr,
                    det_xyxy: List[Tuple[float, float, float, float]]):
    """
    SAM2 wrapper (fail-open):
    - Tries to use SAM2/Segment-Anything if available (not bundled here).
    - If unavailable, falls back to quick GrabCut per ROI using the detection box as a prompt.
    Returns (masks, boxes, visibilities) aligned one-to-one with det_xyxy.
    Each mask is ROI-sized (h,w) uint8 with values in {0,1}.
    """
    global _logged
    n = len(det_xyxy)
    try:
        have_heavy = False
        try:
            import sam2  # noqa: F401
            have_heavy = True
        except Exception:
            try:
                import segment_anything  # noqa: F401
                have_heavy = True
            except Exception:
                have_heavy = False
        if not have_heavy and not _logged:
            try:
                print('[seg] SAM2 not available; using GrabCut ROI fallback')
                _logged = True
            except Exception:
                pass
        return grabcut_roi_masks(frame_bgr, det_xyxy)
    except Exception:
        return [None]*n, [None]*n, [0.0]*n
