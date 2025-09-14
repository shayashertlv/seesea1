import os
from typing import Tuple, Dict, Optional, List

import cv2
import numpy as np


def _identity_H() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


# Simple temporal smoothing for homographies to reduce jitter
_GMC_LAST_H: Optional[np.ndarray] = None

def _smooth_H(H: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    global _GMC_LAST_H
    try:
        Hc = H.astype(np.float32)
        if _GMC_LAST_H is None:
            _GMC_LAST_H = Hc
            return Hc
        # Exponential smoothing across the full matrix, keep H[2,2] normalized
        Hs = (alpha * Hc + (1.0 - alpha) * _GMC_LAST_H).astype(np.float32)
        if abs(float(Hs[2, 2])) > 1e-8:
            Hs = Hs / float(Hs[2, 2])
        _GMC_LAST_H = Hs
        return Hs
    except Exception:
        return H.astype(np.float32)


def estimate_gmc(prev_gray: np.ndarray,
                 curr_gray: np.ndarray,
                 *,
                 method: str = 'orb',
                 nfeatures: int = 1000,
                 ransac_thresh: float = 3.0,
                 downscale: int = 1,
                 mask_exclude_boxes: Optional[List[Tuple[float, float, float, float]]] = None
                 ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Estimate global motion between prev_gray and curr_gray using feature matching and RANSAC homography.
    - Exclude borders and current detection boxes from keypoint regions via mask to focus on static background.
    Returns (H, stats) where H is 3x3 float32 homography (or identity on failure) and stats include match counts.
    Note: stats['ok'] is a coarse success score (1.0 on confident homography, 0.0 otherwise). Callers may log
    periodic GMC health heartbeats (e.g., ratio of 'ok' decisions) to diagnose stability over time.
    """
    stats: Dict[str, float] = {
        'matches': 0.0,
        'good_matches': 0.0,
        'inliers': 0.0,
        'ok': 0.0,
    }
    try:
        if prev_gray is None or curr_gray is None:
            return _identity_H(), stats
        H_img, W_img = prev_gray.shape[:2]
        if downscale and downscale > 1:
            prev_small = cv2.resize(prev_gray, (W_img // downscale, H_img // downscale), interpolation=cv2.INTER_AREA)
            curr_small = cv2.resize(curr_gray, (W_img // downscale, H_img // downscale), interpolation=cv2.INTER_AREA)
            scale = 1.0 / float(downscale)
        else:
            prev_small = prev_gray
            curr_small = curr_gray
            scale = 1.0

        # Build mask to avoid borders and detection boxes (scaled)
        mask = np.ones_like(prev_small, dtype=np.uint8) * 255
        border = max(8, int(0.01 * max(mask.shape)))
        cv2.rectangle(mask, (0, 0), (mask.shape[1] - 1, mask.shape[0] - 1), 255, thickness=-1)
        cv2.rectangle(mask, (0, 0), (mask.shape[1] - 1, border), 0, thickness=-1)
        cv2.rectangle(mask, (0, mask.shape[0] - border), (mask.shape[1] - 1, mask.shape[0] - 1), 0, thickness=-1)
        cv2.rectangle(mask, (0, 0), (border, mask.shape[0] - 1), 0, thickness=-1)
        cv2.rectangle(mask, (mask.shape[1] - border, 0), (mask.shape[1] - 1, mask.shape[0] - 1), 0, thickness=-1)
        if mask_exclude_boxes:
            for (x1, y1, x2, y2) in mask_exclude_boxes:
                xs = int(max(0, min(mask.shape[1] - 1, round(x1 * scale))))
                ys = int(max(0, min(mask.shape[0] - 1, round(y1 * scale))))
                xe = int(max(0, min(mask.shape[1] - 1, round(x2 * scale))))
                ye = int(max(0, min(mask.shape[0] - 1, round(y2 * scale))))
                if xe > xs and ye > ys:
                    cv2.rectangle(mask, (xs, ys), (xe, ye), 0, thickness=-1)

        m = method.lower()
        if m in ('loftr', 'gmflow'):
            # Map to ORB as a lightweight fallback when heavy backends are unavailable
            m = 'orb'
        if m == 'raft':
            # Try RAFT backend (optional). On failure, fall back to ORB/OFLOW below.
            try:
                from gmc_raft import RAFTGMC  # type: ignore
                raft = getattr(estimate_gmc, "_raft", None)
                if raft is None:
                    raft = RAFTGMC(
                        device=(
                            "cuda" if (hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0) else "cpu"
                        )
                    )
                if raft.ok:
                    # Convert gray to BGR for RAFT if needed
                    if prev_small.ndim == 2:
                        prev_bgr = cv2.cvtColor(prev_small, cv2.COLOR_GRAY2BGR)
                        curr_bgr = cv2.cvtColor(curr_small, cv2.COLOR_GRAY2BGR)
                    else:
                        prev_bgr = prev_small
                        curr_bgr = curr_small
                    try:
                        H, stats = raft.estimate(prev_bgr, curr_bgr,
                                                exclude_boxes=mask_exclude_boxes,
                                                downscale=downscale,
                                                ransac_thresh=ransac_thresh)  # type: ignore[arg-type]
                        if H is not None and isinstance(H, np.ndarray):
                            return H.astype(np.float32), stats
                    except NotImplementedError:
                        pass
            except Exception:
                # Fall through to ORB/OFLOW
                pass
        if m == 'orb':
            orb = cv2.ORB_create(nfeatures=nfeatures)
            kpts1, des1 = orb.detectAndCompute(prev_small, mask)
            kpts2, des2 = orb.detectAndCompute(curr_small, mask)
            if des1 is None or des2 is None or len(kpts1) == 0 or len(kpts2) == 0:
                return _identity_H(), stats
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            stats['matches'] = float(len(matches))
            if len(matches) < 30:
                return _identity_H(), stats
            matches = sorted(matches, key=lambda x: x.distance)
            good = matches[: max(30, int(0.5 * len(matches)))]
            stats['good_matches'] = float(len(good))
            pts1 = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts2 = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
            if H is None:
                return _identity_H(), stats
            inlier_count = int(inliers.sum()) if inliers is not None else 0
            stats['inliers'] = float(inlier_count)
            if inlier_count < 30:
                return _identity_H(), stats
            stats['ok'] = 1.0
            return H.astype(np.float32), stats
        elif method.lower() == 'oflow':
            # GPU/CPU Farnebäck optical flow → RANSAC homography
            try:
                # prev_small, curr_small must exist (use your existing downscale logic)
                if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    prev_gpu = cv2.cuda_GpuMat(); prev_gpu.upload(prev_small)
                    curr_gpu = cv2.cuda_GpuMat(); curr_gpu.upload(curr_small)
                    of = cv2.cuda_FarnebackOpticalFlow.create(
                        5, 0.5, False, 15, 3, 5, 1.2, 0
                    )
                    flow = of.calc(prev_gpu, curr_gpu, None).download()  # HxWx2
                else:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_small, curr_small, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                Hh, Ww = prev_small.shape[:2]
                yy, xx = np.mgrid[0:Hh, 0:Ww]
                pts1 = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
                pts2 = (pts1 + flow.reshape(-1, 2)).astype(np.float32)
                # Use a subsample for RANSAC speed
                if pts1.shape[0] > 6000:
                    idx = np.random.choice(pts1.shape[0], size=6000, replace=False)
                    pts1_s, pts2_s = pts1[idx], pts2[idx]
                else:
                    pts1_s, pts2_s = pts1, pts2
                H, inliers = cv2.findHomography(pts1_s, pts2_s, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
                if H is None:
                    return _identity_H(), stats
                stats['ok'] = 1.0
                stats['inliers'] = float(inliers.sum()) if inliers is not None else 0.0
                return H.astype(np.float32), stats
            except Exception:
                return _identity_H(), stats
        else:
            # Fallback: identity
            return _identity_H(), stats
    except Exception:
        return _identity_H(), stats


def warp_point(cx: float, cy: float, H: Optional[np.ndarray]) -> Tuple[float, float]:
    if H is None:
        return float(cx), float(cy)
    try:
        p = np.array([[cx, cy, 1.0]], dtype=np.float32).T
        q = H @ p
        w = float(q[2, 0]) if abs(float(q[2, 0])) > 1e-8 else 1.0
        x = float(q[0, 0] / w)
        y = float(q[1, 0] / w)
        return x, y
    except Exception:
        return float(cx), float(cy)


def warp_box_xyxy(box: Tuple[float, float, float, float], H: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
    if H is None:
        return box
    try:
        x1, y1, x2, y2 = box
        pts = np.array([
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x2, y2, 1.0],
            [x1, y2, 1.0],
        ], dtype=np.float32).T
        q = H @ pts
        w = q[2, :]
        w = np.where(np.abs(w) > 1e-8, w, 1.0)
        xs = (q[0, :] / w).astype(np.float32)
        ys = (q[1, :] / w).astype(np.float32)
        nx1 = float(np.min(xs))
        ny1 = float(np.min(ys))
        nx2 = float(np.max(xs))
        ny2 = float(np.max(ys))
        if not (np.isfinite(nx1) and np.isfinite(ny1) and np.isfinite(nx2) and np.isfinite(ny2)):
            return box
        return (nx1, ny1, nx2, ny2)
    except Exception:
        return box
