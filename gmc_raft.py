import os
import numpy as np
import cv2


class RAFTGMC:
    def __init__(self, model_name: str = "raft-small", model_path: str | None = None,
                 device: str = "auto", fp16: bool = True):
        self.ok = False
        self.model = None
        self.device = "cpu"
        self.fp16 = bool(fp16)
        # Lazy: try to import a RAFT implementation if present in environment
        try:
            import torch
            self.torch = torch
            if device == "cuda" and torch.cuda.is_available():
                self.device = "cuda"
            elif device == "auto" and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            # Attempt to load a RAFT model either from torch.hub or a provided path
            try:
                if model_path and os.path.isfile(model_path):
                    # Attempt to load RAFT from local weights
                    from raft import RAFT  # type: ignore
                    from argparse import Namespace
                    args = Namespace(
                        small=("small" in model_name),
                        mixed_precision=self.fp16,
                        alternate_corr=False,
                    )
                    m = RAFT(args)
                    state = torch.load(model_path, map_location=self.device)
                    m.load_state_dict(state)
                else:
                    # Will fail open if hub/model is not accessible
                    m = torch.hub.load("princeton-vl/RAFT", model_name,
                                       pretrained=True, map_location=self.device)
                m.to(self.device)
                m.eval()
                if self.fp16:
                    m = m.half()
                self.model = m
            except Exception:
                self.model = None
            # If no model loaded, we still allow object creation; ok remains False
            self.ok = self.model is not None
        except Exception:
            self.torch = None
            self.ok = False

    def estimate(self,
                 prev_bgr: np.ndarray,
                 curr_bgr: np.ndarray,
                 exclude_boxes=None,
                 downscale: int = 1,
                 ransac_thresh: float = 3.0):
        """
        Returns (H:3x3 float32, stats: dict).

        If a RAFT model is not available this method raises ``NotImplementedError`` to signal
        that the backend is stubbed and callers should fall back to alternative GMC methods.
        """
        if not self.ok or self.model is None or self.torch is None:
            raise NotImplementedError("RAFT backend not available")

        Hh, Ww = prev_bgr.shape[:2]
        scale = 1.0 / float(max(1, downscale))

        # Build mask to exclude borders and detection boxes
        mask = np.ones((Hh, Ww), dtype=np.uint8) * 255
        border = max(8, int(0.01 * max(mask.shape)))
        cv2.rectangle(mask, (0, 0), (Ww - 1, Hh - 1), 255, thickness=-1)
        cv2.rectangle(mask, (0, 0), (Ww - 1, border), 0, thickness=-1)
        cv2.rectangle(mask, (0, Hh - border), (Ww - 1, Hh - 1), 0, thickness=-1)
        cv2.rectangle(mask, (0, 0), (border, Hh - 1), 0, thickness=-1)
        cv2.rectangle(mask, (Ww - border, 0), (Ww - 1, Hh - 1), 0, thickness=-1)
        if exclude_boxes:
            for (x1, y1, x2, y2) in exclude_boxes:
                xs = int(max(0, min(Ww - 1, round(x1 * scale))))
                ys = int(max(0, min(Hh - 1, round(y1 * scale))))
                xe = int(max(0, min(Ww - 1, round(x2 * scale))))
                ye = int(max(0, min(Hh - 1, round(y2 * scale))))
                if xe > xs and ye > ys:
                    cv2.rectangle(mask, (xs, ys), (xe, ye), 0, thickness=-1)

        torch = self.torch
        with torch.no_grad():
            im1 = torch.from_numpy(prev_bgr).permute(2, 0, 1).float() / 255.0
            im2 = torch.from_numpy(curr_bgr).permute(2, 0, 1).float() / 255.0
            if self.fp16:
                im1 = im1.half()
                im2 = im2.half()
            flow_up = self.model(im1[None].to(self.device),
                                 im2[None].to(self.device),
                                 iters=20, test_mode=True)[1]
            flow = flow_up[0].permute(1, 2, 0).float().cpu().numpy()

        yy, xx = np.mgrid[0:Hh, 0:Ww]
        pts1 = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        pts2 = (pts1 + flow.reshape(-1, 2)).astype(np.float32)
        m_flat = mask.reshape(-1) > 0
        pts1 = pts1[m_flat]
        pts2 = pts2[m_flat]

        stats = {
            'matches': float(len(pts1)),
            'good_matches': float(len(pts1)),
            'inliers': 0.0,
            'ok': 0.0,
        }

        if pts1.shape[0] < 30:
            return np.eye(3, dtype=np.float32), stats

        if pts1.shape[0] > 6000:
            idx = np.random.choice(pts1.shape[0], size=6000, replace=False)
            pts1_s, pts2_s = pts1[idx], pts2[idx]
        else:
            pts1_s, pts2_s = pts1, pts2

        H, inliers = cv2.findHomography(pts1_s, pts2_s, cv2.RANSAC,
                                        ransacReprojThreshold=ransac_thresh)
        if H is None:
            return np.eye(3, dtype=np.float32), stats
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        stats['inliers'] = float(inlier_count)
        stats['ok'] = 1.0 if inlier_count >= 30 else 0.0
        return H.astype(np.float32), stats
