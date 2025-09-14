import numpy as np
import cv2

class RAFTGMC:
    def __init__(self, model_name: str = "raft-small", device: str = "auto", fp16: bool = True):
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
            # Attempt hub load (will fail-open if no internet/weights)
            try:
                # Users may vendor their own RAFT; we keep this attempt guarded
                self.model = None
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
                 downscale: int = 2,
                 ransac_thresh: float = 3.0):
        """
        Returns (H:3x3 float32, stats: dict). If RAFT not available, raises RuntimeError to allow caller fallback.
        """
        if not self.ok or self.model is None:
            raise RuntimeError("RAFT not available")
        # Placeholder: if a real RAFT model was set, this is where you'd run it.
        # Since we do not ship one, signal failure and let caller fallback.
        raise RuntimeError("RAFT model not loaded")
