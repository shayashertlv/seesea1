import logging
import os
import numpy as np
import cv2


LOGGER = logging.getLogger(__name__)

class CropSR:
    def __init__(self, scale:int = 2):
        self.scale = max(1, int(scale))
        self.ok = False
        self._engine = None
        # Try to load Real-ESRGAN if available
        try:
            from realesrgan import RealESRGAN  # type: ignore
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                self._engine = RealESRGAN(torch.device(device), scale=self.scale)
                weights = os.getenv('REAL_ESRGAN_X2_WEIGHTS', None)
                if weights is not None and os.path.exists(weights):
                    self._engine.load_weights(weights)
                self.ok = True
            except (RuntimeError, OSError) as exc:
                LOGGER.debug("Real-ESRGAN initialization failed: %s", exc)
                self._engine = None
                self.ok = False
        except ImportError as exc:
            LOGGER.debug("Real-ESRGAN unavailable: %s", exc)
            self._engine = None
            self.ok = False
        self._warned = False

    def maybe_upscale(self, crop_bgr: np.ndarray, min_side: int = 64) -> np.ndarray:
        try:
            h, w = crop_bgr.shape[:2]
            if min(h, w) >= int(min_side):
                return crop_bgr
            if self.ok and self._engine is not None:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                sr = self._engine.predict(rgb)
                out = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                return out
            if not self._warned:
                LOGGER.warning("Real-ESRGAN unavailable; using bicubic fallback")
                self._warned = True
            scale = max(2, int(round(float(min_side) / float(max(1, min(h, w))))))
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        except cv2.error as exc:
            LOGGER.debug("OpenCV failure during upscale: %s", exc)
            return crop_bgr
