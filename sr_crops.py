import os
import numpy as np
import cv2

class CropSR:
    def __init__(self, scale:int = 2):
        self.scale = max(1, int(scale))
        self.ok = False
        self._engine = None
        # Try to load Real-ESRGAN if available
        try:
            from realesrgan import RealESRGAN
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Use default x2 model if available; otherwise fail-open
            try:
                self._engine = RealESRGAN(torch.device(device), scale=self.scale)
                # Attempt to load default weights path from env; else rely on package defaults
                weights = os.getenv('REAL_ESRGAN_X2_WEIGHTS', None)
                if weights is not None and os.path.exists(weights):
                    self._engine.load_weights(weights)
                else:
                    # Let library handle its default weights resolution if packaged
                    pass
                self.ok = True
            except Exception:
                self._engine = None
                self.ok = False
        except Exception:
            self._engine = None
            self.ok = False
        self._warned = False

    def maybe_upscale(self, crop_bgr: np.ndarray, min_side: int = 64) -> np.ndarray:
        try:
            h, w = crop_bgr.shape[:2]
            if min(h, w) >= int(min_side):
                return crop_bgr
            # target side length
            if self.ok and self._engine is not None:
                # Engine expects RGB
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                sr = self._engine.predict(rgb)
                out = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                return out
            else:
                if not self._warned:
                    print('[sr] Real-ESRGAN unavailable; using bicubic fallback')
                    self._warned = True
                scale = max(2, int(round(float(min_side) / float(max(1, min(h, w))))))
                new_w = int(w * scale)
                new_h = int(h * scale)
                return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        except Exception:
            return crop_bgr
