from typing import Dict, Optional
import numpy as np
import math

class AppearanceMemory:
    def __init__(self,
                 backend: str = "ema",
                 alpha: float = 0.2,
                 k_protos: int = 3,
                 min_vis: float = 0.25,
                 min_side: int = 64,
                 max_blur: float = 2.0,
                 on_conflict_only: bool = True,
                 switch_margin: float = 0.08,
                 freeze_after_switch: int = 5):
        self.backend = (backend or "ema").lower()
        self.alpha = float(alpha)
        self.k_protos = int(k_protos)
        self.min_vis = float(min_vis)
        self.min_side = int(min_side)
        self.max_blur = float(max_blur)
        self.on_conflict_only = bool(on_conflict_only)
        self.switch_margin = float(switch_margin)
        self.freeze_after_switch = int(freeze_after_switch)
        # Per-track data
        self._m: Dict[int, np.ndarray] = {}
        self._protos: Dict[int, list] = {}
        self._cooldown: Dict[int, int] = {}

    def get(self, track_id: int) -> Optional[np.ndarray]:
        v = self._m.get(int(track_id))
        if isinstance(v, np.ndarray) and v.size > 0:
            n = float(np.linalg.norm(v))
            if n > 1e-12:
                return v / n
        return None

    def _quality_ok(self, q: Dict) -> bool:
        try:
            vis = float(q.get("vis", 1.0))
            side = float(q.get("min_side", 0.0))
            blur = float(q.get("blur", 0.0))
            conf = float(q.get("conf", 1.0))
        except Exception:
            return False
        if vis < self.min_vis:
            return False
        if side < self.min_side:
            return False
        # We treat lower LapVar as blurrier; allow if <= max_blur (threshold is in arbitrary units)
        if blur > self.max_blur:
            return False
        if conf < 0.30:
            return False
        return True

    def _clamp_angle(self, m: np.ndarray, z: np.ndarray, max_deg: float = 20.0) -> np.ndarray:
        # Return an adjusted target vector so that angle between m and target <= max_deg
        if m is None or m.size == 0:
            return z
        m_n = m / max(1e-12, np.linalg.norm(m))
        z_n = z / max(1e-12, np.linalg.norm(z))
        cos_t = float(np.clip(np.dot(m_n, z_n), -1.0, 1.0))
        ang = math.degrees(math.acos(cos_t))
        if ang <= max_deg:
            return z_n
        # Slerp towards z by limiting angle
        t = max_deg / max(1e-6, ang)
        v = (1.0 - t) * m_n + t * z_n
        n = float(np.linalg.norm(v))
        return (v / n) if n > 1e-12 else z_n

    def update(self, track_id: int, z_t: np.ndarray, quality: Dict) -> Optional[np.ndarray]:
        tid = int(track_id)
        if not isinstance(z_t, np.ndarray) or z_t.size == 0:
            return self.get(tid)
        # cooldown after switch
        cd = int(self._cooldown.get(tid, 0))
        if cd > 0:
            self._cooldown[tid] = cd - 1
            return self.get(tid)
        if not self._quality_ok(quality):
            return self.get(tid)
        z = z_t.astype(np.float32)
        n = float(np.linalg.norm(z))
        if n > 1e-12:
            z = z / n
        else:
            return self.get(tid)
        m_prev = self._m.get(tid, None)
        if m_prev is None or m_prev.size == 0:
            self._m[tid] = z
        else:
            target = self._clamp_angle(m_prev, z, max_deg=20.0)
            m_new = (1.0 - self.alpha) * m_prev + self.alpha * target
            nn = float(np.linalg.norm(m_new))
            if nn > 1e-12:
                m_new = m_new / nn
            self._m[tid] = m_new.astype(np.float32)
        # update prototypes by simple quality score (vis * conf * side)
        protos = self._protos.get(tid, [])
        score = float(quality.get("vis", 0.0)) * float(quality.get("conf", 1.0)) * max(1.0, float(quality.get("min_side", 0.0)))
        protos.append((score, z))
        protos = sorted(protos, key=lambda x: -x[0])[:max(1, self.k_protos)]
        self._protos[tid] = protos
        return self._m.get(tid)

    def freeze(self, track_id: int, n_frames: Optional[int] = None) -> None:
        self._cooldown[int(track_id)] = int(n_frames if n_frames is not None else self.freeze_after_switch)
