from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import math
import numpy as np


@dataclass
class _Prototype:
    """Container for an appearance prototype."""

    vec: np.ndarray
    score: float
    timestamp: float

    def clone(self) -> "_Prototype":
        return _Prototype(self.vec.copy(), float(self.score), float(self.timestamp))


class AppearanceMemory:
    """Time-decayed prototype memory for per-track appearance."""

    def __init__(self,
                 backend: str = "queue",
                 alpha: float = 0.2,
                 k_protos: int = 3,
                 min_vis: float = 0.25,
                 min_side: int = 64,
                 max_blur: float = 2.0,
                 on_conflict_only: bool = True,
                 switch_margin: float = 0.08,
                 freeze_after_switch: int = 5,
                 time_decay: float = 45.0,
                 max_age: int = 180,
                 cluster_sim: float = 0.92):
        # legacy knobs (kept for backwards compatibility with configs/tests)
        self.backend = (backend or "queue").lower()
        if self.backend not in ("queue", "proto", "ema"):
            self.backend = "queue"
        self.alpha = float(alpha)
        self.k_protos = max(1, int(k_protos))
        self.min_vis = float(min_vis)
        self.min_side = int(min_side)
        self.max_blur = float(max_blur)
        self.on_conflict_only = bool(on_conflict_only)
        self.switch_margin = float(switch_margin)
        self.freeze_after_switch = int(freeze_after_switch)
        self.time_decay = max(0.0, float(time_decay))
        self.max_age = max(0, int(max_age))
        self.cluster_sim = float(np.clip(cluster_sim, -1.0, 1.0))
        # Per-track data
        self._m: Dict[int, np.ndarray] = {}
        self._protos: Dict[int, List[_Prototype]] = {}
        self._last_t: Dict[int, float] = {}
        self._cooldown: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, track_id: int, current_time: Optional[float] = None) -> Optional[np.ndarray]:
        tid = int(track_id)
        protos = self._protos.get(tid, [])
        if not protos:
            return None
        now = current_time
        if now is None:
            now = self._last_t.get(tid, None)
        vec = self._aggregate(tid, protos, now)
        if vec is not None:
            self._m[tid] = vec
        return vec

    def get_prototypes(self, track_id: int, current_time: Optional[float] = None) -> List[np.ndarray]:
        tid = int(track_id)
        protos = self._protos.get(tid, [])
        if not protos:
            return []
        now = current_time
        if now is None:
            now = self._last_t.get(tid, None)
        keep: List[_Prototype] = []
        out: List[np.ndarray] = []
        for proto in protos:
            age = 0.0
            if now is not None:
                age = float(now) - float(proto.timestamp)
            if self.max_age > 0 and age > self.max_age:
                continue
            keep.append(proto)
            out.append(proto.vec.copy())
        self._protos[tid] = keep
        return out

    def remove(self, track_id: int) -> None:
        tid = int(track_id)
        self._protos.pop(tid, None)
        self._m.pop(tid, None)
        self._last_t.pop(tid, None)
        self._cooldown.pop(tid, None)

    def update(self, track_id: int, z_t: np.ndarray, quality: Dict) -> Optional[np.ndarray]:
        tid = int(track_id)
        if not isinstance(z_t, np.ndarray) or z_t.size == 0:
            return self.get(tid)
        cd = int(self._cooldown.get(tid, 0))
        if cd > 0:
            self._cooldown[tid] = cd - 1
            return self.get(tid)
        if not self._quality_ok(quality):
            return self.get(tid)
        weight = self._quality_weight(quality)
        if weight <= 0.0:
            return self.get(tid)
        z = z_t.astype(np.float32)
        n = float(np.linalg.norm(z))
        if n > 1e-12:
            z = z / n
        else:
            return self.get(tid)

        timestamp = self._quality_time(quality)
        if timestamp is None:
            prev = self._last_t.get(tid, None)
            timestamp = 0.0 if prev is None else float(prev + 1.0)
        self._last_t[tid] = float(timestamp)

        score = self._quality_score(weight, quality)
        protos = list(self._protos.get(tid, []))
        merged = False
        best_idx = -1
        best_sim = self.cluster_sim
        for i, proto in enumerate(protos):
            cs = float(np.dot(proto.vec, z))
            if cs > best_sim:
                best_sim = cs
                best_idx = i
        if best_idx >= 0:
            proto = protos[best_idx]
            merged_vec = proto.vec * proto.score + z * score
            n2 = float(np.linalg.norm(merged_vec))
            if n2 > 1e-12:
                merged_vec = merged_vec / n2
            proto.vec = merged_vec.astype(np.float32)
            proto.score = float(np.clip(proto.score + score, 1e-4, 100.0))
            proto.timestamp = float(timestamp)
            protos[best_idx] = proto
            merged = True
        if not merged:
            protos.append(_Prototype(vec=z.astype(np.float32),
                                     score=float(np.clip(score, 1e-4, 100.0)),
                                     timestamp=float(timestamp)))
        protos = self._prune_and_order(protos)
        self._protos[tid] = protos

        vec = self._aggregate(tid, protos, timestamp)
        if vec is not None:
            self._m[tid] = vec
        return vec

    def freeze(self, track_id: int, n_frames: Optional[int] = None) -> None:
        self._cooldown[int(track_id)] = int(n_frames if n_frames is not None else self.freeze_after_switch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        if blur > self.max_blur:
            return False
        if conf < 0.30:
            return False
        return True

    def _quality_weight(self, q: Dict) -> float:
        try:
            vis = float(q.get("vis", 1.0))
            side = float(q.get("min_side", 0.0))
            blur = float(q.get("blur", 0.0))
            conf = float(q.get("conf", 1.0))
        except Exception:
            return 0.0
        vis_score = float(np.clip((vis - self.min_vis) / max(1e-3, 1.0 - self.min_vis), 0.0, 1.0))
        side_score = float(np.clip((side - self.min_side) / max(1.0, self.min_side), 0.0, 1.0))
        blur_score = float(np.clip((self.max_blur - blur) / max(self.max_blur, 1e-3), 0.0, 1.0))
        conf_score = float(np.clip((conf - 0.30) / 0.70, 0.0, 1.0))
        return float(np.clip((vis_score + side_score + blur_score + conf_score) / 4.0, 0.0, 1.0))

    def _aggregate(self, tid: int, protos: Sequence[_Prototype], now: Optional[float]) -> Optional[np.ndarray]:
        keep: List[_Prototype] = []
        acc = None
        total = 0.0
        for proto in protos:
            age = 0.0
            if now is not None:
                age = float(now) - float(proto.timestamp)
            if age < 0.0:
                age = 0.0
            if self.max_age > 0 and age > self.max_age:
                continue
            decay = 1.0
            if self.time_decay > 0.0:
                decay = math.exp(-age / max(1e-6, self.time_decay))
            w = proto.score * decay
            if w <= 0.0:
                continue
            keep.append(proto)
            vec = proto.vec.astype(np.float32)
            if acc is None:
                acc = w * vec
            else:
                acc = acc + w * vec
            total += w
        self._protos[tid] = keep[:self.k_protos]
        if acc is None or total <= 0.0:
            self._m.pop(tid, None)
            return None
        n = float(np.linalg.norm(acc))
        if n <= 1e-12:
            self._m.pop(tid, None)
            return None
        return (acc / n).astype(np.float32)

    def _prune_and_order(self, protos: Sequence[_Prototype]) -> List[_Prototype]:
        ordered = sorted(protos, key=lambda p: (-float(p.timestamp), -float(p.score)))
        if len(ordered) > self.k_protos:
            ordered = ordered[:self.k_protos]
        return ordered

    def _quality_time(self, q: Dict) -> Optional[float]:
        for key in ("timestamp", "ts", "frame", "frame_idx", "t"):
            if key in q:
                try:
                    return float(q[key])
                except Exception:
                    continue
        return None

    def _quality_score(self, weight: float, q: Dict) -> float:
        try:
            vis = float(q.get("vis", 1.0))
            conf = float(q.get("conf", 1.0))
            side = float(q.get("min_side", self.min_side))
        except Exception:
            vis = 1.0
            conf = 1.0
            side = float(self.min_side)
        side_factor = max(1.0, side / max(1.0, float(self.min_side)))
        return float(np.clip(weight * vis * conf * side_factor, 1e-4, 10.0))
