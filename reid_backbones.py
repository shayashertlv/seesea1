"""Simple appearance embedding helper used by the tracker."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ReIDSettings:
    """Configuration for the histogram based appearance encoder."""

    bins_h: int = 32
    bins_s: int = 16
    bins_v: int = 8

    def feature_length(self) -> int:
        return self.bins_h * self.bins_s * self.bins_v


class ReIDExtractor:
    """Compute L2-normalised HSV histograms for a batch of crops.

    The implementation intentionally avoids optional deep learning dependencies
    while still providing a deterministic appearance descriptor.  The extractor
    is thread-safe and can be re-used across worker threads.
    """

    def __init__(self, settings: ReIDSettings | None = None) -> None:
        self.settings = settings or ReIDSettings()
        self._lock = threading.Lock()

    def _encode_one(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.size == 0:
            raise ValueError("Empty crop provided to ReIDExtractor")
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=[self.settings.bins_h, self.settings.bins_s, self.settings.bins_v],
            ranges=[0, 180, 0, 256, 0, 256],
        )
        hist = hist.astype(np.float32).flatten()
        norm = np.linalg.norm(hist)
        if norm == 0.0:
            return np.zeros(self.settings.feature_length(), dtype=np.float32)
        return hist / norm

    def forward(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a batch of crops and return an array of shape ``(N, D)``."""
        with self._lock:
            features = [self._encode_one(crop) for crop in crops_bgr]
        if not features:
            return np.empty((0, self.settings.feature_length()), dtype=np.float32)
        return np.stack(features, axis=0)

    def forward_iter(self, crops_bgr: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
        """Generator variant of :meth:`forward` for streaming pipelines."""
        for crop in crops_bgr:
            yield self._encode_one(crop)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            raise ValueError("Feature vectors must have matching shapes")
        return float(np.clip(np.dot(a, b), -1.0, 1.0))
