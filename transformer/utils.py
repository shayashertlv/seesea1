"""Utility helpers shared by transformer modules."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def stack_embeddings(
    vectors: Sequence[Optional[np.ndarray]],
    max_dim: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Return a dense matrix by zero-padding optional embedding vectors."""
    max_len = 0
    if max_dim is not None:
        max_len = int(max_dim)
    else:
        for vec in vectors:
            if isinstance(vec, np.ndarray) and vec.size > 0:
                max_len = max(max_len, int(vec.size))
    if max_len <= 0:
        return np.zeros((len(vectors), 0), dtype=np.float32), 0
    out = np.zeros((len(vectors), max_len), dtype=np.float32)
    for i, vec in enumerate(vectors):
        if not isinstance(vec, np.ndarray) or vec.size == 0:
            continue
        flat = vec.astype(np.float32, copy=False).reshape(-1)
        if max_dim is not None and flat.size > max_len:
            raise ValueError(
                f"received embedding with width {flat.size} but expected at most {max_len}"
            )
        take = min(max_len, flat.size)
        out[i, :take] = flat[:take]
    return out, max_len


def pack_features(
    rows: Iterable[Sequence[float]],
    width: Optional[int] = None,
) -> np.ndarray:
    """Convert an iterable of numeric rows into a dense float32 array."""
    buf = [tuple(row) for row in rows]
    if not buf:
        if width is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.zeros((0, width), dtype=np.float32)
    if width is None:
        width = len(buf[0])
    if width == 0:
        return np.zeros((len(buf), 0), dtype=np.float32)
    for row in buf:
        if len(row) != width:
            raise ValueError(f"expected {width} features, received {len(row)}")
    return np.asarray(buf, dtype=np.float32)


def normalize_bbox(
    box: Sequence[float],
    width: float,
    height: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(1e-6, float(width))
    h = max(1e-6, float(height))
    return (
        float(x1) / w,
        float(y1) / h,
        float(x2) / w,
        float(y2) / h,
    )
