"""Global ReID embedding bank with FAISS/KD-tree acceleration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - faiss not available
    faiss = None  # type: ignore
    _HAS_FAISS = False

try:  # pragma: no cover - optional dependency
    from sklearn.neighbors import KDTree as _SklearnKDTree  # type: ignore

    _HAS_SKLEARN_KDTREE = True
except Exception:  # pragma: no cover - sklearn not available
    _SklearnKDTree = None  # type: ignore
    _HAS_SKLEARN_KDTREE = False

try:  # pragma: no cover - optional dependency
    from scipy.spatial import cKDTree as _ScipyKDTree  # type: ignore

    _HAS_SCIPY_KDTREE = True
except Exception:  # pragma: no cover - scipy not available
    _ScipyKDTree = None  # type: ignore
    _HAS_SCIPY_KDTREE = False


@dataclass(eq=False)
class _BankRecord:
    """Internal representation of a single embedding in the bank."""

    track_id: int
    embedding: np.ndarray
    last_seen: int
    meta: Optional[Dict[str, Any]] = None


class GlobalReIDBank:
    """A lightweight global embedding bank for long-gap ID recovery."""

    def __init__(
        self,
        dim: Optional[int] = None,
        *,
        max_records: int = 2048,
        ttl_frames: Optional[int] = 900,
        per_track_max: int = 4,
        backend: str = "auto",
    ) -> None:
        self.dim: Optional[int] = dim
        self.max_records = max_records
        self.ttl_frames = ttl_frames if (ttl_frames is not None and ttl_frames > 0) else None
        self.per_track_max = max(1, per_track_max) if per_track_max > 0 else None
        self._records: List[_BankRecord] = []
        self._dirty = True
        self._matrix: Optional[np.ndarray] = None
        self._index: Any = None
        self._backend = self._resolve_backend(backend)
        self._track_ids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(
        self,
        track_id: int,
        embedding: np.ndarray,
        timestamp: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a new embedding for ``track_id`` observed at ``timestamp``."""

        vec = self._normalize(embedding)
        if vec is None:
            return

        if self.dim is None:
            self.dim = int(vec.shape[0])
        elif int(vec.shape[0]) != int(self.dim):
            logger.debug(
                "[global-reid-bank] rejecting vec due to dim mismatch: %s != %s",
                vec.shape[0],
                self.dim,
            )
            return

        self._prune(timestamp)

        if self.per_track_max is not None:
            self._truncate_track(track_id, keep=self.per_track_max - 1)

        rec_meta: Optional[Dict[str, Any]]
        if meta is None:
            rec_meta = None
        else:
            rec_meta = dict(meta)
        record = _BankRecord(track_id=int(track_id), embedding=vec, last_seen=int(timestamp), meta=rec_meta)
        self._records.append(record)
        self._dirty = True
        self._enforce_max_records()

    def query(
        self,
        embedding: Optional[np.ndarray],
        timestamp: Optional[int] = None,
        *,
        top_k: int = 5,
        sim_threshold: float = 0.75,
        exclude_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Return candidate track IDs sorted by cosine similarity."""

        if embedding is None:
            return []

        vec = self._normalize(embedding)
        if vec is None:
            return []

        if self.dim is None:
            self.dim = int(vec.shape[0])
        elif int(vec.shape[0]) != int(self.dim):
            return []

        if timestamp is not None:
            self._prune(timestamp)

        self._rebuild_index()

        if not self._records:
            return []

        k = int(max(1, min(top_k, len(self._records))))

        sims: Sequence[float]
        idxs: Sequence[int]

        if self._backend == "faiss" and self._index is not None:
            scores, indices = self._index.search(vec.reshape(1, -1), k)  # type: ignore[attr-defined]
            sims = scores[0]
            idxs = indices[0]
        elif self._backend == "sklearn" and self._index is not None:
            dists, indices = self._index.query(vec.reshape(1, -1), k=k, return_distance=True)  # type: ignore[attr-defined]
            d = np.atleast_2d(dists)[0]
            # embeddings are L2-normalised -> cosine = 1 - 0.5 * ||u - v||^2
            sims = 1.0 - 0.5 * np.square(d)
            idxs = np.atleast_2d(indices)[0]
        elif self._backend == "scipy" and self._index is not None:
            dists, indices = self._index.query(vec.reshape(1, -1), k=k)  # type: ignore[attr-defined]
            d = np.atleast_2d(dists)[0]
            # embeddings are L2-normalised -> cosine = 1 - 0.5 * ||u - v||^2
            sims = 1.0 - 0.5 * np.square(d)
            idxs = np.atleast_2d(indices)[0]
        else:
            if self._matrix is None:
                self._matrix = np.stack([rec.embedding for rec in self._records])
            sims_full = self._matrix @ vec
            order = np.argsort(-sims_full)[:k]
            sims = sims_full[order]
            idxs = order

        candidates: Dict[int, Tuple[float, _BankRecord]] = {}
        ex_ids = exclude_ids or set()
        for idx, sim in zip(idxs, sims):
            if idx < 0 or idx >= len(self._records):
                continue
            rec = self._records[idx]
            tid = rec.track_id
            if tid in ex_ids:
                continue
            simf = float(sim)
            if not np.isfinite(simf) or simf < sim_threshold:
                continue
            best = candidates.get(tid)
            if (best is None) or (simf > best[0]):
                candidates[tid] = (simf, rec)

        out: List[Tuple[int, float, Dict[str, Any]]] = []
        for tid, (simf, rec) in candidates.items():
            meta = dict(rec.meta) if isinstance(rec.meta, dict) else {}
            meta.setdefault("last_seen", rec.last_seen)
            out.append((tid, simf, meta))

        out.sort(key=lambda item: item[1], reverse=True)
        return out

    def prune(self, timestamp: int) -> None:
        """Public helper to trigger TTL/LRU pruning."""

        self._prune(timestamp)

    def clear(self) -> None:
        self._records.clear()
        self._matrix = None
        self._index = None
        self._track_ids = None
        self._dirty = True

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    @property
    def backend(self) -> str:  # pragma: no cover - trivial
        return self._backend

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize(self, vec: np.ndarray) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        n = float(np.linalg.norm(arr))
        if not np.isfinite(n) or n < 1e-6:
            return None
        return arr / n

    def _resolve_backend(self, backend: str) -> str:
        backend_norm = (backend or "auto").strip().lower()
        if backend_norm == "auto":
            if _HAS_FAISS:
                return "faiss"
            if _HAS_SKLEARN_KDTREE:
                return "sklearn"
            if _HAS_SCIPY_KDTREE:
                return "scipy"
            return "brute"

        if backend_norm == "faiss" and not _HAS_FAISS:
            logger.warning("[global-reid-bank] FAISS backend requested but not available; falling back to auto")
            return self._resolve_backend("auto")
        if backend_norm == "sklearn" and not _HAS_SKLEARN_KDTREE:
            logger.warning("[global-reid-bank] sklearn KD-tree requested but not available; falling back to auto")
            return self._resolve_backend("auto")
        if backend_norm == "scipy" and not _HAS_SCIPY_KDTREE:
            logger.warning("[global-reid-bank] scipy KD-tree requested but not available; falling back to auto")
            return self._resolve_backend("auto")
        if backend_norm not in {"faiss", "sklearn", "scipy", "brute"}:  # pragma: no cover - guard
            logger.warning("[global-reid-bank] unknown backend '%s', using auto", backend_norm)
            return self._resolve_backend("auto")
        return backend_norm

    def _truncate_track(self, track_id: int, *, keep: int) -> None:
        if keep < 0:
            self._records = [rec for rec in self._records if rec.track_id != track_id]
            self._dirty = True
            return

        existing = [rec for rec in self._records if rec.track_id == track_id]
        if not existing:
            return
        existing.sort(key=lambda rec: rec.last_seen, reverse=True)
        keep_records = existing[:keep]
        keep_ids = {id(rec) for rec in keep_records}
        new_records: List[_BankRecord] = []
        for rec in self._records:
            if rec.track_id != track_id:
                new_records.append(rec)
            elif id(rec) in keep_ids:
                new_records.append(rec)
        if len(new_records) != len(self._records):
            self._records = new_records
            self._dirty = True

    def _enforce_max_records(self) -> None:
        if self.max_records is None or self.max_records <= 0:
            return
        if len(self._records) <= self.max_records:
            return
        self._records.sort(key=lambda rec: rec.last_seen, reverse=True)
        del self._records[self.max_records :]
        self._dirty = True

    def _prune(self, timestamp: int) -> None:
        if timestamp is None:
            return
        changed = False
        if self.ttl_frames is not None:
            keep: List[_BankRecord] = []
            for rec in self._records:
                if (timestamp - rec.last_seen) <= self.ttl_frames:
                    keep.append(rec)
            if len(keep) != len(self._records):
                self._records = keep
                changed = True
        if changed:
            self._dirty = True

    def _rebuild_index(self) -> None:
        if not self._dirty:
            return
        if not self._records:
            self._matrix = None
            self._index = None
            self._track_ids = None
            self._dirty = False
            return

        matrix = np.stack([rec.embedding for rec in self._records]).astype(np.float32)
        self._matrix = matrix
        self._track_ids = np.array([rec.track_id for rec in self._records], dtype=np.int32)

        if self._backend == "faiss" and _HAS_FAISS:
            index = faiss.IndexFlatIP(matrix.shape[1])  # type: ignore[attr-defined]
            index.add(matrix)  # type: ignore[attr-defined]
            self._index = index
        elif self._backend == "sklearn" and _HAS_SKLEARN_KDTREE:
            self._index = _SklearnKDTree(matrix, metric="euclidean")  # type: ignore[call-arg]
        elif self._backend == "scipy" and _HAS_SCIPY_KDTREE:
            self._index = _ScipyKDTree(matrix)
        else:
            self._index = None

        self._dirty = False

