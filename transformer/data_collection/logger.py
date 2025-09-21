"""Association sample logger used for transformer training."""
from __future__ import annotations

import atexit
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from ..utils import pack_features, stack_embeddings


@dataclass
class AssociationSample:
    frame_idx: int
    track_ids: Sequence[int]
    detection_count: int
    track_features: np.ndarray
    det_features: np.ndarray
    cost_matrix: np.ndarray
    mask_matrix: np.ndarray
    assigned_track_ids: Sequence[int]
    track_embeddings: np.ndarray
    det_embeddings: np.ndarray
    metadata: Dict[str, float] = field(default_factory=dict)


class AssociationLogger:
    """Writes association training samples to disk as compressed NPZ files."""

    def __init__(self, root: Path, max_embed_dim: Optional[int] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_embed_dim = max_embed_dim
        self._lock = threading.Lock()
        self._manifest_path = self.root / "manifest.jsonl"
        self._used_indices, self._counter = self._resume_state()
        self._manifest_fp = open(self._manifest_path, "a", encoding="utf-8")
        atexit.register(self.close)

    def _resume_state(self) -> tuple[set[int], int]:
        """Return previously used indices and the next index to be written."""

        indices: set[int] = set()
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as manifest:
                    for line in manifest:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        idx = self._extract_index(entry.get("npz"))
                        if idx is not None:
                            indices.add(idx)
            except OSError:
                # If the manifest cannot be read we simply fall back to scanning files.
                pass
        try:
            for path in self.root.glob("sample_*.npz"):
                idx = self._extract_index(path.name)
                if idx is not None:
                    indices.add(idx)
        except OSError:
            # If the directory cannot be read we stick with the default counter.
            pass
        if not indices:
            return indices, 0
        return indices, max(indices) + 1

    @staticmethod
    def _extract_index(filename: Optional[str]) -> Optional[int]:
        if not filename:
            return None
        match = re.fullmatch(r"sample_(\d{7})\.npz", str(filename))
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def close(self) -> None:
        try:
            if self._manifest_fp and not self._manifest_fp.closed:
                self._manifest_fp.close()
        except Exception:
            pass

    def log(
        self,
        frame_idx: int,
        track_ids: Sequence[int],
        det_boxes: Sequence[Sequence[float]],
        track_features: Iterable[Sequence[float]],
        det_features: Iterable[Sequence[float]],
        cost_matrix: np.ndarray,
        mask_matrix: np.ndarray,
        assigned_track_ids: Sequence[int],
        track_embeddings: Sequence[Optional[np.ndarray]],
        det_embeddings: Sequence[Optional[np.ndarray]],
        metadata: Optional[Dict[str, float]] = None,
    ) -> None:
        tracks = list(track_ids)
        assigned = [int(x) for x in assigned_track_ids]
        cost = np.asarray(cost_matrix, dtype=np.float32)
        mask = np.asarray(mask_matrix, dtype=np.float32)
        track_rows = [tuple(row) for row in track_features]
        det_rows = [tuple(row) for row in det_features]
        tf = pack_features(track_rows, len(track_rows[0]) if track_rows else 0)
        df = pack_features(det_rows, len(det_rows[0]) if det_rows else 0)
        t_emb, _ = stack_embeddings(track_embeddings, self.max_embed_dim)
        d_emb, _ = stack_embeddings(det_embeddings, self.max_embed_dim)
        sample = AssociationSample(
            frame_idx=frame_idx,
            track_ids=tracks,
            detection_count=len(det_boxes),
            track_features=tf,
            det_features=df,
            cost_matrix=cost,
            mask_matrix=mask,
            assigned_track_ids=assigned,
            track_embeddings=t_emb,
            det_embeddings=d_emb,
            metadata=metadata or {},
        )
        self._write_sample(sample)

    def _write_sample(self, sample: AssociationSample) -> None:
        with self._lock:
            idx = self._counter
            sample_path = self.root / f"sample_{idx:07d}.npz"
            while idx in self._used_indices or sample_path.exists():
                idx += 1
                sample_path = self.root / f"sample_{idx:07d}.npz"
            self._counter = idx + 1
            self._used_indices.add(idx)
            np.savez_compressed(
                sample_path,
                frame_idx=sample.frame_idx,
                track_ids=np.asarray(sample.track_ids, dtype=np.int64),
                detection_count=np.asarray([sample.detection_count], dtype=np.int64),
                track_features=sample.track_features,
                det_features=sample.det_features,
                cost_matrix=sample.cost_matrix,
                mask_matrix=sample.mask_matrix,
                assigned_track_ids=np.asarray(sample.assigned_track_ids, dtype=np.int64),
                track_embeddings=sample.track_embeddings,
                det_embeddings=sample.det_embeddings,
                metadata=json.dumps(sample.metadata),
            )
            self._manifest_fp.write(json.dumps({"npz": sample_path.name}) + "\n")
            self._manifest_fp.flush()


_LOGGER: Optional[AssociationLogger] = None


def get_association_logger() -> Optional[AssociationLogger]:
    """Return a singleton logger if ``TRANSFORMER_LOG_DIR`` is set."""
    global _LOGGER
    log_dir = os.getenv("TRANSFORMER_LOG_DIR", "").strip()
    if not log_dir:
        return None
    if _LOGGER is not None:
        return _LOGGER
    root = Path(log_dir)
    max_embed_dim_env = os.getenv("TRANSFORMER_LOG_EMBED_DIM", "").strip()
    max_embed_dim = int(max_embed_dim_env) if max_embed_dim_env else None
    _LOGGER = AssociationLogger(root, max_embed_dim=max_embed_dim)
    return _LOGGER
