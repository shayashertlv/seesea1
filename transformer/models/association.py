"""Transformer-based association model for surfer tracking."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional at runtime
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False

from ..utils import normalize_bbox, stack_embeddings


@dataclass
class AssociationFeatures:
    track_features: np.ndarray
    det_features: np.ndarray
    track_embeddings: np.ndarray
    det_embeddings: np.ndarray


class _AssociationBackbone(nn.Module):
    def __init__(self, track_dim: int, det_dim: int, embed_dim: int = 256, nheads: int = 4, nlayers: int = 2) -> None:
        super().__init__()
        self.track_proj = nn.Linear(track_dim, embed_dim)
        self.det_proj = nn.Linear(det_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nheads, batch_first=True)
        self.track_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.det_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, track_feats: torch.Tensor, det_feats: torch.Tensor) -> torch.Tensor:
        t = self.track_encoder(self.track_proj(track_feats))
        d = self.det_encoder(self.det_proj(det_feats))
        t_expand = t.unsqueeze(2).expand(-1, -1, d.size(1), -1)
        d_expand = d.unsqueeze(1).expand(-1, t.size(1), -1, -1)
        pair = torch.cat([t_expand, d_expand], dim=-1)
        scores = self.score_mlp(pair).squeeze(-1)
        return scores


class TransformerAssociation:
    """Wrapper that loads weights and produces association cost matrices."""

    _TRACK_STATIC_FEATURE_DIM = 10
    _DET_STATIC_FEATURE_DIM = 8

    def __init__(
        self,
        device: Optional[str] = None,
        max_embed_dim: Optional[int] = None,
        hidden_dim: int = 256,
        nheads: int = 4,
        nlayers: int = 2,
        weight: float = 0.5,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for TransformerAssociation")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._configured_embed_dim = int(max_embed_dim) if max_embed_dim is not None else None
        self.max_embed_dim: Optional[int] = self._configured_embed_dim
        self.hidden_dim = int(hidden_dim)
        self.nheads = int(nheads)
        self.nlayers = int(nlayers)
        self.weight = float(weight)
        self._model: Optional[_AssociationBackbone] = None
        self._meta: Dict[str, int] = {}
        self._ready = False

    @classmethod
    def from_env(cls) -> "TransformerAssociation":
        weight = float(os.getenv("TRANSFORMER_ASSOC_WEIGHT", "0.5"))
        device = os.getenv("TRANSFORMER_ASSOC_DEVICE", "auto")
        max_embed_env = os.getenv("TRANSFORMER_ASSOC_EMBED_DIM", "").strip()
        max_embed_dim = int(max_embed_env) if max_embed_env else None
        hidden_dim = int(os.getenv("TRANSFORMER_ASSOC_HIDDEN", "256"))
        nheads = int(os.getenv("TRANSFORMER_ASSOC_HEADS", "4"))
        nlayers = int(os.getenv("TRANSFORMER_ASSOC_LAYERS", "2"))
        if device == "auto":
            device = None
        return cls(device=device, max_embed_dim=max_embed_dim, hidden_dim=hidden_dim, nheads=nheads, nlayers=nlayers, weight=weight)

    @property
    def ready(self) -> bool:
        return bool(self._ready)

    def load(self, path: Path) -> None:
        if not HAS_TORCH:
            return
        state = torch.load(path, map_location="cpu")
        meta = state.get("meta", {})
        track_dim = int(meta.get("track_dim", 0))
        det_dim = int(meta.get("det_dim", 0))
        if track_dim <= 0 or det_dim <= 0:
            raise ValueError("Invalid metadata in transformer association weights")
        expected_embed = self._infer_embedding_dim(track_dim, det_dim)
        if (
            self._configured_embed_dim is not None
            and expected_embed != self._configured_embed_dim
        ):
            raise ValueError(
                "TransformerAssociation configured for"
                f" {self._configured_embed_dim}-dim embeddings, but checkpoint"
                f" expects {expected_embed}. Update TRANSFORMER_ASSOC_EMBED_DIM"
                " to match the logger cap or retrain the model."
            )
        self.max_embed_dim = expected_embed if expected_embed > 0 else None
        self._ensure_model(track_dim, det_dim)
        self._model.load_state_dict(state["state_dict"])  # type: ignore[index]
        self._model.to(self.device)
        self._meta = {
            "track_dim": track_dim,
            "det_dim": det_dim,
            "embed_dim": expected_embed,
        }
        self._ready = True

    @classmethod
    def _infer_embedding_dim(cls, track_dim: int, det_dim: int) -> int:
        track_embed = track_dim - cls._TRACK_STATIC_FEATURE_DIM
        det_embed = det_dim - cls._DET_STATIC_FEATURE_DIM
        if track_embed < 0 or det_embed < 0:
            raise ValueError("Checkpoint metadata reports fewer static features than expected")
        if track_embed != det_embed:
            raise ValueError(
                "Checkpoint metadata reports different embedding widths for tracks and detections"
            )
        return track_embed

    def _ensure_model(self, track_dim: int, det_dim: int) -> None:
        if self._model is None:
            self._model = _AssociationBackbone(track_dim, det_dim, embed_dim=self.hidden_dim, nheads=self.nheads, nlayers=self.nlayers)
        else:
            if self._model.track_proj.in_features != track_dim or self._model.det_proj.in_features != det_dim:
                self._model = _AssociationBackbone(track_dim, det_dim, embed_dim=self.hidden_dim, nheads=self.nheads, nlayers=self.nlayers)
        self._model.to(self.device)
        self._model.eval()

    def compute_cost(
        self,
        track_pack: Sequence[Dict],
        det_pack: Sequence[Dict],
        meta: Dict[str, float],
    ) -> Optional[np.ndarray]:
        if not HAS_TORCH:
            return None
        weight_path = os.getenv("TRANSFORMER_ASSOC_WEIGHTS", "").strip()
        if weight_path and not self._ready:
            self.load(Path(weight_path))
        if not self.ready or not track_pack or not det_pack:
            return None
        feats = self._build_features(track_pack, det_pack, meta)
        if feats.track_features.size == 0 or feats.det_features.size == 0:
            return None
        track_tensor = torch.from_numpy(feats.track_features).to(self.device).unsqueeze(0)
        det_tensor = torch.from_numpy(feats.det_features).to(self.device).unsqueeze(0)
        with torch.no_grad():
            scores = self._model(track_tensor, det_tensor)
            logits = scores.squeeze(0).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        cost = 1.0 - probs
        return cost.astype(np.float32)

    # ------------------------------------------------------------------
    def _build_features(
        self,
        tracks: Sequence[Dict],
        dets: Sequence[Dict],
        meta: Dict[str, float],
    ) -> AssociationFeatures:
        W = float(meta.get("width", 1920.0))
        H = float(meta.get("height", 1080.0))
        track_rows: List[List[float]] = []
        det_rows: List[List[float]] = []
        track_embeds_raw: List[Optional[np.ndarray]] = []
        det_embeds_raw: List[Optional[np.ndarray]] = []
        for track in tracks:
            box = track.get("pred_box", track.get("last_bbox", (0, 0, 1, 1)))
            norm = normalize_bbox(box, W, H)
            vel = track.get("velocity", (0.0, 0.0))
            age = float(track.get("age", 0))
            tsu = float(track.get("time_since_update", 0))
            conf = float(track.get("confidence", 0.0))
            vis = float(track.get("visibility", 1.0))
            row = [
                norm[0],
                norm[1],
                norm[2],
                norm[3],
                float(vel[0]) / max(W, 1.0),
                float(vel[1]) / max(H, 1.0),
                age / 100.0,
                tsu / 50.0,
                conf,
                vis,
            ]
            track_rows.append(row)
            track_embeds_raw.append(track.get("embedding"))
        for det in dets:
            box = det.get("box", (0, 0, 1, 1))
            norm = normalize_bbox(box, W, H)
            conf = float(det.get("confidence", 0.0))
            vis = float(det.get("visibility", 1.0))
            speed = det.get("motion", (0.0, 0.0))
            row = [
                norm[0],
                norm[1],
                norm[2],
                norm[3],
                conf,
                vis,
                float(speed[0]) / max(W, 1.0),
                float(speed[1]) / max(H, 1.0),
            ]
            det_rows.append(row)
            det_embeds_raw.append(det.get("embedding"))
        track_feats = np.asarray(track_rows, dtype=np.float32)
        det_feats = np.asarray(det_rows, dtype=np.float32)
        track_embeds, _ = stack_embeddings(track_embeds_raw, self.max_embed_dim)
        det_embeds, _ = stack_embeddings(det_embeds_raw, self.max_embed_dim)
        track_feats = np.concatenate([track_feats, track_embeds], axis=1) if track_embeds.size else track_feats
        det_feats = np.concatenate([det_feats, det_embeds], axis=1) if det_embeds.size else det_feats
        if self._model is None:
            self._ensure_model(track_feats.shape[1], det_feats.shape[1])
        return AssociationFeatures(track_feats, det_feats, track_embeds, det_embeds)
