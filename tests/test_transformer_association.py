"""Tests for the transformer association runtime wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transformer.models.association import (
    TransformerAssociation,
    _AssociationBackbone,
)


def _write_checkpoint(tmp_path: Path, embed_dim: int, *, include_meta: bool = True) -> Path:
    track_dim = TransformerAssociation._TRACK_STATIC_FEATURE_DIM + embed_dim
    det_dim = TransformerAssociation._DET_STATIC_FEATURE_DIM + embed_dim
    model = _AssociationBackbone(track_dim, det_dim, embed_dim=32, nheads=2, nlayers=1)
    path = tmp_path / "assoc.pt"
    meta: Dict[str, int] = {}
    if include_meta:
        meta = {
            "track_dim": track_dim,
            "det_dim": det_dim,
            "hidden_dim": 32,
            "heads": 2,
            "layers": 1,
        }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    return path


def test_transformer_association_embed_dim_mismatch(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSFORMER_ASSOC_EMBED_DIM", raising=False)
    ckpt = _write_checkpoint(tmp_path, embed_dim=64)
    assoc = TransformerAssociation(max_embed_dim=16, hidden_dim=32, nheads=2, nlayers=1, device="cpu")
    with pytest.raises(ValueError, match="embeddings"):
        assoc.load(ckpt)


def test_transformer_association_derives_embed_dim(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSFORMER_ASSOC_EMBED_DIM", raising=False)
    ckpt = _write_checkpoint(tmp_path, embed_dim=48)
    assoc = TransformerAssociation(hidden_dim=32, nheads=2, nlayers=1, device="cpu")
    assoc.load(ckpt)
    assert assoc.max_embed_dim == 48


def test_transformer_association_infers_dim_from_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSFORMER_ASSOC_EMBED_DIM", raising=False)
    ckpt = _write_checkpoint(tmp_path, embed_dim=32, include_meta=False)
    assoc = TransformerAssociation(hidden_dim=32, nheads=2, nlayers=1, device="cpu")
    assoc.load(ckpt)
    assert assoc.max_embed_dim == 32


def test_transformer_association_runtime_width_mismatch(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSFORMER_ASSOC_EMBED_DIM", raising=False)
    ckpt = _write_checkpoint(tmp_path, embed_dim=64)
    assoc = TransformerAssociation(hidden_dim=32, nheads=2, nlayers=1, device="cpu")
    assoc.load(ckpt)
    tracks = [
        {
            "pred_box": (0.0, 0.0, 10.0, 10.0),
            "velocity": (0.0, 0.0),
            "age": 5,
            "time_since_update": 1,
            "confidence": 0.9,
            "visibility": 1.0,
            "embedding": np.ones(32, dtype=np.float32),
        }
    ]
    dets = [
        {
            "box": (0.0, 0.0, 10.0, 10.0),
            "confidence": 0.9,
            "visibility": 1.0,
            "motion": (0.0, 0.0),
            "embedding": np.ones(32, dtype=np.float32),
        }
    ]
    with pytest.raises(ValueError, match="expects 64-dim"):
        assoc.compute_cost(tracks, dets, {"width": 640.0, "height": 480.0})


def test_transformer_association_runtime_wider_than_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSFORMER_ASSOC_EMBED_DIM", raising=False)
    ckpt = _write_checkpoint(tmp_path, embed_dim=64)
    assoc = TransformerAssociation(hidden_dim=32, nheads=2, nlayers=1, device="cpu")
    assoc.load(ckpt)
    tracks = [
        {
            "pred_box": (0.0, 0.0, 10.0, 10.0),
            "velocity": (0.0, 0.0),
            "age": 5,
            "time_since_update": 1,
            "confidence": 0.9,
            "visibility": 1.0,
            "embedding": np.ones(96, dtype=np.float32),
        }
    ]
    dets = [
        {
            "box": (0.0, 0.0, 10.0, 10.0),
            "confidence": 0.9,
            "visibility": 1.0,
            "motion": (0.0, 0.0),
            "embedding": np.ones(96, dtype=np.float32),
        }
    ]
    with pytest.raises(ValueError, match="runtime provided 96"):
        assoc.compute_cost(tracks, dets, {"width": 640.0, "height": 480.0})


def test_transformer_association_handles_wide_embeddings(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSFORMER_ASSOC_EMBED_DIM", raising=False)
    ckpt = _write_checkpoint(tmp_path, embed_dim=192)
    assoc = TransformerAssociation(hidden_dim=32, nheads=2, nlayers=1, device="cpu")
    assoc.load(ckpt)
    assert assoc.max_embed_dim == 192
    tracks = [
        {
            "pred_box": (0.0, 0.0, 10.0, 10.0),
            "velocity": (0.0, 0.0),
            "age": 5,
            "time_since_update": 1,
            "confidence": 0.9,
            "visibility": 1.0,
            "embedding": np.ones(192, dtype=np.float32),
        }
    ]
    dets = [
        {
            "box": (0.0, 0.0, 10.0, 10.0),
            "confidence": 0.9,
            "visibility": 1.0,
            "motion": (0.0, 0.0),
            "embedding": np.ones(192, dtype=np.float32),
        }
    ]
    cost = assoc.compute_cost(tracks, dets, {"width": 640.0, "height": 480.0})
    assert isinstance(cost, np.ndarray)
    assert cost.shape == (1, 1)
