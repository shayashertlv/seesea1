"""Tests for the transformer association runtime wrapper."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from transformer.models.association import (
    TransformerAssociation,
    _AssociationBackbone,
)


def _write_checkpoint(tmp_path: Path, embed_dim: int) -> Path:
    track_dim = TransformerAssociation._TRACK_STATIC_FEATURE_DIM + embed_dim
    det_dim = TransformerAssociation._DET_STATIC_FEATURE_DIM + embed_dim
    model = _AssociationBackbone(track_dim, det_dim, embed_dim=32, nheads=2, nlayers=1)
    path = tmp_path / "assoc.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "track_dim": track_dim,
                "det_dim": det_dim,
                "hidden_dim": 32,
                "heads": 2,
                "layers": 1,
            },
        },
        path,
    )
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
