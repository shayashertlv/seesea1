"""Tests for transformer association training utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_ = torch  # satisfy linters when torch is only needed for availability

from transformer.training import train_assoc


def _write_sample(path: Path, tracks: int, dets: int) -> None:
    track_features = np.full((tracks, 2), fill_value=float(tracks), dtype=np.float32)
    det_features = np.full((dets, 3), fill_value=float(dets), dtype=np.float32)
    track_embeddings = np.full((tracks, 4), fill_value=0.5, dtype=np.float32)
    det_embeddings = np.full((dets, 4), fill_value=-0.5, dtype=np.float32)
    track_ids = np.arange(tracks, dtype=np.int64)
    assigned = np.full((dets,), fill_value=-1, dtype=np.int64)
    assignable = min(tracks, dets)
    if assignable:
        assigned[:assignable] = track_ids[:assignable]
    np.savez(
        path,
        track_features=track_features,
        det_features=det_features,
        track_embeddings=track_embeddings,
        det_embeddings=det_embeddings,
        track_ids=track_ids,
        assigned_track_ids=assigned,
    )


def _prepare_dataset(root: Path, specs: Sequence[tuple[int, int]] = ((1, 2),)) -> Path:
    data_dir = root / "assoc"
    data_dir.mkdir()
    manifest = data_dir / "manifest.jsonl"
    entries = []
    for idx, (tracks, dets) in enumerate(specs):
        sample_path = data_dir / f"sample_{idx}.npz"
        _write_sample(sample_path, tracks=tracks, dets=dets)
        entries.append({"npz": sample_path.name})
    with manifest.open("w", encoding="utf-8") as fp:
        for entry in entries:
            fp.write(json.dumps(entry) + "\n")
    return data_dir


def test_train_assoc_supports_batch_sizes_above_one(tmp_path) -> None:
    data_dir = _prepare_dataset(tmp_path, specs=[(1, 2), (2, 3), (3, 1)])

    output_path = tmp_path / "weights.pt"
    args = train_assoc.parse_args(
        [
            "--data",
            str(data_dir),
            "--output",
            str(output_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )

    train_assoc.train(args)

    assert output_path.exists()


def test_train_assoc_inspect_mode_reports_dataset(tmp_path, capsys) -> None:
    data_dir = _prepare_dataset(tmp_path, specs=[(2, 3)])

    args = train_assoc.parse_args(
        [
            "--data",
            str(data_dir),
            "--inspect",
        ]
    )

    train_assoc.train(args)

    output = capsys.readouterr().out
    assert "samples: 1" in output
