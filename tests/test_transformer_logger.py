"""Tests for the transformer association logger resume behaviour."""

from pathlib import Path
import importlib.util
import sys
import types

import numpy as np


def _get_association_logger() -> type:
    module_name = "transformer.data_collection.logger"
    if module_name in sys.modules:
        module = sys.modules[module_name]
        return module.AssociationLogger  # type: ignore[attr-defined]

    root = Path(__file__).resolve().parents[1]
    transformer_pkg_name = "transformer"
    if transformer_pkg_name not in sys.modules:
        transformer_pkg = types.ModuleType(transformer_pkg_name)
        transformer_pkg.__path__ = [str(root / "transformer")]  # type: ignore[attr-defined]
        sys.modules[transformer_pkg_name] = transformer_pkg
    dc_pkg_name = "transformer.data_collection"
    if dc_pkg_name not in sys.modules:
        dc_pkg = types.ModuleType(dc_pkg_name)
        dc_pkg.__path__ = [str(root / "transformer/data_collection")]  # type: ignore[attr-defined]
        sys.modules[dc_pkg_name] = dc_pkg

    spec = importlib.util.spec_from_file_location(
        module_name, root / "transformer/data_collection/logger.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load AssociationLogger module")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = dc_pkg_name
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.AssociationLogger  # type: ignore[attr-defined]


AssociationLogger = _get_association_logger()


def _log_dummy_sample(logger: AssociationLogger) -> None:
    logger.log(
        frame_idx=0,
        track_ids=[1],
        det_boxes=[[0.0, 0.0, 1.0, 1.0]],
        track_features=[[0.1, 0.2]],
        det_features=[[0.3, 0.4]],
        cost_matrix=np.zeros((1, 1), dtype=np.float32),
        mask_matrix=np.ones((1, 1), dtype=np.float32),
        assigned_track_ids=[1],
        track_embeddings=[np.zeros((2,), dtype=np.float32)],
        det_embeddings=[np.zeros((2,), dtype=np.float32)],
        metadata={"score": 1.0},
    )


def test_logger_resume_creates_unique_filenames(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    logger = AssociationLogger(log_dir)
    _log_dummy_sample(logger)
    _log_dummy_sample(logger)
    logger.close()

    resumed_logger = AssociationLogger(log_dir)
    _log_dummy_sample(resumed_logger)
    _log_dummy_sample(resumed_logger)
    resumed_logger.close()

    npz_files = sorted(p.name for p in log_dir.glob("sample_*.npz"))
    assert npz_files == [
        "sample_0000000.npz",
        "sample_0000001.npz",
        "sample_0000002.npz",
        "sample_0000003.npz",
    ]

    manifest_path = log_dir / "manifest.jsonl"
    lines = [line for line in manifest_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 4


def test_logger_second_run_appends_new_indices(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    first_logger = AssociationLogger(log_dir)
    _log_dummy_sample(first_logger)
    _log_dummy_sample(first_logger)
    first_logger.close()

    resumed_logger = AssociationLogger(log_dir)
    _log_dummy_sample(resumed_logger)
    _log_dummy_sample(resumed_logger)
    resumed_logger.close()

    npz_files = sorted(p.name for p in log_dir.glob("sample_*.npz"))
    assert npz_files == [
        "sample_0000000.npz",
        "sample_0000001.npz",
        "sample_0000002.npz",
        "sample_0000003.npz",
    ]
