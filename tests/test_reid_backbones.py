import sys
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

try:
    import torch  # noqa: F401
    import numpy as np  # noqa: F401
except Exception:
    torch = None
    np = None

if torch is None or np is None:
    pytest.skip("torch or numpy not installed", allow_module_level=True)

import reid_backbones
from reid_extractor import ReIDExtractor


def dummy_loader(extractor: ReIDExtractor) -> None:
    extractor.model = "dummy"


def test_registry_contains_defaults():
    assert "osnet" in reid_backbones.BACKBONE_LOADERS


def test_default_loader_invoked(monkeypatch):
    if torch is None:
        pytest.skip("torch not installed")
    monkeypatch.setitem(reid_backbones.BACKBONE_LOADERS, "osnet", dummy_loader)
    ex = ReIDExtractor(backend="osnet", device="cpu")
    assert ex.model == "dummy"
