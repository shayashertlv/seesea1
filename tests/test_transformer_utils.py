"""Unit tests for utility helpers used by the transformer stack."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_stack_embeddings():
    module_name = "transformer.utils"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        root = Path(__file__).resolve().parents[1]
        transformer_pkg_name = "transformer"
        if transformer_pkg_name not in sys.modules:
            transformer_pkg = types.ModuleType(transformer_pkg_name)
            transformer_pkg.__path__ = [str(root / "transformer")]  # type: ignore[attr-defined]
            sys.modules[transformer_pkg_name] = transformer_pkg
        spec = importlib.util.spec_from_file_location(module_name, root / "transformer/utils.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load transformer.utils module")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = transformer_pkg_name
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.stack_embeddings  # type: ignore[attr-defined]


stack_embeddings = _load_stack_embeddings()


def test_stack_embeddings_truncates_and_warns(recwarn: pytest.WarningsRecorder) -> None:
    """Embeddings wider than the cap should be truncated with a warning."""

    vectors = [np.arange(4, dtype=np.float32)]

    stacked, width, truncated = stack_embeddings(vectors, max_dim=2)

    assert width == 2
    assert truncated is True
    np.testing.assert_array_equal(stacked, np.asarray([[0.0, 1.0]], dtype=np.float32))

    warning = recwarn.pop(RuntimeWarning)
    assert "Truncated embeddings" in str(warning.message)
