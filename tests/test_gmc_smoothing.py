import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gmc import _smooth_H, reset_gmc_smoothing, GMCFilter


def _translation(tx: float, ty: float) -> np.ndarray:
    """Utility to create a simple translation homography."""
    return np.array([[1.0, 0.0, tx],
                     [0.0, 1.0, ty],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def test_reset_gmc_smoothing_independent_sequences():
    """Resetting should prevent state leakage between sequences."""
    reset_gmc_smoothing()
    h1 = _translation(10, 0)
    _smooth_H(h1)
    h2 = _translation(20, 0)
    # Without reset, smoothing is influenced by h1
    influenced = _smooth_H(h2)
    assert not np.allclose(influenced, h2)
    # After reset, smoothing starts fresh
    reset_gmc_smoothing()
    fresh = _smooth_H(h2)
    assert np.allclose(fresh, h2)


def test_gmcfilter_independent_instances():
    """Separate GMCFilter instances maintain independent state."""
    f1 = GMCFilter()
    f2 = GMCFilter()
    h1 = _translation(5, 0)
    h2 = _translation(-5, 0)
    # First call sets their internal state
    f1.smooth(h1)
    f2.smooth(h2)
    # Second call to each should be influenced only by its own history
    out1 = f1.smooth(h1)
    out2 = f2.smooth(h2)
    assert not np.allclose(out1, out2)
    # If we reset one filter, it should revert to returning the raw input
    f1.reset()
    reset_out = f1.smooth(h1)
    assert np.allclose(reset_out, h1)
