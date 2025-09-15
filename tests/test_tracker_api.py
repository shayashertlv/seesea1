import pytest

pytest.importorskip("cv2")

from tracker import run_pipeline_notebook, run_tracking_with_supervision


def test_tracker_exports() -> None:
    assert callable(run_pipeline_notebook)
    assert callable(run_tracking_with_supervision)
