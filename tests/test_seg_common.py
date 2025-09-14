import numpy as np

from seg_common import grabcut_roi_masks


def test_grabcut_roi_masks_basic():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [
        (10, 10, 90, 90),  # valid box
        (0, 0, 0, 10),     # invalid width
    ]
    masks, boxes_out, vis = grabcut_roi_masks(frame, boxes)

    assert masks[0] is not None
    assert masks[0].shape == (80, 80)
    # mask values should be 0 or 1
    assert set(np.unique(masks[0])).issubset({0, 1})
    assert boxes_out[0] == (10, 10, 90, 90)
    assert 0.0 <= vis[0] <= 1.0

    assert masks[1] is None
    assert boxes_out[1] is None
    assert vis[1] == 0.0
