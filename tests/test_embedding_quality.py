import copy
import pytest

np = pytest.importorskip("numpy")

import tracker


@pytest.mark.parametrize("use_low_quality", [False, True])
def test_embedding_quality_gating_prevents_switch(monkeypatch, use_low_quality) -> None:
    # Configure globals for deterministic association
    monkeypatch.setattr(tracker, "APPEAR_ENABLE", False)
    monkeypatch.setattr(tracker, "APPEAR_EMB_ENABLE", True)
    monkeypatch.setattr(tracker, "SEG_ENABLE", False)
    monkeypatch.setattr(tracker, "APPEAR_ESCAPE_ENABLE", False)
    monkeypatch.setattr(tracker, "ADAPTIVE_WEIGHT", False)
    monkeypatch.setattr(tracker, "ANISO_GATE_ENABLE", False)
    monkeypatch.setattr(tracker, "ASSOC_W_IOU", 0.1)
    monkeypatch.setattr(tracker, "ASSOC_W_EMB", 0.8)
    monkeypatch.setattr(tracker, "ASSOC_W_HIST", 0.1)
    monkeypatch.setattr(tracker, "ASSOC_W_MOT", 0.0)
    monkeypatch.setattr(tracker, "ASSOC_MIN_IOU", 0.0)
    monkeypatch.setattr(tracker, "ASSOC_MIN_SIM", 0.0)
    monkeypatch.setattr(tracker, "ASSOC_MAX_CENTER_DIST", 1.0)
    monkeypatch.setattr(tracker, "GLOBAL_REID_BANK", None)

    # Two tracks with opposing embeddings
    state = {
        1: {
            "last_bbox": (0.0, 0.0, 40.0, 40.0),
            "last_center": (20.0, 20.0),
            "last_vel": (0.0, 0.0),
            "prev_vel": (0.0, 0.0),
            "age": 6,
            "hits": 6,
            "time_since_update": 0,
            "emb": np.array([1.0, 0.0], dtype=np.float32),
            "emb_bank": [np.array([1.0, 0.0], dtype=np.float32)],
        },
        2: {
            "last_bbox": (35.0, 0.0, 75.0, 40.0),
            "last_center": (55.0, 20.0),
            "last_vel": (0.0, 0.0),
            "prev_vel": (0.0, 0.0),
            "age": 6,
            "hits": 6,
            "time_since_update": 0,
            "emb": np.array([0.0, 1.0], dtype=np.float32),
            "emb_bank": [np.array([0.0, 1.0], dtype=np.float32)],
        },
    }

    boxes = [
        (0.0, 0.0, 40.0, 40.0),
        (35.0, 0.0, 75.0, 40.0),
    ]

    det_embs = [
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    ]
    det_hists = [None, None]
    det_confs = [0.9, 0.9]

    if use_low_quality:
        det_quality = [
            {"vis": 0.1, "min_side": 20.0, "blur": 5.0, "conf": 0.9},
            {"vis": 0.9, "min_side": 80.0, "blur": 0.5, "conf": 0.9},
        ]
    else:
        det_quality = [
            {"vis": 0.9, "min_side": 80.0, "blur": 0.2, "conf": 0.9},
            {"vis": 0.9, "min_side": 80.0, "blur": 0.2, "conf": 0.9},
        ]

    assigned_ids, _, _, _, _, _ = tracker.appearance_associate(
        copy.deepcopy(state),
        boxes,
        det_embs,
        det_hists,
        frame_idx=0,
        W=128,
        H=72,
        next_tid=3,
        cost_debug_acc=None,
        H_cam=None,
        fps=30.0,
        det_confs=det_confs,
        tm=None,
        det_masks=None,
        det_mask_boxes=None,
        det_vis=None,
        assoc_gates=None,
        assoc_weights=None,
        det_emb_quality=det_quality,
    )

    if use_low_quality:
        assert assigned_ids == [1, 2]
    else:
        # Without quality gating the cross-similarity triggers the swap
        assert assigned_ids == [2, 1]
