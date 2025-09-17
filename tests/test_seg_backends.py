import importlib
import sys
import types

import numpy as np
import pytest


def _make_dummy_masks(frame_bgr, boxes):
    h, w = frame_bgr.shape[:2]
    masks = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        mask = np.zeros((h, w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        masks.append(mask)
    return masks


class _DummyPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, frame_bgr, boxes):
        self.calls += 1
        return _make_dummy_masks(frame_bgr, boxes)


class _FailingPredictor:
    def __init__(self, exc=RuntimeError("fail")):
        self.exc = exc
        self.calls = 0

    def predict(self, frame_bgr, boxes):
        self.calls += 1
        raise self.exc


@pytest.fixture(autouse=True)
def _cleanup_modules():
    managed = {
        name: sys.modules.get(name)
        for name in [
            "sam2",
            "segment_anything",
            "detectron2",
            "detectron2.projects.mask2former",
            "detectron2.engine",
            "detectron2.engine.defaults",
            "seg_sam2",
            "seg_mask2former",
        ]
    }
    yield
    for name, module in managed.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _reload_module(name):
    module = importlib.import_module(name)
    return importlib.reload(module)


def test_seg_sam2_uses_cached_predictor(monkeypatch):
    dummy = types.SimpleNamespace()
    predictor = _DummyPredictor()
    build_calls = {"count": 0}

    def builder():
        build_calls["count"] += 1
        return predictor

    dummy.build_sam2_predictor = builder
    monkeypatch.setitem(sys.modules, "sam2", dummy)
    seg_sam2 = _reload_module("seg_sam2")
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    boxes = [(1, 1, 5, 5), (2, 3, 9, 10)]

    masks, boxes_out, vis = seg_sam2.infer_roi_masks(frame, boxes)
    assert build_calls["count"] == 1
    assert predictor.calls == 1
    assert [tuple(map(float, b)) for b in boxes] == boxes_out
    assert all(mask is not None and mask.mean() == pytest.approx(1.0) for mask in masks)
    assert all(v == pytest.approx(1.0) for v in vis)

    predictor.calls = 0
    masks2, boxes_out2, vis2 = seg_sam2.infer_roi_masks(frame, boxes)
    assert build_calls["count"] == 1  # cached predictor reused
    assert predictor.calls == 1
    assert boxes_out2 == boxes_out
    assert vis2 == vis
    assert all(np.array_equal(m1, m2) for m1, m2 in zip(masks, masks2))


def test_seg_sam2_falls_back_when_predictor_errors(monkeypatch):
    dummy = types.SimpleNamespace()
    predictor = _FailingPredictor()
    build_calls = {"count": 0}

    def builder():
        build_calls["count"] += 1
        return predictor

    dummy.build_sam2_predictor = builder
    monkeypatch.setitem(sys.modules, "sam2", dummy)
    seg_sam2 = _reload_module("seg_sam2")

    fallback_mask = np.ones((4, 4), dtype=np.uint8)
    fallback_calls = {"count": 0}

    def fake_grabcut(frame, box):
        fallback_calls["count"] += 1
        return fallback_mask.copy(), 0.5

    monkeypatch.setattr(seg_sam2, "_grabcut_roi", fake_grabcut)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    boxes = [(1, 1, 6, 7)]

    masks, boxes_out, vis = seg_sam2.infer_roi_masks(frame, boxes)
    assert build_calls["count"] == 1
    assert predictor.calls == 1
    assert fallback_calls["count"] == 1
    assert masks[0].shape == fallback_mask.shape
    assert boxes_out[0] == tuple(map(float, boxes[0]))
    assert vis[0] == pytest.approx(0.5)

    # Subsequent call should skip heavy predictor entirely and still use fallback
    predictor.calls = 0
    masks2, boxes_out2, vis2 = seg_sam2.infer_roi_masks(frame, boxes)
    assert build_calls["count"] == 1
    assert predictor.calls == 0
    assert fallback_calls["count"] == 2
    assert boxes_out2 == boxes_out
    assert vis2 == vis
    assert all(np.array_equal(m1, m2) for m1, m2 in zip(masks, masks2))


def test_mask2former_uses_cached_predictor(monkeypatch):
    dummy = types.SimpleNamespace()
    predictor = _DummyPredictor()
    build_calls = {"count": 0}

    def builder():
        build_calls["count"] += 1
        return predictor

    dummy.build_mask2former_predictor = builder
    monkeypatch.setitem(sys.modules, "detectron2", dummy)
    seg_mask2former = _reload_module("seg_mask2former")

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    boxes = [(0, 0, 4, 4), (5, 2, 9, 9)]

    masks, boxes_out, vis = seg_mask2former.infer_roi_masks(frame, boxes)
    assert build_calls["count"] == 1
    assert predictor.calls == 1
    assert boxes_out == [tuple(map(float, b)) for b in boxes]
    assert all(mask.mean() == pytest.approx(1.0) for mask in masks)
    assert all(v == pytest.approx(1.0) for v in vis)

    predictor.calls = 0
    seg_mask2former.infer_roi_masks(frame, boxes)
    assert build_calls["count"] == 1
    assert predictor.calls == 1


def test_mask2former_falls_back_when_predictor_errors(monkeypatch):
    dummy = types.SimpleNamespace()
    predictor = _FailingPredictor()
    build_calls = {"count": 0}

    def builder():
        build_calls["count"] += 1
        return predictor

    dummy.build_mask2former_predictor = builder
    monkeypatch.setitem(sys.modules, "detectron2", dummy)
    seg_mask2former = _reload_module("seg_mask2former")

    fallback_mask = np.ones((5, 5), dtype=np.uint8)
    fallback_calls = {"count": 0}

    def fake_grabcut(frame, box):
        fallback_calls["count"] += 1
        return fallback_mask.copy(), 0.2

    monkeypatch.setattr(seg_mask2former, "_grabcut_roi", fake_grabcut)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    boxes = [(2, 2, 8, 9)]

    masks, boxes_out, vis = seg_mask2former.infer_roi_masks(frame, boxes)
    assert build_calls["count"] == 1
    assert predictor.calls == 1
    assert fallback_calls["count"] == 1
    assert boxes_out[0] == tuple(map(float, boxes[0]))
    assert vis[0] == pytest.approx(0.2)

    predictor.calls = 0
    seg_mask2former.infer_roi_masks(frame, boxes)
    assert build_calls["count"] == 1
    assert predictor.calls == 0
    assert fallback_calls["count"] == 2
