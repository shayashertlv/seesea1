import importlib
import json
import sys
import types

import numpy as np
import pytest


class _FakeCV2:
    INTER_NEAREST = 0

    @staticmethod
    def resize(arr, size, interpolation=None):
        target_w, target_h = size
        arr_np = np.asarray(arr)
        if arr_np.ndim == 0:
            return arr_np
        src_h, src_w = arr_np.shape[:2]
        if src_h == 0 or src_w == 0:
            return arr_np
        row_idx = np.clip(
            np.round(np.linspace(0, src_h - 1, target_h)).astype(int),
            0,
            src_h - 1,
        )
        col_idx = np.clip(
            np.round(np.linspace(0, src_w - 1, target_w)).astype(int),
            0,
            src_w - 1,
        )
        resized = arr_np[row_idx][:, col_idx]
        return resized.astype(arr_np.dtype, copy=False)


sys.modules.setdefault("cv2", _FakeCV2())


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
            "cv2",
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


def test_seg_sam2_factory_from_env(monkeypatch):
    predictor = _DummyPredictor()
    calls = {}

    def factory(**kwargs):
        calls.update(kwargs)
        return predictor

    module = types.ModuleType("custom_sam_factory")
    module.make_predictor = factory
    monkeypatch.setitem(sys.modules, "custom_sam_factory", module)
    monkeypatch.setenv("SEG_SAM2_FACTORY", "custom_sam_factory:make_predictor")
    monkeypatch.setenv("SEG_SAM2_ARGS", json.dumps({"foo": 1, "bar": "baz"}))

    seg_sam2 = _reload_module("seg_sam2")
    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    boxes = [(0, 0, 5, 5)]

    masks, boxes_out, vis = seg_sam2.infer_roi_masks(frame, boxes)
    assert calls == {"foo": 1, "bar": "baz"}
    assert predictor.calls == 1
    assert masks[0] is not None and masks[0].mean() == pytest.approx(1.0)
    assert boxes_out == [tuple(map(float, boxes[0]))]
    assert vis == [pytest.approx(1.0)]


def test_seg_sam2_builder_receives_env_arguments(monkeypatch):
    dummy = types.SimpleNamespace()
    predictor = _DummyPredictor()
    builder_args = {}

    def builder(*, checkpoint, model_type):
        builder_args["checkpoint"] = checkpoint
        builder_args["model_type"] = model_type
        return predictor

    dummy.build_sam2_predictor = builder
    monkeypatch.setitem(sys.modules, "sam2", dummy)
    monkeypatch.setenv("SAM2_CHECKPOINT", "sam2/path.pt")
    monkeypatch.setenv("SAM_MODEL_TYPE", "vit_h")

    seg_sam2 = _reload_module("seg_sam2")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    boxes = [(1, 1, 4, 5)]

    masks, boxes_out, vis = seg_sam2.infer_roi_masks(frame, boxes)
    assert builder_args == {"checkpoint": "sam2/path.pt", "model_type": "vit_h"}
    assert masks[0] is not None and masks[0].mean() == pytest.approx(1.0)
    assert boxes_out == [tuple(map(float, boxes[0]))]
    assert vis == [pytest.approx(1.0)]


def test_seg_sam2_env_arguments_when_builder_fails(monkeypatch):
    dummy = types.SimpleNamespace()
    builder_args = {"count": 0}

    def builder(*, checkpoint, model_type=None):
        builder_args["count"] += 1
        builder_args["checkpoint"] = checkpoint
        builder_args["model_type"] = model_type
        raise RuntimeError("cannot build")

    dummy.build_sam2_predictor = builder
    monkeypatch.setitem(sys.modules, "sam2", dummy)
    monkeypatch.delenv("SAM2_CHECKPOINT", raising=False)
    monkeypatch.setenv("SAM_CHECKPOINT", "legacy/model.pth")
    monkeypatch.setenv("SAM_MODEL_TYPE", "vit_b")

    seg_sam2 = _reload_module("seg_sam2")

    fallback_mask = np.ones((6, 6), dtype=np.uint8)
    fallback_calls = {"count": 0}

    def fake_grabcut(frame, box):
        fallback_calls["count"] += 1
        return fallback_mask.copy(), 0.4

    monkeypatch.setattr(seg_sam2, "_grabcut_roi", fake_grabcut)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    boxes = [(2, 2, 7, 8)]

    masks, boxes_out, vis = seg_sam2.infer_roi_masks(frame, boxes)
    assert builder_args == {
        "count": 1,
        "checkpoint": "legacy/model.pth",
        "model_type": "vit_b",
    }
    assert fallback_calls["count"] == 1
    assert masks[0].shape == fallback_mask.shape
    assert boxes_out == [tuple(map(float, boxes[0]))]
    assert vis == [pytest.approx(0.4)]

    masks2, boxes_out2, vis2 = seg_sam2.infer_roi_masks(frame, boxes)
    assert builder_args["count"] == 1  # heavy path skipped after failure
    assert fallback_calls["count"] == 2
    assert boxes_out2 == boxes_out
    assert vis2 == vis
    assert np.array_equal(masks2[0], masks[0])


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


def test_mask2former_factory_from_env(monkeypatch):
    predictor = _DummyPredictor()
    calls = {}

    def factory(**kwargs):
        calls.update(kwargs)
        return predictor

    module = types.ModuleType("mask2former_factory_mod")
    module.build = factory
    monkeypatch.setitem(sys.modules, "mask2former_factory_mod", module)
    monkeypatch.setenv("SEG_MASK2FORMER_FACTORY", "mask2former_factory_mod:build")
    monkeypatch.setenv("SEG_MASK2FORMER_ARGS", json.dumps({"cfg": "config.yaml", "score_thresh": 0.5}))

    seg_mask2former = _reload_module("seg_mask2former")
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    boxes = [(1, 1, 6, 7)]

    masks, boxes_out, vis = seg_mask2former.infer_roi_masks(frame, boxes)
    assert calls == {"cfg": "config.yaml", "score_thresh": 0.5}
    assert predictor.calls == 1
    assert masks[0] is not None and masks[0].mean() == pytest.approx(1.0)
    assert boxes_out == [tuple(map(float, boxes[0]))]
    assert vis == [pytest.approx(1.0)]


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
