import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tracker_pipeline as tp
from types import SimpleNamespace


def test_config_parsing(monkeypatch):
    monkeypatch.setenv("VIDEO_PATH", "/tmp/video.mp4")
    monkeypatch.setenv("YOLO_WEIGHTS_PATH", "/tmp/weights.pt")
    monkeypatch.setenv("CAPTURES_FOLDER", "/tmp/captures")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.5")
    monkeypatch.setenv("MAX_FRAMES", "10")
    monkeypatch.setenv("TRACK_CONF", "0.7")
    monkeypatch.setenv("TRACK_IOU", "0.3")

    cfg = tp.TrackerConfig.from_env()
    assert cfg.video_path == "/tmp/video.mp4"
    assert cfg.weights_path == "/tmp/weights.pt"
    assert cfg.captures_dir == "/tmp/captures"
    assert cfg.confidence_threshold == 0.5
    assert cfg.max_frames == 10
    assert cfg.track_conf == 0.7
    assert cfg.track_iou == 0.3


def test_main_pipeline_invocation(monkeypatch, tmp_path):
    called = {}

    def fake_import(name):
        assert name == "supervision_n_yolo"
        return SimpleNamespace(
            VIDEO_PATH=None,
            WEIGHTS_PATH=None,
            CAPTURES_DIR=None,
            CONFIDENCE_THRESHOLD=None,
            MAX_FRAMES=None,
            TRACK_CONF=None,
            TRACK_IOU=None,
            run_tracking_with_supervision=lambda: called.setdefault("ok", True),
        )

    monkeypatch.setenv("CAPTURES_FOLDER", str(tmp_path))
    monkeypatch.setattr(tp.importlib, "import_module", fake_import)

    tp.main()
    assert called.get("ok") is True
