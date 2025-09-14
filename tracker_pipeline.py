from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrackerConfig:
    """Configuration for the tracking pipeline."""

    video_path: str = "/workspace/seesea/data/aviv1.mp4"
    weights_path: str = "/workspace/runs_y11x/mix_ftA_clean/weights/best.pt"
    captures_dir: str = "/workspace/seesea/captures"
    confidence_threshold: float = 0.10
    max_frames: int = 100000
    track_conf: float = 0.35
    track_iou: float = 0.9

    @classmethod
    def from_env(cls, env: Dict[str, str] | None = None) -> "TrackerConfig":
        """Create a configuration from environment variables."""
        if env is None:
            env = os.environ
        return cls(
            video_path=env.get("VIDEO_PATH", cls.video_path),
            weights_path=env.get("YOLO_WEIGHTS_PATH", cls.weights_path),
            captures_dir=env.get("CAPTURES_FOLDER", cls.captures_dir),
            confidence_threshold=float(env.get("CONFIDENCE_THRESHOLD", str(cls.confidence_threshold))),
            max_frames=int(env.get("MAX_FRAMES", str(cls.max_frames))),
            track_conf=float(env.get("TRACK_CONF", str(cls.track_conf))),
            track_iou=float(env.get("TRACK_IOU", str(cls.track_iou))),
        )


def run_tracker(config: TrackerConfig) -> Dict[str, Any]:
    """Run the tracking pipeline with the provided configuration."""
    module = importlib.import_module("supervision_n_yolo")
    module.VIDEO_PATH = config.video_path
    module.WEIGHTS_PATH = config.weights_path
    module.CAPTURES_DIR = config.captures_dir
    module.CONFIDENCE_THRESHOLD = config.confidence_threshold
    module.MAX_FRAMES = config.max_frames
    module.TRACK_CONF = config.track_conf
    module.TRACK_IOU = config.track_iou
    print(f"[weights] using: {os.path.abspath(config.weights_path)}")
    os.makedirs(config.captures_dir, exist_ok=True)
    return module.run_tracking_with_supervision()


def main() -> None:
    """CLI entry point for running the tracker."""
    config = TrackerConfig.from_env()
    run_tracker(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
