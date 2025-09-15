import os
import logging
from dataclasses import dataclass

@dataclass
class TrackerConfig:
    """Simple environment driven configuration."""
    video_path: str = os.getenv("VIDEO_PATH", "/workspace/seesea/data/aviv1.mp4")
    weights_path: str = os.getenv("YOLO_WEIGHTS_PATH", "/workspace/runs_y11x/mix_ftA_clean/weights/best.pt")
    captures_dir: str = os.getenv("CAPTURES_FOLDER", "/workspace/seesea/captures")
    track_conf: float = float(os.getenv("TRACK_CONF", "0.35"))
    track_iou: float = float(os.getenv("TRACK_IOU", "0.9"))


def load_config() -> TrackerConfig:
    """Load configuration from environment variables."""
    return TrackerConfig()


def configure_logging(level: str | None = None) -> None:
    """Configure basic library logging."""
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )
