"""Environment driven configuration with documented thresholds."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Dict


@dataclass(frozen=True)
class DetectionThresholds:
    """Detector related thresholds.

    Attributes
    ----------
    confidence:
        Minimum detection confidence accepted from the detector.  Raising this
        keeps only high-quality boxes at the cost of missing faint surfers;
        lowering it yields more proposals but increases clutter.
    iou:
        IoU threshold used when the detector performs NMS.  Higher values keep
        more overlapping boxes, lower values prune aggressively.
    min_area_ratio / max_area_ratio:
        Bounding boxes smaller or larger than these ratios (relative to the
        frame area) are rejected before association.  Widen the interval to
        track more sizes, shrink it to focus on the expected surfer scale.
    """

    confidence: float = 0.30
    iou: float = 0.50
    min_area_ratio: float = 0.0002
    max_area_ratio: float = 0.40

    def describe(self) -> Dict[str, str]:
        return {
            "confidence": "Minimum score produced by the detector.  Lowering it"\
                " yields more candidates but raises the risk of false positives.",
            "iou": "IoU threshold used for detector NMS.  Increase it to keep"\
                " overlapping boxes, decrease it to prune more aggressively.",
            "min_area_ratio": "Reject detections smaller than this share of the"\
                " frame.  Lower to follow tiny objects, raise to ignore speckles.",
            "max_area_ratio": "Reject detections larger than this share of the"\
                " frame.  Increase for close-ups, decrease to ignore huge blobs.",
        }


@dataclass(frozen=True)
class AssociationThresholds:
    """Association gates between detections and active tracks.

    The thresholds make the tracker behaviour explicit:

    * ``min_iou`` – geometric overlap required before we consider a match.
      Raising it enforces tighter spatial continuity, lowering it allows larger
      jumps between frames.
    * ``max_center_distance_ratio`` – maximum allowed distance between the
      detection and the predicted centre, expressed as a fraction of the frame
      diagonal.  Larger values admit faster motion; smaller values demand
      smoother trajectories.
    * ``min_similarity`` – minimum cosine similarity between appearance
      descriptors.  Increasing it ensures that only visually consistent
      detections are linked; decreasing it allows linking noisier crops.
    """

    min_iou: float = 0.35
    max_center_distance_ratio: float = 0.08
    min_similarity: float = 0.40

    def describe(self) -> Dict[str, str]:
        return {
            "min_iou": "Geometric overlap gate.  Raise for strict spatial"\
                " continuity, lower to tolerate fast or jittery motion.",
            "max_center_distance_ratio": "Maximum centre displacement as a"\
                " fraction of the frame diagonal.  Increase for rapid motion,"\
                " decrease to avoid long jumps.",
            "min_similarity": "Appearance cosine similarity gate.  Raise to"\
                " demand consistent crops, lower to allow noisier matches.",
        }


@dataclass(frozen=True)
class AssociationWeights:
    """Weights applied when combining IoU, similarity and centre distance."""

    iou: float = 0.6
    similarity: float = 0.35
    distance: float = 0.05

    def normalised(self) -> "AssociationWeights":
        total = self.iou + self.similarity + self.distance
        if total <= 0:
            raise ValueError("Association weights must sum to a positive value")
        return AssociationWeights(
            iou=self.iou / total,
            similarity=self.similarity / total,
            distance=self.distance / total,
        )

    def describe(self) -> Dict[str, str]:
        return {
            "iou": "Contribution of geometric overlap to the association score.",
            "similarity": "Contribution of appearance cosine similarity.",
            "distance": "Penalty applied to the normalised centre distance.",
        }


@dataclass(frozen=True)
class TrackerConfig:
    """Complete configuration for the tracking pipeline."""

    video_path: str
    weights_path: str
    captures_dir: str
    detection: DetectionThresholds
    association: AssociationThresholds
    association_weights: AssociationWeights
    max_missed_frames: int = 15
    min_track_length: int = 3
    embedding_momentum: float = 0.75

    def describe_thresholds(self) -> Dict[str, Dict[str, str]]:
        """Return human readable explanations for every threshold."""
        return {
            "detection": self.detection.describe(),
            "association": self.association.describe(),
            "association_weights": self.association_weights.describe(),
        }

    @classmethod
    def from_env(cls) -> "TrackerConfig":
        """Build a configuration instance from environment variables."""
        video_path = os.getenv("VIDEO_PATH", "/workspace/seesea/data/aviv1.mp4")
        weights_path = os.getenv(
            "YOLO_WEIGHTS_PATH",
            "/workspace/runs_y11x/mix_ftA_clean/weights/best.pt",
        )
        captures_dir = os.getenv("CAPTURES_FOLDER", "/workspace/seesea/captures")
        detection = DetectionThresholds(
            confidence=float(os.getenv("DETECT_CONFIDENCE", "0.30")),
            iou=float(os.getenv("DETECT_IOU", "0.50")),
            min_area_ratio=float(os.getenv("DETECT_MIN_AREA", "0.0002")),
            max_area_ratio=float(os.getenv("DETECT_MAX_AREA", "0.40")),
        )
        association = AssociationThresholds(
            min_iou=float(os.getenv("ASSOC_MIN_IOU", "0.35")),
            max_center_distance_ratio=float(os.getenv("ASSOC_MAX_CENTER", "0.08")),
            min_similarity=float(os.getenv("ASSOC_MIN_SIM", "0.40")),
        )
        weights = AssociationWeights(
            iou=float(os.getenv("ASSOC_WEIGHT_IOU", "0.6")),
            similarity=float(os.getenv("ASSOC_WEIGHT_SIM", "0.35")),
            distance=float(os.getenv("ASSOC_WEIGHT_DIST", "0.05")),
        ).normalised()
        max_missed = int(os.getenv("TRACK_MAX_MISSED", "15"))
        min_length = int(os.getenv("TRACK_MIN_LENGTH", "3"))
        momentum = float(os.getenv("TRACK_EMB_MOMENTUM", "0.75"))
        return cls(
            video_path=video_path,
            weights_path=weights_path,
            captures_dir=captures_dir,
            detection=detection,
            association=association,
            association_weights=weights,
            max_missed_frames=max_missed,
            min_track_length=min_length,
            embedding_momentum=momentum,
        )


def load_config() -> TrackerConfig:
    """Convenience helper mirroring the previous API."""
    return TrackerConfig.from_env()


def configure_logging(level: str | None = None) -> None:
    """Configure basic library logging."""
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )
