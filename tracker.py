"""Clean detection and tracking pipeline with explicit thresholds."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from config import TrackerConfig, load_config
from reid_backbones import ReIDExtractor


@dataclass(frozen=True)
class Detection:
    """Single detector output in absolute pixel coordinates."""

    bbox: np.ndarray  # (x1, y1, x2, y2)
    confidence: float
    class_id: Optional[int] = None

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox.astype(float)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def centre(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox.astype(float)
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


@dataclass
class TrackState:
    """Internal tracker representation for an active track."""

    track_id: int
    bbox: np.ndarray
    feature: np.ndarray
    confidence: float
    last_frame: int
    hits: int
    missed: int

    def observation(self, frame_index: int) -> "TrackObservation":
        return TrackObservation(
            frame_index=frame_index,
            track_id=self.track_id,
            bbox=self.bbox.copy(),
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class TrackObservation:
    """Public tracking output."""

    frame_index: int
    track_id: int
    bbox: np.ndarray
    confidence: float


@dataclass(frozen=True)
class PipelineResult:
    """Container returned by the high level helpers."""

    config: TrackerConfig
    observations: List[TrackObservation]

    def track_count(self) -> int:
        return len({obs.track_id for obs in self.observations})

    def frame_count(self) -> int:
        return 0 if not self.observations else (self.observations[-1].frame_index + 1)

    def threshold_explanations(self) -> Dict[str, Dict[str, str]]:
        return self.config.describe_thresholds()


class DetectionModel:
    """Thin wrapper around an Ultralytics YOLO model."""

    def __init__(self, weights_path: str, config: TrackerConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Ultralytics is required to use the detection model."
            ) from exc
        self.model = YOLO(weights_path)
        self.thresholds = config.detection

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        frame_area = float(height * width)
        result = self.model(
            frame,
            conf=self.thresholds.confidence,
            iou=self.thresholds.iou,
            verbose=False,
        )[0]
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None
        detections: List[Detection] = []
        for i, box in enumerate(xyxy):
            det = Detection(bbox=box.astype(np.float32), confidence=float(conf[i]))
            if cls is not None:
                det = Detection(bbox=det.bbox, confidence=det.confidence, class_id=int(cls[i]))
            ratio = det.area() / frame_area if frame_area > 0 else 0.0
            if ratio < self.thresholds.min_area_ratio:
                continue
            if ratio > self.thresholds.max_area_ratio:
                continue
            detections.append(det)
        return detections


def _clip_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    if x2 <= x1:
        x2 = min(width - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(height - 1.0, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _extract_crop(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
    x1_i, y1_i = int(math.floor(x1)), int(math.floor(y1))
    x2_i, y2_i = int(math.ceil(x2)), int(math.ceil(y2))
    return frame[y1_i:y2_i, x1_i:x2_i]


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _bbox_centre(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(float)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


class SimpleTracker:
    """Minimal yet explicit detection to track association pipeline."""

    def __init__(self, config: TrackerConfig, reid: Optional[ReIDExtractor] = None) -> None:
        self.config = config
        self.reid = reid or ReIDExtractor()
        self.tracks: Dict[int, TrackState] = {}
        self._next_id = 1
        self._history: List[TrackObservation] = []

    @property
    def history(self) -> Sequence[TrackObservation]:
        return tuple(self._history)

    def _create_track(self, detection: Detection, feature: np.ndarray, frame_index: int) -> TrackState:
        feature = feature.astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm > 0.0:
            feature = feature / norm
        track = TrackState(
            track_id=self._next_id,
            bbox=detection.bbox.copy(),
            feature=feature,
            confidence=detection.confidence,
            last_frame=frame_index,
            hits=1,
            missed=0,
        )
        self.tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _association_score(
        self,
        track: TrackState,
        detection: Detection,
        detection_feature: np.ndarray,
        frame_diag: float,
    ) -> Optional[float]:
        iou = _bbox_iou(track.bbox, detection.bbox)
        if iou < self.config.association.min_iou:
            return None
        centre_track = _bbox_centre(track.bbox)
        centre_det = detection.centre()
        centre_distance = float(np.linalg.norm(centre_track - centre_det) / frame_diag)
        if centre_distance > self.config.association.max_center_distance_ratio:
            return None
        similarity = ReIDExtractor.cosine_similarity(track.feature, detection_feature)
        if similarity < self.config.association.min_similarity:
            return None
        weights = self.config.association_weights
        score = (
            weights.iou * iou
            + weights.similarity * similarity
            - weights.distance * centre_distance
        )
        return score

    def step(self, frame_index: int, frame: np.ndarray, detections: Sequence[Detection]) -> List[TrackObservation]:
        if frame.size == 0:
            raise ValueError("Empty frame provided to tracker")
        active: List[TrackObservation] = []
        if not detections:
            self._expire_tracks(frame_index)
            return active
        crops = [_extract_crop(frame, det.bbox) for det in detections]
        features = self.reid.forward(crops)
        height, width = frame.shape[:2]
        frame_diag = float(math.hypot(width, height))
        order = sorted(range(len(detections)), key=lambda idx: detections[idx].confidence, reverse=True)
        used_tracks: set[int] = set()
        for det_idx in order:
            detection = detections[det_idx]
            feature = features[det_idx]
            best_track_id: Optional[int] = None
            best_score = -np.inf
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                score = self._association_score(track, detection, feature, frame_diag)
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_track_id = track_id
            if best_track_id is None:
                track = self._create_track(detection, feature, frame_index)
            else:
                track = self.tracks[best_track_id]
                momentum = self.config.embedding_momentum
                updated_feature = momentum * track.feature + (1.0 - momentum) * feature
                norm = float(np.linalg.norm(updated_feature))
                if norm > 0.0:
                    updated_feature = updated_feature / norm
                track.feature = updated_feature.astype(np.float32)
                track.bbox = detection.bbox.copy()
                track.confidence = detection.confidence
                track.last_frame = frame_index
                track.hits += 1
                track.missed = 0
                self.tracks[best_track_id] = track
            if track.hits >= self.config.min_track_length and track.missed == 0:
                obs = track.observation(frame_index)
                active.append(obs)
                self._history.append(obs)
            used_tracks.add(track.track_id)
        self._expire_tracks(frame_index)
        return active

    def _expire_tracks(self, frame_index: int) -> None:
        to_delete: List[int] = []
        for track_id, track in list(self.tracks.items()):
            if track.last_frame == frame_index:
                continue
            track.missed += 1
            if track.missed > self.config.max_missed_frames:
                to_delete.append(track_id)
            else:
                self.tracks[track_id] = track
        for track_id in to_delete:
            self.tracks.pop(track_id, None)


def run_tracking_with_supervision(config: Optional[TrackerConfig] = None) -> PipelineResult:
    """Run the tracking pipeline and return the collected observations."""

    cfg = config or load_config()
    detector = DetectionModel(cfg.weights_path, cfg)
    tracker = SimpleTracker(cfg)
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {cfg.video_path}")
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector(frame)
            tracker.step(frame_index, frame, detections)
            frame_index += 1
    finally:
        cap.release()
    return PipelineResult(config=cfg, observations=list(tracker.history))


def run_pipeline_notebook(config: Optional[TrackerConfig] = None) -> PipelineResult:
    """Alias used by the notebooks and CLI entry point."""

    return run_tracking_with_supervision(config=config)
