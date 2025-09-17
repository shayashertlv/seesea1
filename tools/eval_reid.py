"""Utilities for evaluating saved re-identification embeddings and track files.

This module exposes a small CLI with two subcommands:

```
python tools/eval_reid.py reid --query query_embeddings.npz --gallery gallery_embeddings.npz
python tools/eval_reid.py idf1 --gt mot17_gt.txt --pred tracker_output.txt
```

Both commands emit a JSON summary to stdout and can optionally persist the
metrics to disk via ``--output``.

The implementation intentionally avoids heavy third party dependencies so the
script can be executed in lightweight CI environments.  ``motmetrics`` is used
when available but the module falls back to a deterministic, frame-wise
approximation otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import motmetrics as _motmetrics
except Exception:  # pragma: no cover - dependency not available
    _motmetrics = None

try:  # pragma: no cover - optional dependency
    from scipy.optimize import linear_sum_assignment as _hungarian
except Exception:  # pragma: no cover - dependency not available
    _hungarian = None


ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Embedding evaluation
# ---------------------------------------------------------------------------

def _resolve_array(data: MutableMapping[str, ArrayLike], keys: Sequence[str], *, name: str) -> ArrayLike:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    raise KeyError(f"Unable to locate '{name}' in the provided embedding file. Tried keys: {', '.join(keys)}")


def load_embeddings(path: pathlib.Path) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Load feature vectors, identity labels, and camera identifiers from an ``.npz`` file.

    The loader is intentionally permissive and accepts multiple common naming
    schemes used in re-identification research.
    """

    with np.load(path, allow_pickle=True) as data:
        features = _resolve_array(data, ("features", "embeddings", "feats"), name="features").astype(np.float32)
        ids = _resolve_array(data, ("labels", "ids", "pids", "identities"), name="labels").astype(np.int64)
        if "camids" in data or "cameras" in data or "cam_ids" in data:
            cams = _resolve_array(data, ("camids", "cameras", "cam_ids"), name="camids").astype(np.int64)
        else:
            cams = np.zeros_like(ids)

    if features.ndim != 2:
        raise ValueError("Expected a 2D array of feature vectors with shape (N, D).")
    if ids.shape[0] != features.shape[0]:
        raise ValueError("The number of labels does not match the number of feature vectors.")
    if cams.shape[0] != features.shape[0]:
        raise ValueError("The number of camera identifiers does not match the number of feature vectors.")

    return features, ids, cams


def _normalize(features: ArrayLike) -> ArrayLike:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def compute_reid_metrics(
    query_features: ArrayLike,
    query_ids: ArrayLike,
    query_cams: ArrayLike,
    gallery_features: ArrayLike,
    gallery_ids: ArrayLike,
    gallery_cams: ArrayLike,
    *,
    metric: str = "cosine",
    topk: int = 100,
) -> Dict[str, object]:
    """Compute mAP and CMC curves for a set of query/gallery embeddings."""

    if metric not in {"cosine", "euclidean"}:
        raise ValueError("Unsupported metric: %s" % metric)

    if metric == "cosine":
        query_norm = _normalize(query_features)
        gallery_norm = _normalize(gallery_features)
        distances = query_norm @ gallery_norm.T
        descending = True
    else:
        distances = np.linalg.norm(query_features[:, None, :] - gallery_features[None, :, :], axis=2)
        descending = False

    topk = min(topk, gallery_features.shape[0])
    cmc_curve = np.zeros(topk, dtype=np.float64)
    average_precisions: List[float] = []
    valid_queries = 0

    for q_idx in range(query_features.shape[0]):
        q_id = query_ids[q_idx]
        q_cam = query_cams[q_idx]
        scores = distances[q_idx]

        if metric == "cosine":
            order = np.argsort(scores)[::-1] if descending else np.argsort(scores)
        else:
            order = np.argsort(scores)

        # Filter out gallery samples from the same identity and camera
        mask = ~((gallery_ids == q_id) & (gallery_cams == q_cam))
        order = order[mask[order]]

        matches = gallery_ids[order] == q_id
        if not np.any(matches):
            continue
        valid_queries += 1

        # CMC update
        first_match_idx = np.argmax(matches)
        if matches[first_match_idx]:
            if first_match_idx < topk:
                cmc_curve[first_match_idx:] += 1

        # Average precision
        match_indices = np.flatnonzero(matches)
        precisions = []
        for rank, idx in enumerate(match_indices, start=1):
            if idx >= topk and metric == "cosine":
                # mAP still counts full ranking; CMC truncated.
                pass
            hits = np.count_nonzero(matches[: idx + 1])
            precisions.append(hits / (idx + 1))
        if precisions:
            average_precisions.append(np.mean(precisions))

    if valid_queries == 0:
        raise ValueError("No valid queries with matching gallery identities were found.")

    cmc_curve /= valid_queries
    mAP = float(np.mean(average_precisions)) if average_precisions else 0.0
    cmc_dict = {f"rank-{idx + 1}": float(value) for idx, value in enumerate(cmc_curve)}

    return {
        "mAP": mAP,
        "CMC": cmc_dict,
        "valid_queries": int(valid_queries),
    }


# ---------------------------------------------------------------------------
# MOT / IDF1 evaluation
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)


def _compute_iou(a: BBox, b: BBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _read_mot_file(path: pathlib.Path) -> Dict[int, List[Tuple[int, BBox]]]:
    frames: Dict[int, List[Tuple[int, BBox]]] = {}
    with path.open("r", newline="") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            bbox = BBox(x, y, w, h)
            frames.setdefault(frame, []).append((track_id, bbox))
    return frames


def _linear_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    if _hungarian is not None:  # pragma: no branch - fast path when SciPy is available
        row_ind, col_ind = _hungarian(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    # Fallback: greedy matching by lowest cost.
    matches: List[Tuple[int, int]] = []
    used_rows = set()
    used_cols = set()
    flat_indices = np.argsort(cost, axis=None)
    n_rows, n_cols = cost.shape
    for flat_idx in flat_indices:
        row = flat_idx // n_cols
        col = flat_idx % n_cols
        if row in used_rows or col in used_cols:
            continue
        matches.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
    return matches


def _compute_idf1_simple(
    gt_frames: Dict[int, List[Tuple[int, BBox]]],
    pred_frames: Dict[int, List[Tuple[int, BBox]]],
    *,
    iou_threshold: float,
) -> Dict[str, float]:
    idtp = 0
    idfp = 0
    idfn = 0

    all_frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))
    for frame in all_frames:
        gts = gt_frames.get(frame, [])
        preds = pred_frames.get(frame, [])
        if not gts and not preds:
            continue
        if not gts:
            idfp += len(preds)
            continue
        if not preds:
            idfn += len(gts)
            continue

        cost = np.ones((len(preds), len(gts)), dtype=np.float32)
        for i, (_, pred_box) in enumerate(preds):
            for j, (_, gt_box) in enumerate(gts):
                iou = _compute_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    cost[i, j] = 1.0 - iou
        matches = [(r, c) for r, c in _linear_assignment(cost) if cost[r, c] <= (1.0 - iou_threshold)]
        matched_preds = set()
        matched_gts = set()
        for r, c in matches:
            matched_preds.add(r)
            matched_gts.add(c)
            idtp += 1
        idfp += len(preds) - len(matched_preds)
        idfn += len(gts) - len(matched_gts)

    denom = (2 * idtp + idfp + idfn)
    idf1 = (2 * idtp / denom) if denom > 0 else 0.0
    idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
    idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
    return {
        "IDF1": idf1,
        "IDP": idp,
        "IDR": idr,
        "IDTP": float(idtp),
        "IDFP": float(idfp),
        "IDFN": float(idfn),
    }


def compute_idf1(
    gt_path: pathlib.Path,
    pred_path: pathlib.Path,
    *,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute ID metrics for MOT-style tracking files."""

    gt_frames = _read_mot_file(gt_path)
    pred_frames = _read_mot_file(pred_path)

    if _motmetrics is not None:  # pragma: no cover - depends on optional package
        acc = _motmetrics.MOTAccumulator(auto_id=True)
        frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))
        for frame in frames:
            gts = gt_frames.get(frame, [])
            preds = pred_frames.get(frame, [])
            gt_ids = [str(tid) for tid, _ in gts]
            pred_ids = [str(tid) for tid, _ in preds]
            gt_boxes = [bbox.as_tuple() for _, bbox in gts]
            pred_boxes = [bbox.as_tuple() for _, bbox in preds]
            if gt_boxes and pred_boxes:
                distances = _motmetrics.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0)
                distances = 1.0 - distances
                distances[distances > (1.0 - iou_threshold)] = np.nan
            else:
                distances = np.empty((len(gt_boxes), len(pred_boxes)))
            acc.update(gt_ids, pred_ids, distances)
        metrics_handler = _motmetrics.metrics.create()
        summary = metrics_handler.compute([acc], metrics=["idf1", "idp", "idr", "idtp", "idfp", "idfn"], names=["acc"])
        row = summary.loc["acc"]
        return {
            "IDF1": float(row["idf1"]) if not math.isnan(row["idf1"]) else 0.0,
            "IDP": float(row["idp"]) if not math.isnan(row["idp"]) else 0.0,
            "IDR": float(row["idr"]) if not math.isnan(row["idr"]) else 0.0,
            "IDTP": float(row["idtp"]),
            "IDFP": float(row["idfp"]),
            "IDFN": float(row["idfn"]),
        }

    return _compute_idf1_simple(gt_frames, pred_frames, iou_threshold=iou_threshold)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_reid(args: argparse.Namespace) -> Dict[str, object]:
    query_features, query_ids, query_cams = load_embeddings(pathlib.Path(args.query))
    gallery_features, gallery_ids, gallery_cams = load_embeddings(pathlib.Path(args.gallery))
    metrics = compute_reid_metrics(
        query_features,
        query_ids,
        query_cams,
        gallery_features,
        gallery_ids,
        gallery_cams,
        metric=args.metric,
        topk=args.topk,
    )
    return metrics


def _cmd_idf1(args: argparse.Namespace) -> Dict[str, object]:
    metrics = compute_idf1(
        pathlib.Path(args.gt),
        pathlib.Path(args.pred),
        iou_threshold=args.iou_threshold,
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate re-identification embeddings and tracking outputs.")
    parser.add_argument("--output", type=pathlib.Path, help="Optional JSON file to store the metrics.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_reid = subparsers.add_parser("reid", help="Evaluate embeddings using mAP and CMC metrics.")
    parser_reid.add_argument("--query", required=True, type=pathlib.Path, help="Query embeddings (.npz).")
    parser_reid.add_argument("--gallery", required=True, type=pathlib.Path, help="Gallery embeddings (.npz).")
    parser_reid.add_argument(
        "--metric",
        default="cosine",
        choices=("cosine", "euclidean"),
        help="Distance metric to use for ranking.",
    )
    parser_reid.add_argument("--topk", type=int, default=100, help="Maximum rank considered for the CMC curve.")
    parser_reid.set_defaults(func=_cmd_reid)

    parser_idf1 = subparsers.add_parser("idf1", help="Evaluate MOT challenge style tracks using IDF1/IDP/IDR.")
    parser_idf1.add_argument("--gt", required=True, type=pathlib.Path, help="Ground truth MOT file.")
    parser_idf1.add_argument("--pred", required=True, type=pathlib.Path, help="Predicted MOT file.")
    parser_idf1.add_argument("--iou-threshold", type=float, default=0.5, help="Minimum IoU for a match.")
    parser_idf1.set_defaults(func=_cmd_idf1)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    parser = build_parser()
    args = parser.parse_args(argv)
    metrics = args.func(args)
    if args.output:
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
