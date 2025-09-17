import importlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from seg_utils import _grabcut_roi

import logging

logger = logging.getLogger(__name__)

_logged = False
_mask2former_predictor: Optional[Any] = None
_mask2former_failed = False
_mask2former_failure_reason: Optional[str] = None


def _get_mask2former_predictor() -> Optional[Any]:
    global _mask2former_failed, _mask2former_predictor, _mask2former_failure_reason, _logged
    if _mask2former_predictor is not None:
        return _mask2former_predictor
    if _mask2former_failed:
        if not _logged and _mask2former_failure_reason:
            logger.warning("[seg] Mask2Former predictor unavailable: %s", _mask2former_failure_reason)
            _logged = True
        return None
    predictor = _build_mask2former_predictor()
    if predictor is None:
        _mask2former_failed = True
        if not _logged and _mask2former_failure_reason:
            logger.warning("[seg] Mask2Former predictor unavailable: %s", _mask2former_failure_reason)
            _logged = True
        return None
    _mask2former_predictor = predictor
    return _mask2former_predictor


def _load_factory(factory_path: str) -> Optional[Any]:
    global _mask2former_failure_reason
    try:
        module_name, attr = factory_path.rsplit(":", 1)
    except ValueError as exc:
        _mask2former_failure_reason = f"invalid factory '{factory_path}': {exc}"
        logger.error("Mask2Former factory path '%s' is invalid", factory_path)
        return None
    try:
        factory_module = importlib.import_module(module_name)
    except Exception as exc:
        _mask2former_failure_reason = f"import '{module_name}' failed: {exc}"
        logger.exception("Mask2Former factory module '%s' import failed", module_name)
        return None
    factory = getattr(factory_module, attr, None)
    if factory is None:
        _mask2former_failure_reason = f"factory '{factory_path}' not found"
        logger.error("Mask2Former factory attribute '%s' missing in module '%s'", attr, module_name)
        return None
    kwargs: Dict[str, Any] = {}
    args_env = os.getenv("SEG_MASK2FORMER_ARGS")
    if args_env:
        try:
            parsed = json.loads(args_env)
            if not isinstance(parsed, dict):
                raise TypeError("SEG_MASK2FORMER_ARGS must decode to a JSON object")
            kwargs.update(parsed)
        except Exception as exc:
            logger.warning("Mask2Former factory args parsing failed: %s", exc)
    try:
        predictor = factory(**kwargs) if callable(factory) else factory
    except Exception as exc:
        _mask2former_failure_reason = f"factory '{factory_path}' raised {exc.__class__.__name__}: {exc}"
        logger.exception("Mask2Former factory '%s' raised an exception", factory_path)
        return None
    if predictor is not None:
        logger.debug("Mask2Former predictor created via factory '%s'", factory_path)
        _mask2former_failure_reason = None
    else:
        _mask2former_failure_reason = f"factory '{factory_path}' returned None"
    return predictor


def _build_mask2former_predictor() -> Optional[Any]:
    global _mask2former_failure_reason
    factory_path = os.getenv("SEG_MASK2FORMER_FACTORY")
    if factory_path:
        predictor = _load_factory(factory_path)
        if predictor is not None:
            return predictor
    module_names = ["detectron2"]
    module_names.extend([
        "detectron2.projects.mask2former",  # common extension location
        "detectron2.engine",  # DefaultPredictor lives here
        "detectron2.engine.defaults",
    ])
    modules = []
    last_error: Optional[str] = None
    for name in module_names:
        try:
            modules.append(importlib.import_module(name))
        except Exception as exc:
            logger.debug("Mask2Former import failed for '%s': %s", name, exc)
            last_error = f"import '{name}' failed: {exc}"
            continue
    for module in modules:
        try:
            predictor = _instantiate_predictor(module)
        except Exception as exc:
            logger.exception("Mask2Former predictor builder from '%s' raised an exception", module.__name__)
            _mask2former_failure_reason = f"builder from '{module.__name__}' raised {exc.__class__.__name__}: {exc}"
            return None
        if predictor is not None:
            logger.debug("Mask2Former predictor instantiated via '%s'", module.__name__)
            _mask2former_failure_reason = None
            return predictor
    if last_error and not _mask2former_failure_reason:
        _mask2former_failure_reason = last_error
    if not _mask2former_failure_reason:
        _mask2former_failure_reason = "no compatible Mask2Former builder found"
    return None


def _instantiate_predictor(module: Any) -> Optional[Any]:
    builder_names = (
        "build_mask2former_predictor",
        "build_predictor",
        "create_predictor",
        "get_predictor",
    )
    for attr in builder_names:
        builder = getattr(module, attr, None)
        if callable(builder):
            try:
                predictor = builder()
            except TypeError:
                logger.debug("Mask2Former builder '%s.%s' rejected default kwargs", module.__name__, attr)
                continue
            except Exception:
                raise
            if predictor is not None:
                return predictor
    class_names = (
        "Mask2FormerPredictor",
        "DefaultPredictor",
        "Predictor",
    )
    for attr in class_names:
        cls = getattr(module, attr, None)
        if cls is None:
            continue
        try:
            predictor = cls()
        except TypeError:
            logger.debug("Mask2Former class '%s.%s' rejected default kwargs", module.__name__, attr)
            continue
        except Exception:
            raise
        if predictor is not None:
            return predictor
    return None


def _call_predictor(predictor: Any,
                    frame_bgr: np.ndarray,
                    det_xyxy: List[Tuple[float, float, float, float]]) -> Any:
    boxes_np = np.asarray(det_xyxy, dtype=np.float32)
    try:
        if hasattr(predictor, "predict_masks") and callable(getattr(predictor, "predict_masks")):
            return predictor.predict_masks(frame_bgr, boxes_np)
        if hasattr(predictor, "predict") and callable(getattr(predictor, "predict")):
            try:
                return predictor.predict(frame_bgr, boxes_np)
            except TypeError:
                return predictor.predict(frame_bgr=frame_bgr, boxes=boxes_np)
        if callable(predictor):
            try:
                return predictor(frame_bgr, boxes_np)
            except TypeError:
                return predictor(frame_bgr=frame_bgr, boxes=boxes_np)
    except Exception:
        raise
    raise RuntimeError("Mask2Former predictor does not expose a usable callable interface")


def _extract_roi(mask: Any,
                 box: Tuple[float, float, float, float],
                 frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    try:
        arr = np.asarray(mask)
    except Exception:
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8)
    Hh, Ww = frame_shape[:2]
    try:
        x1, y1, x2, y2 = map(float, box)
    except Exception:
        return None
    xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    xi2, yi2 = min(Ww, int(np.ceil(x2))), min(Hh, int(np.ceil(y2)))
    if xi2 - xi1 <= 1 or yi2 - yi1 <= 1:
        return None
    roi_h, roi_w = yi2 - yi1, xi2 - xi1
    if arr.shape[0] == Hh and arr.shape[1] == Ww:
        arr = arr[yi1:yi2, xi1:xi2]
    elif arr.shape[0] != roi_h or arr.shape[1] != roi_w:
        try:
            arr = cv2.resize(arr, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            return None
    if arr.size == 0:
        return None
    arr = (arr > 0).astype(np.uint8)
    return arr


def _assign_prediction(output: Any,
                       frame_bgr: np.ndarray,
                       det_xyxy: List[Tuple[float, float, float, float]]
                       ) -> Optional[Tuple[List[Optional[np.ndarray]],
                                            List[Optional[Tuple[float, float, float, float]]],
                                            List[float]]]:
    n = len(det_xyxy)
    masks: List[Optional[np.ndarray]] = [None] * n
    boxes: List[Optional[Tuple[float, float, float, float]]] = [None] * n
    vis: List[float] = [0.0] * n

    def _handle(idx: int,
                mask_obj: Any,
                box_obj: Optional[Tuple[float, float, float, float]] = None,
                visibility: Optional[float] = None) -> None:
        base_box = det_xyxy[idx]
        roi_mask = _extract_roi(mask_obj, base_box if box_obj is None else box_obj, frame_bgr.shape)
        if roi_mask is None:
            return
        masks[idx] = roi_mask
        boxes[idx] = tuple(map(float, base_box if box_obj is None else box_obj))
        vis[idx] = float(visibility if visibility is not None else roi_mask.mean())

    def _from_triplet(data: Any) -> bool:
        if not isinstance(data, (list, tuple)) or len(data) != 3:
            return False
        m_seq, b_seq, v_seq = data
        for i in range(n):
            mask_obj = m_seq[i] if hasattr(m_seq, "__len__") and i < len(m_seq) else None  # type: ignore[arg-type]
            box_obj = None
            if hasattr(b_seq, "__len__") and i < len(b_seq):  # type: ignore[arg-type]
                candidate = b_seq[i]
                if candidate is not None:
                    try:
                        box_obj = tuple(map(float, candidate))  # type: ignore[arg-type]
                    except Exception:
                        box_obj = None
            vis_obj = None
            if hasattr(v_seq, "__len__") and i < len(v_seq):  # type: ignore[arg-type]
                try:
                    vis_obj = float(v_seq[i])  # type: ignore[arg-type]
                except Exception:
                    vis_obj = None
            _handle(i, mask_obj, box_obj, vis_obj)
        return True

    parsed = False
    if isinstance(output, dict):
        m_seq = output.get("masks") or output.get("mask")
        b_seq = output.get("boxes") or output.get("box") or output.get("bboxes")
        v_seq = output.get("vis") or output.get("visibility") or output.get("visibilities")
        if m_seq is not None:
            parsed = _from_triplet((m_seq, b_seq or [None] * n, v_seq or [None] * n))
    elif isinstance(output, (list, tuple)):
        if len(output) == 3:
            parsed = _from_triplet(output)
        if not parsed and len(output) == len(det_xyxy):
            parsed = True
            for i, item in enumerate(output):
                mask_obj: Any = None
                box_obj: Optional[Tuple[float, float, float, float]] = None
                vis_obj: Optional[float] = None
                if isinstance(item, dict):
                    mask_obj = item.get("mask") or item.get("masks")
                    bbox_candidate = item.get("box") or item.get("bbox") or item.get("boxes")
                    if bbox_candidate is not None:
                        try:
                            box_obj = tuple(map(float, bbox_candidate))  # type: ignore[arg-type]
                        except Exception:
                            box_obj = None
                    vis_candidate = item.get("vis") or item.get("visibility")
                    if vis_candidate is not None:
                        try:
                            vis_obj = float(vis_candidate)
                        except Exception:
                            vis_obj = None
                elif isinstance(item, (list, tuple)) and item:
                    mask_obj = item[0]
                    if len(item) > 1:
                        try:
                            box_obj = tuple(map(float, item[1]))  # type: ignore[arg-type]
                        except Exception:
                            box_obj = None
                    if len(item) > 2:
                        try:
                            vis_obj = float(item[2])
                        except Exception:
                            vis_obj = None
                else:
                    mask_obj = item
                _handle(i, mask_obj, box_obj, vis_obj)
    if not parsed:
        return None
    if not any(mask is not None for mask in masks):
        return None
    return masks, boxes, vis


def _fallback_grabcut(frame_bgr: np.ndarray,
                      det_xyxy: List[Tuple[float, float, float, float]]
                      ) -> Tuple[List[Optional[np.ndarray]],
                                 List[Optional[Tuple[float, float, float, float]]],
                                 List[float]]:
    n = len(det_xyxy)
    masks: List[Optional[np.ndarray]] = [None] * n
    boxes: List[Optional[Tuple[float, float, float, float]]] = [None] * n
    vis: List[float] = [0.0] * n
    for i, b in enumerate(det_xyxy):
        mask, vis_i = _grabcut_roi(frame_bgr, b)
        if mask is not None:
            masks[i] = mask
            boxes[i] = tuple(map(float, b))
            vis[i] = vis_i
    return masks, boxes, vis


def infer_roi_masks(frame_bgr: np.ndarray,
                    det_xyxy: List[Tuple[float, float, float, float]]):
    """
    Heavy segmentation placeholder for Mask2Former. Fail-open:
    - If Mask2Former deps are not available, fall back to a quick GrabCut on each ROI.
    Returns (masks, boxes, visibilities) aligned one-to-one with det_xyxy.
    Each mask is ROI-sized (h,w) uint8 with values in {0,1}.
    """
    global _logged, _mask2former_failed, _mask2former_predictor
    n = len(det_xyxy)
    if n == 0:
        return [], [], []
    predictor = None
    try:
        predictor = _get_mask2former_predictor()
        if predictor is not None:
            try:
                raw_output = _call_predictor(predictor, frame_bgr, det_xyxy)
                parsed = _assign_prediction(raw_output, frame_bgr, det_xyxy)
                if parsed is not None:
                    return parsed
            except Exception:
                _mask2former_failed = True
                _mask2former_predictor = None
    except Exception:
        _mask2former_failed = True
        _mask2former_predictor = None
    if not _logged:
        try:
            logger.warning('[seg] Mask2Former not available; using GrabCut ROI fallback')
        except Exception:
            pass
        _logged = True
    return _fallback_grabcut(frame_bgr, det_xyxy)
