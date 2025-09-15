import os
from typing import Any, Dict, List, Tuple, Optional
from typing import Tuple as _Tuple

import cv2
import numpy as np
import json
from collections import deque
import logging
from models.seqtrack import SeqTrackLSTM

# GMC & ReID pluggable backbones
try:
    from gmc import estimate_gmc, warp_box_xyxy, warp_point, reset_gmc_smoothing
except Exception:
    estimate_gmc = None  # type: ignore
    warp_box_xyxy = None  # type: ignore
    warp_point = None  # type: ignore

    def reset_gmc_smoothing() -> None:  # type: ignore
        pass

# Optional modules (graceful)
try:
    from appearance_memory import AppearanceMemory  # type: ignore
except Exception:
    AppearanceMemory = None  # type: ignore
try:
    from sr_crops import CropSR  # type: ignore
except Exception:
    CropSR = None  # type: ignore

try:
    from reid_backbones import ReIDExtractor as _PluggableReID
except Exception:
    _PluggableReID = None  # type: ignore

# Pluggable ReID instance placeholder (explicit global)
_REID_PLUG = None  # pluggable ReID instance if available

# Notebook inline display (graceful if not in Jupyter)
IN_NOTEBOOK = False



# ================================================================
# Surfers tracker: YOLO + Supervision(ByteTrack) with appearance-aware association
# How to enable appearance-aware association:
#   Set ASSOC_APPEAR_ENABLE=1 (default). This uses fused IoU + deep ReID cosine + HSV hist with Hungarian.
#   Set ASSOC_APPEAR_ENABLE=0 to use the original ByteTrack-only path.
# Recommended envs for surfers (1080p–4K):
#   ASSOC_W_IOU=0.45 ASSOC_W_EMB=0.45 ASSOC_W_HIST=0.10
#   ASSOC_MIN_IOU=0.15 ASSOC_MAX_CENTER_DIST=0.08
#   BT_MATCH=0.45 REID_PAD_RATIO=0.12 REID_BATCH_SIZE=32
# Notes on OSNet fallback:
#   We prefer OSNet via torchreid (or torchvision.osnet_x1_0 if available). If unavailable, we fall back to ResNet-50.
# Minimal output:
#   By default, only periodic summary every LOG_EVERY frames and the final one-line summary are printed.
#   Set DRAW_DEBUG=1 to enable on-frame drawings/HUD for debugging.
# ================================================================

# Ultralytics YOLO
try:
    from ultralytics import YOLO

    # Monkey-patch Ultralytics model fusion (guarded by UL_SAFE_FUSE)
    try:
        import os as _os_ul

        if int(_os_ul.getenv("UL_SAFE_FUSE", "0")) == 1:
            from ultralytics.nn.tasks import BaseModel as _UL_BaseModel  # type: ignore

            _ORIG_FUSE = getattr(_UL_BaseModel, "fuse", None)


            def _safe_fuse(self, verbose: bool = True):  # noqa: ANN001
                try:
                    if callable(_ORIG_FUSE):
                        return _ORIG_FUSE(self, verbose=verbose)
                except Exception as e:  # pragma: no cover - runtime safety
                    logger.warning("[sv-pipeline] Warning: skipping model fusion due to: %s", e)
                return self


            _UL_BaseModel.fuse = _safe_fuse  # type: ignore[attr-defined]
    except Exception:
        # If patching fails, continue without it
        pass
    HAS_YOLO = True
except Exception:
    YOLO = None
    HAS_YOLO = False

# Supervision (Roboflow)
try:
    import supervision as sv

    HAS_SUPERVISION = True
except Exception:
    sv = None
    HAS_SUPERVISION = False

from config import load_config

logger = logging.getLogger(__name__)

_cfg = load_config()
VIDEO_PATH = _cfg.video_path
WEIGHTS_PATH = _cfg.weights_path
CAPTURES_DIR = _cfg.captures_dir
TRACK_CONF = _cfg.track_conf
TRACK_IOU = _cfg.track_iou

# Prefer Shay's local weights on Windows if env not set and file exists
_default_weights_win = r"C:\Users\Shay\PycharmProjects\seesea\weights\surf_polish640_best.pt"
if os.name == "nt" and "YOLO_WEIGHTS_PATH" not in os.environ and os.path.exists(_default_weights_win):
    WEIGHTS_PATH = _default_weights_win
logger.info("[weights] using: %s", os.path.abspath(WEIGHTS_PATH))

# On Windows, prefer explicit absolute default if not overridden
if os.name == "nt" and os.getenv("CAPTURES_FOLDER", "").strip() == "":
    CAPTURES_DIR = r"C:\Users\Shay\PycharmProjects\seesea\app\static\captures"

# Per-frame knobs
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.10"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "100000"))

# Optional Torch/Torchvision (for deep ReID embeddings)
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as _T

    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    torchvision = None
    _T = None
    HAS_TORCH = False

# Hoisted normalization tensors (CPU) for preprocessing reuse
if HAS_TORCH:
    try:
        _NORM_MEAN_T = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        _NORM_STD_T = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    except Exception:
        _NORM_MEAN_T = None
        _NORM_STD_T = None
else:
    _NORM_MEAN_T = None
    _NORM_STD_T = None

# LSTM-based trajectory predictor (SeqTrack-LSTM)
# implementation lives in models.seqtrack
traj_predictor = None

# Optional PIL for preprocessing
try:
    from PIL import Image

    HAS_PIL = True
except Exception:
    Image = None
    HAS_PIL = False

# Class filtering: comma-separated indices, e.g. "0" or "0,1"
_TRACK_CLASSES_STR = os.getenv("TRACK_CLASSES", "").strip()
TRACK_CLASSES = None
if _TRACK_CLASSES_STR:
    try:
        TRACK_CLASSES = [int(x) for x in _TRACK_CLASSES_STR.split(",") if x.strip() != ""]
    except Exception:
        TRACK_CLASSES = None

# Visual stability helpers and drawing (all gated)
DRAW_DEBUG = int(os.getenv("DRAW_DEBUG", "0")) == 1
DEBUG = int(os.getenv("DEBUG", "0")) == 1
SMOOTH_BOXES = int(os.getenv("SMOOTH_BOXES", "1")) == 1  # ON
SMOOTH_LENGTH = int(os.getenv("SMOOTH_LENGTH", "11"))  # sliding window (frames)
DRAW_PRED_FOR = int(os.getenv("DRAW_PRED_FOR", "10"))  # predicted overlay length
DRAW_TRAILS = int(os.getenv("DRAW_TRAILS", "1")) == 1
TRAIL_LENGTH = int(os.getenv("TRAIL_LENGTH", "50"))
WRITE_VIDEO = int(os.getenv("WRITE_VIDEO", "1")) == 1
DRY_RUN = int(os.getenv("DRY_RUN", "0")) == 1
LOG_EVERY = int(os.getenv("LOG_EVERY", "0"))

# ===== Diagnostics toggles (lightweight) =====
DIAG = int(os.getenv("DIAG", "1")) == 1  # master switch
DIAG_JSON = int(os.getenv("DIAG_JSON", "0")) == 1  # also write jsonl in CAPTURES_DIR
DIAG_EVERY = max(1, int(os.getenv("DIAG_EVERY", str(max(1, LOG_EVERY or 60)))))  # default aligns with LOG_EVERY or 60

# json helper (lazy file open)
_diag_fp = None


def _diag_emit(event: str, **kv):
    try:
        if not DIAG:
            return
        rec = {"evt": event, **kv}
        logger.debug("[sv][diag] %s", json.dumps(rec))
        if DIAG_JSON:
            global _diag_fp
            if _diag_fp is None:
                try:
                    ensure_dir(CAPTURES_DIR)
                except Exception:
                    pass
                base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
                _diag_fp = open(os.path.join(CAPTURES_DIR, f"diag_{base}.jsonl"), "a", encoding="utf-8")
            _diag_fp.write(json.dumps(rec) + "\n")
            _diag_fp.flush()
    except Exception:
        pass


ASSOC_COST_DEBUG = int(os.getenv("ASSOC_COST_DEBUG", "0")) == 1
PROFILE_EMB = int(os.getenv("PROFILE_EMB", "0")) == 1
# Eval render (annotated video; independent of DRAW_DEBUG)
EVAL_RENDER = int(os.getenv("EVAL_RENDER", "1")) == 1
EVAL_RENDER_TRAILS = int(os.getenv("EVAL_RENDER_TRAILS", "1")) == 1
EVAL_TRAIL_LENGTH = int(os.getenv("EVAL_TRAIL_LENGTH", "50"))

# Optional ROI polygon: "x1,y1;x2,y2;..."
ROI_POLY_STR = os.getenv("ROI_POLY", "").strip()
ROI_ENFORCE = int(os.getenv("ROI_ENFORCE", "1")) == 1  # if 1, filter to ROI *before* tracking

# Supervision ByteTrack knobs (mapped from your prior envs or sensible defaults)
BT_TRACK_THRESH = float(os.getenv("BT_HIGH", "0.50"))  # activation threshold
BT_BUFFER = int(os.getenv("BT_BUFFER", "90"))  # frames to keep lost tracks
BT_MATCH_THRESH = float(os.getenv("BT_MATCH", "0.80"))  # IoU matching thresh (surfer tuned)

# Pre-tracker gating knobs (surfer-friendly defaults)
MIN_AREA_RATIO = float(os.getenv("MIN_AREA_RATIO", "0.00005"))  # min box area relative to frame
MAX_AREA_RATIO = float(os.getenv("MAX_AREA_RATIO", "0.40"))  # max box area relative to frame
MIN_ASPECT_RATIO = float(os.getenv("MIN_ASPECT_RATIO", "0.15"))  # w/h lower bound
MAX_ASPECT_RATIO = float(os.getenv("MAX_ASPECT_RATIO", "5.0"))  # w/h upper bound

# Appearance-aware association flags (BoT-SORT-ish)
ASSOC_APPEAR_ENABLE = int(os.getenv("ASSOC_APPEAR_ENABLE", "1")) == 1
ASSOC_W_IOU = float(os.getenv("ASSOC_W_IOU", "0.35"))
ASSOC_W_EMB = float(os.getenv("ASSOC_W_EMB", "0.55"))
ASSOC_W_HIST = float(os.getenv("ASSOC_W_HIST", "0.20"))
ASSOC_MIN_IOU = float(os.getenv("ASSOC_MIN_IOU", "0.18"))
ASSOC_MAX_CENTER_DIST = float(os.getenv("ASSOC_MAX_CENTER_DIST", "0.055"))  # fraction of frame diag
# EMA for appearance (per-track)
EMB_EMA_ENABLE = int(os.getenv("EMB_EMA_ENABLE", "1")) == 1
EMB_EMA_M = float(os.getenv("EMB_EMA_M", "0.80"))
HIST_EMA_ENABLE = int(os.getenv("HIST_EMA_ENABLE", "1")) == 1
HIST_EMA_M = float(os.getenv("HIST_EMA_M", "0.60"))
# Anisotropic motion gate (elliptical)
ANISO_GATE_ENABLE = int(os.getenv("ANISO_GATE_ENABLE", "1")) == 1
GATE_SIGMA_X_FRAC = float(os.getenv("GATE_SIGMA_X_FRAC", "0.10"))
GATE_SIGMA_Y_FRAC = float(os.getenv("GATE_SIGMA_Y_FRAC", "0.05"))
# Commit delay for new IDs (reclaim window)
TENTATIVE_FRAMES = int(os.getenv("TENTATIVE_FRAMES", "5"))
# Histogram focus (board-aware)
HIST_FOCUS = os.getenv("HIST_FOCUS", "lower").strip()
# Adaptive appearance weighting envs
ADAPTIVE_WEIGHT = int(os.getenv("ADAPTIVE_WEIGHT", "1")) == 1
LOW_IOU_THRESH = float(os.getenv("LOW_IOU_THRESH", "0.35"))
HIGH_IOU_THRESH = float(os.getenv("HIGH_IOU_THRESH", "0.60"))
VEL_K = float(os.getenv("VEL_K", "0.03"))
ACC_K = float(os.getenv("ACC_K", "0.02"))
# Preset and GMC ORB params
PRESET = os.getenv("PRESET", "").strip().lower()
ORB_NFEATURES = int(os.getenv("ORB_NFEATURES", "1000"))
ORB_RANSAC_THRESH = float(os.getenv("ORB_RANSAC_THRESH", "3.0"))
ORB_DOWNSCALE = int(os.getenv("ORB_DOWNSCALE", "1"))
# Detector test-time augmentation cadence (legacy Ultralytics augment flag)
DET_TTA_EVERY = int(os.getenv("DET_TTA_EVERY", "0"))
# New detector ensemble + TTA flags
DET_ENSEMBLE = int(os.getenv("DET_ENSEMBLE", "0")) == 1
DET_MODELS = os.getenv("DET_MODELS", "").strip()  # semicolon-separated YOLO weight paths
DET_WBF_IOU = float(os.getenv("DET_WBF_IOU", "0.55"))
DET_WBF_SKIP_BOXES = int(os.getenv("DET_WBF_SKIP_BOXES", "0"))
DET_TTA = int(os.getenv("DET_TTA", "0")) == 1
DET_TTA_FLIP = int(os.getenv("DET_TTA_FLIP", "1")) == 1
DET_TTA_SCALES = os.getenv("DET_TTA_SCALES", "1.0").strip()
DET_TTA_WBF_IOU = float(os.getenv("DET_TTA_WBF_IOU", str(os.getenv("DET_WBF_IOU", "0.55"))))

# Appearance-memory RNN over recent embeddings/hists
LSTM_APPEAR_ENABLE = int(os.getenv("LSTM_APPEAR_ENABLE", "1")) == 1
LSTM_APPEAR_HIDDEN = int(os.getenv("LSTM_APPEAR_HIDDEN", "128"))
LSTM_APPEAR_WINDOW = int(os.getenv("LSTM_APPEAR_WINDOW", "16"))

# Camera motion compensation method: auto|orb|flow|raft|loftr|gmflow
GMC_METHOD = os.getenv("GMC_METHOD", "orb").strip().lower()
GMC_BACKEND = os.getenv("GMC_BACKEND", "").strip().lower()
if GMC_BACKEND in ("orb", "flow", "raft", "loftr", "gmflow"):
    # Map GMC_BACKEND to GMC_METHOD for internal use
    GMC_METHOD = ("oflow" if GMC_BACKEND == "flow" else GMC_BACKEND)
elif GMC_METHOD not in ("auto", "orb", "flow", "oflow", "raft", "loftr", "gmflow"):
    GMC_METHOD = "auto"
GMC_INLIER_MIN = float(os.getenv("GMC_INLIER_MIN", "0.35"))
GMC_DOWNSCALE = int(os.getenv("GMC_DOWNSCALE", str(ORB_DOWNSCALE)))
GMC_MESH = int(os.getenv("GMC_MESH", "0")) == 1

# --- New tuning/env flags ---
KALMAN_ENABLE = int(os.getenv("KALMAN_ENABLE", "1")) == 1
KALMAN_Q_POS = float(os.getenv("KALMAN_Q_POS", "0.02"))
KALMAN_Q_VEL = float(os.getenv("KALMAN_Q_VEL", "0.02"))
KALMAN_R_POS = float(os.getenv("KALMAN_R_POS", "0.6"))

# LSTM motion predictor toggles
LSTM_MOTION_ENABLE = int(os.getenv("LSTM_MOTION_ENABLE", "1")) == 1
LSTM_HISTORY_LEN = int(os.getenv("LSTM_HISTORY_LEN", "15"))
LSTM_MIN_HISTORY = int(os.getenv("LSTM_MIN_HISTORY", "8"))
LSTM_VARIANT = os.getenv("LSTM_VARIANT", "B").strip().upper()
LSTM_HIDDEN = int(os.getenv("LSTM_HIDDEN", "128"))
LSTM_FP16 = int(os.getenv("LSTM_FP16", "1")) == 1

MOT_EXPORT = int(os.getenv("MOT_EXPORT", "1")) == 1
# normalize association weights once (may be updated by preset overrides below)
ASSOC_W_MOT = float(os.getenv("ASSOC_W_MOT", "0.35"))
_WI, _WE, _WH, _WM = ASSOC_W_IOU, ASSOC_W_EMB, ASSOC_W_HIST, ASSOC_W_MOT
_ws = max(1e-6, (_WI + _WE + _WH + _WM))
ASSOC_W_IOU, ASSOC_W_EMB, ASSOC_W_HIST, ASSOC_W_MOT = _WI / _ws, _WE / _ws, _WH / _ws, _WM / _ws

# ReID backbone and batching
REID_BACKBONE = os.getenv("REID_BACKBONE", "dinov2_vits14")  # osnet|fastreid_r50|dinov2_vits14
REID_PAD_RATIO = float(os.getenv("REID_PAD_RATIO", "0.18"))
# Prefer bigger batches on GPU; fall back if OOM occurs
if "REID_BATCH_SIZE" not in os.environ:
    REID_BATCH_SIZE = 64
else:
    REID_BATCH_SIZE = int(os.getenv("REID_BATCH_SIZE", "32"))
REID_DEVICE = os.getenv("REID_DEVICE", "cuda")  # auto|cpu|cuda
REID_EVERY_N = int(os.getenv("REID_EVERY_N", "1"))
REID_FP16 = int(os.getenv("REID_FP16", "1")) == 1
# Auto-enable FP16 for ReID on CUDA unless explicitly overridden by env
try:
    if (
            "REID_FP16" not in os.environ) and HAS_TORCH and torch is not None and torch.cuda.is_available() and REID_DEVICE in (
            "auto", "cuda"):
        REID_FP16 = True
except Exception:
    pass

# Apply PRESET defaults if provided and specific envs not set
if PRESET in ("1080p_calm", "4k_choppy"):
    has_w = ("ASSOC_W_IOU" in os.environ) or ("ASSOC_W_EMB" in os.environ) or ("ASSOC_W_HIST" in os.environ)
    if PRESET == "1080p_calm":
        if not has_w:
            ASSOC_W_IOU, ASSOC_W_EMB, ASSOC_W_HIST = 0.6, 0.3, 0.1
        if "BT_BUFFER" not in os.environ:
            BT_BUFFER = 30
        if "ORB_NFEATURES" not in os.environ:
            ORB_NFEATURES = 1000
        if "ORB_DOWNSCALE" not in os.environ:
            ORB_DOWNSCALE = 1
    elif PRESET == "4k_choppy":
        if not has_w:
            ASSOC_W_IOU, ASSOC_W_EMB, ASSOC_W_HIST = 0.5, 0.4, 0.1
        if "BT_BUFFER" not in os.environ:
            BT_BUFFER = 50
        if "ORB_NFEATURES" not in os.environ:
            ORB_NFEATURES = 1500
        if "ORB_DOWNSCALE" not in os.environ:
            ORB_DOWNSCALE = 2
    # renormalize (include MOT if present)
    try:
        _WW = ASSOC_W_IOU + ASSOC_W_EMB + ASSOC_W_HIST + ASSOC_W_MOT
    except Exception:
        _WW = ASSOC_W_IOU + ASSOC_W_EMB + ASSOC_W_HIST
    _WW = max(1e-6, float(_WW))
    try:
        ASSOC_W_IOU, ASSOC_W_EMB, ASSOC_W_HIST, ASSOC_W_MOT = ASSOC_W_IOU / _WW, ASSOC_W_EMB / _WW, ASSOC_W_HIST / _WW, ASSOC_W_MOT / _WW
    except Exception:
        ASSOC_W_IOU, ASSOC_W_EMB, ASSOC_W_HIST = ASSOC_W_IOU / _WW, ASSOC_W_EMB / _WW, ASSOC_W_HIST / _WW

# Safer default: flexible association weights unless explicitly overridden
if ("ASSOC_W_IOU" not in os.environ) and ("ASSOC_W_EMB" not in os.environ) and ("ASSOC_W_HIST" not in os.environ):
    ASSOC_W_IOU, ASSOC_W_EMB, ASSOC_W_HIST = 0.55, 0.35, 0.10
    _WW = max(1e-6, (ASSOC_W_IOU + ASSOC_W_EMB + ASSOC_W_HIST))
    ASSOC_W_IOU, ASSOC_W_EMB, ASSOC_W_HIST = ASSOC_W_IOU / _WW, ASSOC_W_EMB / _WW, ASSOC_W_HIST / _WW

# Supervision InferenceSlicer (SAHI) – optional tiling to avoid OOM and catch small targets
SLICER_ENABLE = int(os.getenv("SLICER_ENABLE", "0")) == 1
SLICE_W = int(os.getenv("SLICE_W", "640"))
SLICE_H = int(os.getenv("SLICE_H", "640"))
SLICER_OVERLAP_RATIO_W = float(os.getenv("SLICER_OVERLAP_RATIO_W", "0.20"))
SLICER_OVERLAP_RATIO_H = float(os.getenv("SLICER_OVERLAP_RATIO_H", "0.20"))
SLICER_THREADS = int(os.getenv("SLICER_THREADS", "2"))
# Slicer call safety knobs
SLICER_CALL_TIMEOUT = float(os.getenv("SLICER_CALL_TIMEOUT", "4.0"))  # seconds
SLICER_TIMEOUT_DISABLE_AFTER = int(os.getenv("SLICER_TIMEOUT_DISABLE_AFTER", "2"))

# ID re-assignment (stitching) knobs
ID_REASSIGN_WINDOW = int(os.getenv("ID_REASSIGN_WINDOW", "60"))  # frames to consider a recently lost ID
ID_REASSIGN_IOU = float(os.getenv("ID_REASSIGN_IOU", "0.05"))  # min IoU between predicted ghost box and detection
ID_REASSIGN_DIST_RATIO = float(
    os.getenv("ID_REASSIGN_DIST_RATIO", "0.08"))  # max center distance as ratio of frame diagonal

# Appearance-based re-id knobs (HSV histogram)
APPEAR_ENABLE = int(os.getenv("APPEAR_ENABLE", "1")) == 1
APPEAR_MIN_SIM = float(os.getenv("APPEAR_MIN_SIM", "0.20"))  # min histogram similarity (0..1 for CORREL-ish)
HIST_BINS_H = int(os.getenv("HIST_BINS_H", "32"))
HIST_BINS_S = int(os.getenv("HIST_BINS_S", "32"))
HIST_BINS_V = int(os.getenv("HIST_BINS_V", "8"))
APPEAR_ALPHA = float(os.getenv("APPEAR_ALPHA", "0.40"))  # EMA factor for hist update

# Deep ReID embedding knobs (stronger re-id)
APPEAR_EMB_ENABLE = int(os.getenv("APPEAR_EMB_ENABLE", "1")) == 1
APPEAR_EMB_MIN_SIM = float(os.getenv("APPEAR_EMB_MIN_SIM", "0.25"))  # min cosine similarity (0..1)
APPEAR_EMB_ALPHA = float(os.getenv("APPEAR_EMB_ALPHA", "0.50"))  # EMA for embedding
APPEAR_SIM_W_EMB = float(os.getenv("APPEAR_SIM_W_EMB", "0.7"))  # weight for embedding sim
APPEAR_SIM_W_HIST = float(os.getenv("APPEAR_SIM_W_HIST", "0.3"))  # weight for histogram sim

# ---- Lightweight masking (ROI segmentation) ----
SEG_ENABLE = int(os.getenv("SEG_ENABLE", "1")) == 1
FORCE_LITE_SEG = int(os.getenv("FORCE_LITE_SEG", "0")) == 1
SEG_BACKEND = os.getenv("SEG_BACKEND", "grabcut").strip()  # grabcut|bgs
SEG_ON_AMBIGUITY = int(os.getenv("SEG_ON_AMBIGUITY", "1")) == 1  # near-threshold sim
SEG_ON_OVERLAP = int(os.getenv("SEG_ON_OVERLAP", "1")) == 1  # det-det IoU cluster
SEG_ON_REAPPEAR = int(os.getenv("SEG_ON_REAPPEAR", "1")) == 1  # ghost stitch cases
SEG_MAX_PER_FRAME = int(os.getenv("SEG_MAX_PER_FRAME", "3"))  # hard cap per frame
SEG_MS_BUDGET = float(os.getenv("SEG_MS_BUDGET", "18.0"))  # soft time budget (ms)
SEG_REUSE_FOR = int(os.getenv("SEG_REUSE_FOR", "6"))  # frames to keep a mask
SEG_AMBIG_MARGIN = float(os.getenv("SEG_AMBIG_MARGIN", "0.06"))  # sim in [ASSOC_MIN_SIM, +margin]
SEG_OVERLAP_IOU = float(os.getenv("SEG_OVERLAP_IOU", "0.20"))

# mask-guided features & updates
MASK_GUIDED_HIST = int(os.getenv("MASK_GUIDED_HIST", "1")) == 1
MASK_GUIDED_EMB = int(os.getenv("MASK_GUIDED_EMB", "1")) == 1
VIS_MIN_FOR_UPDATE = float(os.getenv("VIS_MIN_FOR_UPDATE", "0.50"))  # skip EMA if vis below this
ALPHA_MASK_IOU = float(os.getenv("ALPHA_MASK_IOU", "0.50"))  # blend: IoU_soft = b1*box + (1-b1)*mask

# GrabCut/BGS backend knobs
SEG_PAD_RATIO = float(os.getenv("SEG_PAD_RATIO", "0.08"))
SEG_GC_ITERS = int(os.getenv("SEG_GC_ITERS", "3"))
SEG_MORPH_KERNEL = int(os.getenv("SEG_MORPH_KERNEL", "3"))  # odd size, 0=off
SEG_BGS_KIND = os.getenv("SEG_BGS_KIND", "knn").strip()  # knn|gsoc (hint mask)
SEG_DEBUG_DRAW = int(os.getenv("SEG_DEBUG_DRAW", "0")) == 1

# Appearance memory (EMA) envs
APPMEM_ENABLE = int(os.getenv("APPMEM_ENABLE", "1")) == 1
APPMEM_BACKEND = os.getenv("APPMEM_BACKEND", "ema").strip()
APPMEM_ON_CONFLICT = int(os.getenv("APPMEM_ON_CONFLICT", "1")) == 1
APPMEM_ALPHA = float(os.getenv("APPMEM_ALPHA", "0.20"))
APPMEM_SWITCH_MARGIN = float(os.getenv("APPMEM_SWITCH_MARGIN", "0.08"))
APPMEM_FREEZE_AFTER_SWITCH = int(os.getenv("APPMEM_FREEZE_AFTER_SWITCH", "5"))
APPMEM_MIN_VIS = float(os.getenv("APPMEM_MIN_VIS", "0.25"))
APPMEM_MIN_SIDE = int(os.getenv("APPMEM_MIN_SIDE", "64"))
APPMEM_MAX_BLUR = float(os.getenv("APPMEM_MAX_BLUR", "2.0"))

# Super-resolution for small ReID crops
SR_REID_ENABLE = int(os.getenv("SR_REID_ENABLE", "1")) == 1
SR_MIN_SIDE = int(os.getenv("SR_MIN_SIDE", "64"))

# --- safer association + updates ---
ASSOC_MIN_SIM = float(os.getenv("ASSOC_MIN_SIM", "0.42"))  # min fused sim to accept a pair
UPDATE_EMB_IOU = float(os.getenv("UPDATE_EMB_IOU", "0.55"))  # update appearance only when IoU is solid
UPDATE_EMB_CONF = float(os.getenv("UPDATE_EMB_CONF", "0.50"))  # ...and detection confidence is decent
UPDATE_EMB_MIN_AGE = int(os.getenv("UPDATE_EMB_MIN_AGE", "4"))  # learn appearance only after the track matures

# --- narrower, safer ghost reclaim horizon ---
REASSIGN_MAX_MISS = int(os.getenv("REASSIGN_MAX_MISS", "24"))  # e.g., ~1s @ 24 fps

# Run profiles (reference):
# Baseline sanity (tracker only):
#   ASSOC_APPEAR_ENABLE=0
#   BT_MATCH=0.45 BT_HIGH=0.25 BT_BUFFER=180
#   SMOOTH_BOXES=1
#
# Safe appearance ON (geometry-first):
#   ASSOC_APPEAR_ENABLE=1
#   ASSOC_W_IOU=0.70 ASSOC_W_EMB=0.25 ASSOC_W_HIST=0.05
#   ASSOC_MIN_IOU=0.15 ASSOC_MAX_CENTER_DIST=0.08
#   ASSOC_MIN_SIM=0.38
#   APPEAR_ESCAPE_ENABLE=0
#   UPDATE_EMB_IOU=0.45 UPDATE_EMB_CONF=0.40 UPDATE_EMB_MIN_AGE=3
#   REASSIGN_MAX_MISS=24 ID_REASSIGN_WINDOW=24
#   ID_REASSIGN_IOU=0.20 ID_REASSIGN_DIST_RATIO=0.06
#   REID_EVERY_N=1
#   SMOOTH_BOXES=1

# Follow (per-ID square crop exporter)
FOLLOW_EXPORT = int(os.getenv("FOLLOW_EXPORT", "1")) == 1
FOLLOW_SIZE = int(os.getenv("FOLLOW_SIZE", "256"))
FOLLOW_PAD = float(os.getenv("FOLLOW_PAD", "0.25"))
FOLLOW_MIN_AGE = int(os.getenv("FOLLOW_MIN_AGE", "8"))
FOLLOW_DIRNAME = os.getenv("FOLLOW_DIRNAME", "follows")

APPEAR_ESCAPE_ENABLE = int(os.getenv("APPEAR_ESCAPE_ENABLE", "0")) == 0
APPEAR_ESCAPE_COS = float(os.getenv("APPEAR_ESCAPE_COS", "0.62"))  # cosine in [0..1]
APPEAR_ESCAPE_HIST = float(os.getenv("APPEAR_ESCAPE_HIST", "0.68"))  # histogram sim in [0..1]
# Feature bank and ambiguity handling
EMB_BANK_MAX = int(os.getenv("EMB_BANK_MAX", "30"))
EMB_BANK_MIN_SIM_UPDATE = float(os.getenv("EMB_BANK_MIN_SIM_UPDATE", "0.30"))
EMB_BANK_DRIFT_INIT_GUARD = float(os.getenv("EMB_BANK_DRIFT_INIT_GUARD", "0.20"))
APPEAR_MARGIN_MIN = float(
    os.getenv("APPEAR_MARGIN_MIN", "0.08"))  # min margin between best and second-best emb sim per det
ESCAPE_CENTER_MULT = float(os.getenv("ESCAPE_CENTER_MULT", "2.0"))  # cap appearance override to <=2x gate
GATE_VEL_BOOST = float(os.getenv("GATE_VEL_BOOST", "1.4"))  # expand gate for fast movers
AA_STITCH_ENABLE = int(os.getenv("AA_STITCH_ENABLE", "1")) == 1  # enable AA short-gap stitcher
# Long-term archive for ID reuse
ARCHIVE_ENABLE = int(os.getenv("ARCHIVE_ENABLE", "1")) == 1
ARCHIVE_MAX = int(os.getenv("ARCHIVE_MAX", "100"))
ARCHIVE_SIM_THR = float(os.getenv("ARCHIVE_SIM_THR", "0.80"))
ARCHIVE_TTL = int(os.getenv("ARCHIVE_TTL", "600"))  # frames
TRACK_ARCHIVE: list = []

# --- Auto-Flex thresholds ---
FLEX_ENABLE = int(os.getenv("FLEX_ENABLE", "0")) == 0
FLEX_ALPHA = float(os.getenv("FLEX_ALPHA", "0.25"))
FLEX_LOG = int(os.getenv("FLEX_LOG", "0")) == 1  # print tuned values periodically

# Hard rails
FLEX_CONF_MIN = float(os.getenv("FLEX_CONF_MIN", "0.25"))
FLEX_CONF_MAX = float(os.getenv("FLEX_CONF_MAX", "0.55"))
FLEX_IOU_MIN = float(os.getenv("FLEX_IOU_MIN", "0.02"))
FLEX_IOU_MAX = float(os.getenv("FLEX_IOU_MAX", "0.50"))
FLEX_SIM_MIN = float(os.getenv("FLEX_SIM_MIN", "0.08"))
FLEX_SIM_MAX = float(os.getenv("FLEX_SIM_MAX", "0.65"))
FLEX_CENTER_MAX = float(os.getenv("FLEX_CENTER_MAX", "0.35"))

# Capture baseline values to scale around
ASSOC_MIN_SIM_BASE = ASSOC_MIN_SIM
ASSOC_MIN_IOU_BASE = ASSOC_MIN_IOU
ASSOC_MAX_CENTER_DIST_BASE = ASSOC_MAX_CENTER_DIST
ID_REASSIGN_IOU_BASE = ID_REASSIGN_IOU


# Auto-Flex docs:
# EMAs tracked per frame:
# - ema_conf: smoothed p30 (30th percentile) of detection confidences for current frame.
# - ema_speed: smoothed mean track speed normalized by frame diagonal.
# - ema_density: smoothed scene density = min(1.0, len(detections)/12.0).
# Toggle with FLEX_ENABLE (1=ON default), adjust smoothing via FLEX_ALPHA. Safety rails via FLEX_* above.
# Enable logs with FLEX_LOG=1 to print a JSON status line every 60 frames, e.g.:
# {"frame":600,"conf_thr":0.18,"assoc_min_iou":0.08,"assoc_min_sim":0.22,"assoc_center":0.12,"reassign_iou":0.10,"ema_conf":0.21,"ema_speed":0.0041,"ema_density":0.46}

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -----------------------------
# ThresholdManager: dynamic, scene-adaptive association gates/weights
# -----------------------------
class ThresholdManager:
    """
    Maintains rolling windows of association stats and produces adaptive gates/weights.
    - update_batch(stats): ingest dict of lists for iou, cdist_frac, emb, hist, conf, accel, size_jump.
    - gates(): returns dict with effective thresholds.
    - weights(): returns dict of association weights.
    - snapshot(): returns a compact dict of final learned gates/weights and internal percentiles.
    All outputs are clamped to safe ranges. Controlled by ADAPTIVE_WEIGHT flag.
    """

    def __init__(self, window: int = 120):
        self.W = int(max(30, window))
        self.buf = {
            "iou": [], "emb": [], "hist": [], "cdist_frac": [], "conf": [], "accel": [], "size_jump": []
        }
        self.last_weights = {"w_iou": ASSOC_W_IOU, "w_emb": ASSOC_W_EMB, "w_hist": ASSOC_W_HIST}
        self.last_gates = {
            "ASSOC_MIN_IOU": ASSOC_MIN_IOU,
            "ASSOC_MAX_CENTER_DIST": ASSOC_MAX_CENTER_DIST,
            "ASSOC_MIN_SIM": ASSOC_MIN_SIM,
            "ID_REASSIGN_IOU": ID_REASSIGN_IOU,
            "ID_REASSIGN_DIST_RATIO": ID_REASSIGN_DIST_RATIO,
            "APPEAR_EMB_MIN_SIM": APPEAR_EMB_MIN_SIM,
            "APPEAR_MIN_SIM": APPEAR_MIN_SIM,
        }

    def _push(self, k: str, vals: list):
        if not isinstance(vals, (list, tuple)):
            return
        b = self.buf.get(k)
        if b is None:
            return
        for v in vals:
            try:
                if v is None:
                    continue
                v = float(v)
                if np.isfinite(v):
                    b.append(v)
            except Exception:
                continue
        # keep tail
        if len(b) > self.W:
            self.buf[k] = b[-self.W:]

    def update_batch(self, stats: dict) -> None:
        if not ADAPTIVE_WEIGHT:
            return
        try:
            for k in ("iou", "emb", "hist", "cdist_frac", "conf", "accel", "size_jump"):
                if k in stats:
                    self._push(k, stats.get(k, []))
        except Exception:
            pass
        # recompute gates/weights lazily here
        self._recompute()

    def _p(self, arr: list, q: float, default: float) -> float:
        try:
            if not arr:
                return default
            return float(np.percentile(np.array(arr, dtype=np.float32), q))
        except Exception:
            return default

    def _recompute(self) -> None:
        if not ADAPTIVE_WEIGHT:
            return
        # Robust deciles
        iou_p10 = self._p(self.buf["iou"], 10, ASSOC_MIN_IOU)
        iou_p50 = self._p(self.buf["iou"], 50, ASSOC_MIN_IOU)
        iou_p90 = self._p(self.buf["iou"], 90, ASSOC_MIN_IOU)
        emb_p10 = self._p(self.buf["emb"], 10, APPEAR_EMB_MIN_SIM)
        emb_p50 = self._p(self.buf["emb"], 50, APPEAR_EMB_MIN_SIM)
        hist_p50 = self._p(self.buf["hist"], 50, APPEAR_MIN_SIM)
        cdist_p50 = self._p(self.buf["cdist_frac"], 50, ASSOC_MAX_CENTER_DIST)
        conf_p75 = self._p(self.buf["conf"], 75, CONFIDENCE_THRESHOLD)
        acc_p90 = self._p(self.buf["accel"], 90, 0.0)
        size_jump_p90 = self._p(self.buf["size_jump"], 90, 2.0)

        # Gates
        assoc_min_iou = ASSOC_MIN_IOU
        assoc_max_center = ASSOC_MAX_CENTER_DIST
        assoc_min_sim = ASSOC_MIN_SIM
        id_reassign_iou = ID_REASSIGN_IOU
        id_reassign_dist = ID_REASSIGN_DIST_RATIO
        appear_emb_min = APPEAR_EMB_MIN_SIM
        appear_hist_min = APPEAR_MIN_SIM

        # Logic: low IoU but high emb -> lower IoU gate slightly, up-weight emb
        if (iou_p50 < 0.25) and (emb_p50 > 0.45):
            assoc_min_iou = float(np.clip(assoc_min_iou * 0.85, FLEX_IOU_MIN, FLEX_IOU_MAX))
        # choppy motion -> expand center gate, cap with GATE_VEL_BOOST
        if acc_p90 > 0.02:  # heuristic in diag units/frame^2
            assoc_max_center = float(np.clip(assoc_max_center * min(GATE_VEL_BOOST, 1.8), 0.02, FLEX_CENTER_MAX))
        # when confident detections dominate -> require higher sim to avoid merges
        if conf_p75 > 0.35:
            assoc_min_sim = float(np.clip(assoc_min_sim * 1.10, FLEX_SIM_MIN, FLEX_SIM_MAX))
            appear_emb_min = float(np.clip(appear_emb_min * 1.05, 0.05, 0.95))
            appear_hist_min = float(np.clip(appear_hist_min * 1.05, 0.05, 0.95))

        # Clamp
        assoc_min_iou = float(np.clip(assoc_min_iou, FLEX_IOU_MIN, FLEX_IOU_MAX))
        assoc_max_center = float(np.clip(max(assoc_max_center, cdist_p50 * 0.8), 0.02, FLEX_CENTER_MAX))
        assoc_min_sim = float(np.clip(assoc_min_sim, FLEX_SIM_MIN, FLEX_SIM_MAX))
        id_reassign_iou = float(np.clip(id_reassign_iou, FLEX_IOU_MIN, 0.50))
        id_reassign_dist = float(np.clip(id_reassign_dist, 0.02, 0.50))
        appear_emb_min = float(np.clip(appear_emb_min, 0.05, 0.95))
        appear_hist_min = float(np.clip(appear_hist_min, 0.05, 0.95))

        self.last_gates = {
            "ASSOC_MIN_IOU": assoc_min_iou,
            "ASSOC_MAX_CENTER_DIST": assoc_max_center,
            "ASSOC_MIN_SIM": assoc_min_sim,
            "ID_REASSIGN_IOU": id_reassign_iou,
            "ID_REASSIGN_DIST_RATIO": id_reassign_dist,
            "APPEAR_EMB_MIN_SIM": appear_emb_min,
            "APPEAR_MIN_SIM": appear_hist_min,
            "SIZE_JUMP_P90": size_jump_p90,
        }

        # Weights adaptation
        w_iou = ASSOC_W_IOU
        w_emb = ASSOC_W_EMB
        w_hist = ASSOC_W_HIST
        if (iou_p50 < 0.25 and emb_p50 > 0.45) or (acc_p90 > 0.02):
            w_emb *= 1.20
            w_iou *= 0.85
        if hist_p50 > 0.6:
            w_hist *= 1.10
        # renorm and clamp
        w_iou = float(np.clip(w_iou, 0.05, 0.90))
        w_emb = float(np.clip(w_emb, 0.05, 0.90))
        w_hist = float(np.clip(w_hist, 0.05, 0.90))
        s = max(1e-6, w_iou + w_emb + w_hist)
        self.last_weights = {"w_iou": w_iou / s, "w_emb": w_emb / s, "w_hist": w_hist / s}

    def gates(self) -> dict:
        return dict(self.last_gates)

    def weights(self) -> dict:
        return dict(self.last_weights)

    def snapshot(self) -> dict:
        return {
            "gates": self.last_gates,
            "weights": self.last_weights,
            "percentiles": {
                "iou_p10": self._p(self.buf["iou"], 10, ASSOC_MIN_IOU),
                "iou_p50": self._p(self.buf["iou"], 50, ASSOC_MIN_IOU),
                "emb_p50": self._p(self.buf["emb"], 50, APPEAR_EMB_MIN_SIM),
                "hist_p50": self._p(self.buf["hist"], 50, APPEAR_MIN_SIM),
                "cdist_p50": self._p(self.buf["cdist_frac"], 50, ASSOC_MAX_CENTER_DIST),
                "conf_p75": self._p(self.buf["conf"], 75, CONFIDENCE_THRESHOLD),
                "acc_p90": self._p(self.buf["accel"], 90, 0.0),
                "size_jump_p90": self._p(self.buf["size_jump"], 90, 2.0),
            }
        }


def is_identity_H(H: Optional[np.ndarray]) -> bool:
    try:
        if H is None:
            return True
        I = np.eye(3, dtype=np.float32)
        return float(np.linalg.norm(H.astype(np.float32) - I)) < 1e-4
    except Exception:
        return True


def write_mot_txt(path: str, tracks_per_frame: Dict[int, List[Tuple[int, float, float, float, float]]], W: int,
                  H: int) -> None:
    # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x,y,z
    lines = []
    for f in sorted(tracks_per_frame.keys()):
        for (tid, x1, y1, x2, y2) in tracks_per_frame[f]:
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            bb_left = max(0.0, min(float(W - 1), x1))
            bb_top = max(0.0, min(float(H - 1), y1))
            lines.append(f"{f},{tid},{bb_left:.2f},{bb_top:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass


def _track_color(tid: int) -> Tuple[int, int, int]:
    hue = (tid * 37) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _bbox_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _bbox_from_center_wh(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


def _bbox_wh(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (max(1.0, x2 - x1), max(1.0, y2 - y1))


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
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
    aa = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    bb = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = aa + bb - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _secondary_nms_detections(detections: "sv.Detections", iou_thr: float = 0.45) -> "sv.Detections":
    if detections.xyxy is None or len(detections) == 0:
        return detections
    boxes = detections.xyxy.astype(np.float32)
    scores = detections.confidence if detections.confidence is not None else np.ones((len(detections),), np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xi1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yi1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xi2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yi2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        iw = np.maximum(0.0, xi2 - xi1)
        ih = np.maximum(0.0, yi2 - yi1)
        inter = iw * ih
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_r - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[np.where(iou <= iou_thr)[0]]
    mask = np.zeros((len(detections),), dtype=bool)
    mask[np.array(keep, dtype=int)] = True
    return detections[mask]


# --- Weighted Box Fusion helper and TTA/Ensemble inference ---
try:
    from typing import Iterable as _Iterable
except Exception:
    _Iterable = None  # type: ignore


def _merge_detections_wbf(dets_list: List["sv.Detections"], iou_thr: float = 0.55,
                          skip_boxes: int = 0) -> "sv.Detections":
    if (dets_list is None) or (len(dets_list) == 0):
        return sv.Detections.empty()
    # Concatenate all boxes/scores/classes
    boxes_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    classes_all: List[np.ndarray] = []
    for d in dets_list:
        if d is None or d.xyxy is None or len(d) == 0:
            continue
        boxes_all.append(d.xyxy.astype(np.float32))
        sc = d.confidence if d.confidence is not None else np.ones((len(d),), dtype=np.float32)
        scores_all.append(sc.astype(np.float32))
        cl = d.class_id if d.class_id is not None else np.zeros((len(d),), dtype=np.int32)
        classes_all.append(cl.astype(np.int32))
    if len(boxes_all) == 0:
        return sv.Detections.empty()
    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    classes = np.concatenate(classes_all, axis=0)
    # Process per-class clusters
    out_boxes: List[List[float]] = []
    out_scores: List[float] = []
    out_classes: List[int] = []
    unique_cls = np.unique(classes)
    for c in unique_cls:
        idx = np.where(classes == c)[0]
        if idx.size == 0:
            continue
        b = boxes[idx]
        s = scores[idx]
        # Sort by score desc
        order = s.argsort()[::-1]
        used = np.zeros((order.size,), dtype=bool)
        for ii, oi in enumerate(order):
            if used[ii]:
                continue
            cluster = [oi]
            used[ii] = True
            # Compare with remaining
            for jj in range(ii + 1, order.size):
                if used[jj]:
                    continue
                j = int(order[jj])
                # IoU with current cluster's best box
                iou = _iou_xyxy(tuple(b[oi].tolist()), tuple(b[j].tolist()))
                if iou >= iou_thr:
                    cluster.append(j)
                    used[jj] = True
            # Fuse cluster
            if len(cluster) == 1:
                bi = b[cluster[0]];
                si = s[cluster[0]]
                out_boxes.append(bi.tolist())
                out_scores.append(float(si))
                out_classes.append(int(c))
            else:
                bw = b[cluster]
                sw = s[cluster]
                wsum = float(np.maximum(1e-6, sw.sum()))
                # Weighted average by scores
                x1 = float((bw[:, 0] * sw).sum() / wsum)
                y1 = float((bw[:, 1] * sw).sum() / wsum)
                x2 = float((bw[:, 2] * sw).sum() / wsum)
                y2 = float((bw[:, 3] * sw).sum() / wsum)
                sc = float(sw.max()) if skip_boxes <= 0 or len(cluster) >= int(skip_boxes) else float(sw.max() * 0.95)
                out_boxes.append([x1, y1, x2, y2])
                out_scores.append(sc)
                out_classes.append(int(c))
    if len(out_boxes) == 0:
        return sv.Detections.empty()
    xyxy = np.array(out_boxes, dtype=np.float32)
    conf = np.array(out_scores, dtype=np.float32)
    clsid = np.array(out_classes, dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=clsid)


def _run_single_model(model: Any, image_bgr: np.ndarray, conf_thr: float, iou_thr: float) -> "sv.Detections":
    res = model(image_bgr, verbose=False, conf=conf_thr, iou=iou_thr)[0]
    det = sv.Detections.from_ultralytics(res)
    return det


def _infer_with_tta(model: Any,
                    frame_bgr: np.ndarray,
                    conf_thr: float,
                    iou_thr: float,
                    flip: bool,
                    scales: List[float],
                    wbf_iou: float,
                    skip_boxes: int) -> "sv.Detections":
    H, W = frame_bgr.shape[:2]
    parts: List[sv.Detections] = []
    for s in scales:
        try:
            if not (isinstance(s, (int, float)) and s > 0):
                continue
            if abs(s - 1.0) < 1e-6:
                im = frame_bgr
                det = _run_single_model(model, im, conf_thr, iou_thr)
                parts.append(det)
                if flip:
                    imf = cv2.flip(im, 1)
                    detf = _run_single_model(model, imf, conf_thr, iou_thr)
                    # unflip boxes
                    if detf.xyxy is not None and len(detf) > 0:
                        xy = detf.xyxy.copy()
                        xy[:, 0], xy[:, 2] = (W - xy[:, 2]), (W - xy[:, 0])
                        detf.xyxy = xy
                    parts.append(detf)
            else:
                newW = max(16, int(round(W * float(s))))
                newH = max(16, int(round(H * float(s))))
                im = cv2.resize(frame_bgr, (newW, newH), interpolation=cv2.INTER_LINEAR)
                det = _run_single_model(model, im, conf_thr, iou_thr)
                # map back to original
                if det.xyxy is not None and len(det) > 0:
                    xy = det.xyxy.copy()
                    xy[:, 0:4:2] = xy[:, 0:4:2] / float(s)
                    xy[:, 1:4:2] = xy[:, 1:4:2] / float(s)
                    det.xyxy = xy
                parts.append(det)
                if flip:
                    imf = cv2.flip(im, 1)
                    detf = _run_single_model(model, imf, conf_thr, iou_thr)
                    if detf.xyxy is not None and len(detf) > 0:
                        xy = detf.xyxy.copy()
                        # unflip in scaled space then rescale back
                        xy[:, 0], xy[:, 2] = (newW - xy[:, 2]), (newW - xy[:, 0])
                        xy[:, 0:4:2] = xy[:, 0:4:2] / float(s)
                        xy[:, 1:4:2] = xy[:, 1:4:2] / float(s)
                        detf.xyxy = xy
                    parts.append(detf)
        except Exception:
            continue
    if len(parts) == 0:
        return sv.Detections.empty()
    return _merge_detections_wbf(parts, iou_thr=wbf_iou, skip_boxes=skip_boxes)


def _infer_ensemble(models: List[Any],
                    frame_bgr: np.ndarray,
                    conf_thr: float,
                    iou_thr: float,
                    use_tta: bool,
                    flip: bool,
                    scales: List[float],
                    wbf_iou: float,
                    skip_boxes: int) -> "sv.Detections":
    all_dets: List[sv.Detections] = []
    for m in models:
        try:
            if use_tta:
                det = _infer_with_tta(m, frame_bgr, conf_thr, iou_thr, flip, scales, wbf_iou, skip_boxes)
            else:
                det = _run_single_model(m, frame_bgr, conf_thr, iou_thr)
            all_dets.append(det)
        except Exception:
            continue
    if len(all_dets) == 0:
        return sv.Detections.empty()
    return _merge_detections_wbf(all_dets, iou_thr=wbf_iou, skip_boxes=skip_boxes)


def _safe_crop(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    xi1 = max(0, int(np.floor(x1)))
    yi1 = max(0, int(np.floor(y1)))
    xi2 = min(w, int(np.ceil(x2)))
    yi2 = min(h, int(np.ceil(y2)))
    if xi2 - xi1 <= 1 or yi2 - yi1 <= 1:
        return None
    return frame[yi1:yi2, xi1:xi2]


def _compute_hsv_hist(frame: np.ndarray, bbox: Tuple[float, float, float, float],
                      mask_roi: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if not APPEAR_ENABLE:
        return None
    roi = _safe_crop(frame, bbox)
    if roi is None:
        return None
    # Optional board-aware focus
    try:
        if HIST_FOCUS == "lower":
            hh = roi.shape[0]
            roi = roi[int(hh * 0.55):, :]
            if mask_roi is not None:
                mask_roi = mask_roi[int(mask_roi.shape[0] * 0.55):, :]
        elif HIST_FOCUS == "center":
            hh, ww = roi.shape[:2]
            y1, y2 = int(hh * 0.25), int(hh * 0.75)
            x1, x2 = int(roi.shape[1] * 0.15), int(roi.shape[1] * 0.85)
            roi = roi[y1:y2, x1:x2]
            if mask_roi is not None:
                my1, my2 = int(mask_roi.shape[0] * 0.25), int(mask_roi.shape[0] * 0.75)
                mx1, mx2 = int(mask_roi.shape[1] * 0.15), int(mask_roi.shape[1] * 0.85)
                mask_roi = mask_roi[my1:my2, mx1:mx2]
    except Exception:
        pass
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_img = None
    if MASK_GUIDED_HIST and mask_roi is not None:
        mask_img = mask_roi
        if mask_img.shape[:2] != hsv.shape[:2]:
            try:
                mask_img = cv2.resize(mask_img, (hsv.shape[1], hsv.shape[0]), interpolation=cv2.INTER_NEAREST)
            except Exception:
                mask_img = None
        if mask_img is not None:
            mask_img = (mask_img * 255).astype(np.uint8)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask_img,
                        [HIST_BINS_H, HIST_BINS_S, HIST_BINS_V],
                        [0, 180, 0, 256, 0, 256])
    if hist is None:
        return None
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def _hist_similarity(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> Optional[float]:
    if h1 is None or h2 is None:
        return None
    try:
        # CORREL gives -1..1, map to 0..1 for convenience
        corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        sim = (corr + 1.0) * 0.5
        # clamp
        if sim < 0.0:
            sim = 0.0
        elif sim > 1.0:
            sim = 1.0
        return sim
    except Exception:
        return None


# -----------------------------
# Kalman helper (constant velocity in image space)
# -----------------------------
class KalmanCV:
    """Constant-velocity Kalman in image space: state [cx, cy, vx, vy]."""

    def __init__(self, q_pos=0.02, q_vel=0.02, r_pos=0.6):
        self.x = np.zeros((4, 1), np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.q_pos, self.q_vel, self.r_pos = float(q_pos), float(q_vel), float(r_pos)

    def predict(self):
        A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        Q = np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel]).astype(np.float32)
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + Q
        return self.x.copy(), self.P.copy()

    def update(self, z_cx, z_cy):
        Hm = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]], dtype=np.float32)
        R = np.diag([self.r_pos, self.r_pos]).astype(np.float32)
        z = np.array([[float(z_cx)], [float(z_cy)]], dtype=np.float32)

        y = z - (Hm @ self.x)
        S = Hm @ self.P @ Hm.T + R
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ Hm) @ self.P
        return self.x.copy(), self.P.copy()


# -----------------------------
# Appearance-aware association (Hungarian with IoU+appearance fusion)
# -----------------------------

def _center_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def appearance_associate(
        state: Dict[int, Dict[str, Any]],
        boxes: List[Tuple[float, float, float, float]],
        det_embs: List[Optional[np.ndarray]],
        det_hists: List[Optional[np.ndarray]],
        frame_idx: int,
        W: int,
        H: int,
        next_tid: int,
        cost_debug_acc: Optional[Dict[str, float]] = None,
        H_cam: Optional[np.ndarray] = None,
        fps: float = 30.0,
        det_confs: Optional[List[float]] = None,
        tm: Optional[ThresholdManager] = None,
        det_masks: Optional[List[Optional[np.ndarray]]] = None,
        det_mask_boxes: Optional[List[Optional[Tuple[float, float, float, float]]]] = None,
        det_vis: Optional[List[float]] = None,
        assoc_gates: Optional[Dict[str, float]] = None,
        assoc_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[int], Dict[int, Dict[str, Any]], int, List[int], int, int]:
    """
    Returns (assigned_ids, new_state, new_next_tid, dropped_pairs, recovered_ids, id_switch_est)
    dropped_pairs: List[Tuple[int, int]] where each pair is (track_id, age).
    Drop policy: tracks removed if time_since_update > BT_BUFFER.
    """
    num_dets = len(boxes)
    if num_dets == 0:
        # Increment time_since_update and drop expired
        dropped = []
        new_state = {}
        for tid, s in state.items():
            miss = int(s.get("time_since_update", 0)) + 1
            s2 = dict(s)
            s2["time_since_update"] = miss
            if miss > BT_BUFFER:
                dropped.append(tid)
            else:
                new_state[tid] = s2
        return [], new_state, next_tid, dropped, 0, 0

    # Prepare current track predictions
    diag = float(np.hypot(W, H))
    tracks = [(tid, s) for tid, s in state.items()]
    T = len(tracks)
    pred_boxes: List[Tuple[float, float, float, float]] = []
    pred_centers: List[Tuple[float, float]] = []
    track_embs: List[Optional[np.ndarray]] = []
    track_hists: List[Optional[np.ndarray]] = []
    for tid, s in tracks:
        cx, cy = s.get("last_center", _bbox_center(s.get("last_bbox", (0, 0, 0, 0))))
        w, h = _bbox_wh(s.get("last_bbox", (0, 0, 1, 1)))
        pcx, pcy = cx, cy  # default no movement

        history = s.get("center_history", deque([], maxlen=LSTM_HISTORY_LEN))

        # Use LSTM if enabled and history is sufficient, otherwise fallback to Kalman/velocity
        if LSTM_MOTION_ENABLE and (traj_predictor is not None):
            try:
                # Assemble history window
                centers_hist = list(s.get("trail", []))
                if not centers_hist or len(centers_hist) < LSTM_MIN_HISTORY:
                    centers_hist = []
                    # fallback from box_hist
                    for bb in s.get("box_hist", [])[-LSTM_HISTORY_LEN:]:
                        centers_hist.append(_bbox_center(tuple(bb)))
                boxes_hist = list(s.get("box_hist", []))[-LSTM_HISTORY_LEN:]
                conf_hist = list(s.get("conf_hist", []))[-LSTM_HISTORY_LEN:]
                # Ensure enough history
                if centers_hist and len(centers_hist) >= LSTM_MIN_HISTORY:
                    # Short appearance sequences for app-memory GRU
                    emb_bank = list(s.get("emb_bank", []))
                    hist_bank = list(s.get("hist_bank", [])) if isinstance(s.get("hist_bank", []), list) else []
                    win = int(LSTM_APPEAR_WINDOW)
                    if win > 0:
                        reid_seq = emb_bank[-win:]
                        hist_seq = hist_bank[-win:]
                    else:
                        reid_seq = []
                        hist_seq = []

                    tw = {
                        'centers': centers_hist[-LSTM_HISTORY_LEN:],
                        'boxes': boxes_hist,
                        'conf': conf_hist,
                        'reid': s.get('emb', None),
                        'hist': s.get('hist', None),
                        'reid_seq': reid_seq,
                        'hist_seq': hist_seq,
                    }

                    pred = traj_predictor.predict(tw)  # type: ignore[operator]
                    d = pred.get('delta', (0.0, 0.0, 0.0, 0.0))
                    cont = float(pred.get('cont', 0.5))
                    cov = pred.get('cov', None)
                    try:
                        if "app_mem" in pred and isinstance(pred["app_mem"], np.ndarray) and pred["app_mem"].size > 0:
                            s["emb_mem"] = pred["app_mem"]
                    except Exception:
                        pass

                    s["_lstm_cont"] = cont
                    if cov is not None:
                        s["_pred_cov"] = cov
                    # apply delta to last known center/size
                    pcx, pcy = float(cx + d[0]), float(cy + d[1])
                    # size adjust in log space
                    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
                        w = float(max(1.0, w * float(np.exp(d[2]))))
                        h = float(max(1.0, h * float(np.exp(d[3]))))
                else:
                    # insufficient history: fallback to Kalman if available
                    raise RuntimeError('insufficient_history')
            except Exception:
                if KALMAN_ENABLE and "kf" in s:
                    try:
                        xpred, Ppred = s["kf"].predict()
                        pcx, pcy = float(xpred[0, 0]), float(xpred[1, 0])
                        s["_pred_cov"] = Ppred
                    except Exception:
                        pass
        elif KALMAN_ENABLE:
            kf = s.get("kf")
            if kf is None:
                kf = KalmanCV(q_pos=KALMAN_Q_POS, q_vel=KALMAN_Q_VEL, r_pos=KALMAN_R_POS)
                kf.x[0, 0], kf.x[1, 0] = float(cx), float(cy)
            xpred, Ppred = kf.predict()
            pcx, pcy = float(xpred[0, 0]), float(xpred[1, 0])
            s["_pred_cov"] = Ppred
            s["kf"] = kf
        else:
            vx, vy = s.get("last_vel", (0.0, 0.0))
            pcx = cx + vx
            pcy = cy + vy

        pb = _bbox_from_center_wh(pcx, pcy, w, h)
        pred_boxes.append(pb)
        pred_centers.append((pcx, pcy))
        track_embs.append(s.get("emb_mem", s.get("emb_surfer", s.get("emb"))))
        track_hists.append(s.get("hist_surfer", s.get("hist")))

    # Warp predictions by camera motion if provided
    pred_boxes_w: List[Tuple[float, float, float, float]] = []
    pred_centers_w: List[Tuple[float, float]] = []
    use_H = (H_cam is not None) and (not is_identity_H(H_cam)) and (warp_box_xyxy is not None) and (
                warp_point is not None)
    for pb, (pcx, pcy) in zip(pred_boxes, pred_centers):
        if use_H:
            wpb = warp_box_xyxy(pb, H_cam)
            wpcx, wpcy = warp_point(pcx, pcy, H_cam)
        else:
            wpb = pb
            wpcx, wpcy = pcx, pcy
        pred_boxes_w.append(wpb)
        pred_centers_w.append((wpcx, wpcy))

    # Build cost matrix (T x D)
    D = num_dets
    cost = np.full((max(1, T), D), 1.0, dtype=np.float32)
    masks = np.zeros((max(1, T), D), dtype=np.uint8)
    emb_sim_mat = np.full((max(1, T), D), -1.0, dtype=np.float32)
    # precompute det centers
    det_centers = [_bbox_center(b) for b in boxes]
    # detection confidences default to 1.0
    if det_confs is None or len(det_confs) != len(boxes):
        det_confs = [1.0] * len(boxes)
    # velocity/acc thresholds
    scale_f = 30.0 / max(1e-3, float(fps))
    v_thr = VEL_K * diag * scale_f
    a_thr = ACC_K * diag * scale_f * scale_f
    # per-track acceleration magnitude based on last and prev vel
    track_vels = []
    track_accs = []
    for tid, s in tracks:
        v = s.get("last_vel", (0.0, 0.0))
        pv = s.get("prev_vel", v)
        track_vels.append(v)
        track_accs.append(float(np.hypot(v[0] - pv[0], v[1] - pv[1])))

    for ti, (_, s) in enumerate(tracks):
        for di, db in enumerate(boxes):
            iou = _iou_xyxy(pred_boxes_w[ti], db) if T > 0 else 0.0
            cdist = _center_distance(pred_centers_w[ti], det_centers[di]) if T > 0 else 1e9

            # Base center gate with size term
            tw, th = _bbox_wh(pred_boxes_w[ti])
            scale_term = 0.5 * float(np.hypot(tw, th))  # px
            # Local gates (with safe fallbacks)
            MIN_IOU = (assoc_gates or {}).get("ASSOC_MIN_IOU", ASSOC_MIN_IOU)
            MIN_SIM = (assoc_gates or {}).get("ASSOC_MIN_SIM", ASSOC_MIN_SIM)
            MAX_CDIST = (assoc_gates or {}).get("ASSOC_MAX_CENTER_DIST", ASSOC_MAX_CENTER_DIST)
            RE_IOU = (assoc_gates or {}).get("ID_REASSIGN_IOU", ID_REASSIGN_IOU)
            RE_DIST = (assoc_gates or {}).get("ID_REASSIGN_DIST_RATIO", ID_REASSIGN_DIST_RATIO)
            EMB_MIN = (assoc_gates or {}).get("APPEAR_EMB_MIN_SIM", APPEAR_EMB_MIN_SIM)
            HIST_MIN = (assoc_gates or {}).get("APPEAR_MIN_SIM", APPEAR_MIN_SIM)
            # Local association weights (with safe fallbacks)
            W_IOU = (assoc_weights or {}).get("w_iou", ASSOC_W_IOU)
            W_EMB = (assoc_weights or {}).get("w_emb", ASSOC_W_EMB)
            W_HIST = (assoc_weights or {}).get("w_hist", ASSOC_W_HIST)
            W_MOT = (assoc_weights or {}).get("w_mot", ASSOC_W_MOT)
            max_dist_px = MAX_CDIST * diag + scale_term

            # Modulate embedding weight by LSTM continuation (trust appearance more when motion is consistent)
            try:
                cont = float(s.get("_lstm_cont", 0.5))
                W_EMB = float(W_EMB) * (0.5 + 0.5 * max(0.0, min(1.0, cont)))
            except Exception:
                pass

            # Optional: uncertainty-aware inflation from covariance (Kalman or LSTM)
            try:
                Ppred = s.get("_pred_cov", None)
                if Ppred is not None:
                    try:
                        Pp = np.array(Ppred, dtype=np.float32)
                        if Pp.ndim == 2 and Pp.shape[0] >= 2 and Pp.shape[1] >= 2:
                            unc = float(np.trace(Pp[:2, :2]))
                        elif Pp.ndim == 1 and Pp.size >= 2:
                            unc = float(Pp[0] + Pp[1])
                        else:
                            unc = 0.0
                        max_dist_px *= (1.0 + min(0.5, 0.03 * unc))
                    except Exception:
                        pass
                # Continuation-aware inflation: if model is confident the track continues, do not over-prune
                contp = float(s.get("_lstm_cont", 0.0) or 0.0)
                if contp > 0.5:
                    max_dist_px *= (1.0 + 0.4 * (contp - 0.5) * 2.0)  # up to +40%
            except Exception:
                pass

            # Velocity-aware expansion for fast movers
            try:
                vmag = float(np.hypot(track_vels[ti][0], track_vels[ti][1]))
            except Exception:
                vmag = 0.0
            if vmag > v_thr:
                max_dist_px *= GATE_VEL_BOOST

            # Compute appearance early so it can influence the gate
            _tid0 = tracks[ti][0] if ti < len(tracks) else None
            t_emb = state.get(_tid0, {}).get("emb_ema") if (_tid0 is not None) else None
            if t_emb is None:
                t_emb = track_embs[ti]
            t_hist = state.get(_tid0, {}).get("hist_ema") if (_tid0 is not None) else None
            if t_hist is None:
                t_hist = track_hists[ti]
            # Part-based readiness: use surfer features for now
            t_emb_surfer = s.get("emb_surfer", t_emb)
            t_hist_surfer = s.get("hist_surfer", t_hist)
            det_emb_surfer = det_embs[di] if di < len(det_embs) else None
            det_hist_surfer = det_hists[di] if di < len(det_hists) else None

            # Feature-bank aware embedding similarity: max cosine over recent embeddings
            emb_sim = None
            if APPEAR_EMB_ENABLE:
                det_vec = det_emb_surfer
                if det_vec is not None:
                    bank = s.get("emb_bank", None)
                    if isinstance(bank, list) and len(bank) > 0:
                        best_cs = -1.0
                        for vb in bank:
                            cs = _cosine_sim(vb, det_vec)
                            if cs is not None and cs > best_cs:
                                best_cs = cs
                        emb_sim = best_cs if best_cs >= -0.5 else None
                    else:
                        emb_sim = _cosine_sim(t_emb_surfer, det_vec)
            hist_sim = _hist_similarity(t_hist_surfer, det_hist_surfer) if APPEAR_ENABLE else None
            # Placeholder for future weighted part similarity:
            # emb_sim_board = _cosine_sim(s.get("emb_board"), det_embs_board[di])
            # emb_sim = 0.7 * (emb_sim or 0.0) + 0.3 * (emb_sim_board or 0.0)
            try:
                emb_sim_mat[ti, di] = float(emb_sim) if (emb_sim is not None and np.isfinite(emb_sim)) else -1.0
            except Exception:
                pass

            # Mask IoU blending
            iou_soft = iou
            if SEG_ENABLE and (det_masks is not None) and (det_mask_boxes is not None):
                try:
                    track_mask = s.get("mask", None)
                    track_mask_box = s.get("mask_box", None)
                    dmask = det_masks[di] if di < len(det_masks) else None
                    if (track_mask is not None) and (dmask is not None):
                        miou = _mask_iou(track_mask, track_mask_box, dmask, db)
                        if (miou is not None) and np.isfinite(miou):
                            iou_soft = float(ALPHA_MASK_IOU * iou + (1.0 - ALPHA_MASK_IOU) * float(miou))
                except Exception:
                    pass

            # Anisotropic gate (elliptical) optionally replaces isotropic center distance
            dx = det_centers[di][0] - pred_centers_w[ti][0]
            dy = det_centers[di][1] - pred_centers_w[ti][1]
            if ANISO_GATE_ENABLE:
                diag_local = float(np.hypot(W, H))
                sx = max(4.0, GATE_SIGMA_X_FRAC * diag_local)
                sy = max(4.0, GATE_SIGMA_Y_FRAC * diag_local)
                ell = (dx * dx) / (sx * sx) + (dy * dy) / (sy * sy)
                center_ok = (ell <= 1.0)
            else:
                center_ok = (cdist <= max_dist_px)
            # gate_keep computed from geometry (use blended IoU) and center gate
            gate_keep = ((iou_soft >= ASSOC_MIN_IOU) and center_ok)
            # tighten appearance escape: require BOTH strong emb and hist, matured track, and confident det
            if APPEAR_ESCAPE_ENABLE:
                strong_appear = (
                        (emb_sim is not None and emb_sim >= max(APPEAR_ESCAPE_COS, 0.65)) and
                        (hist_sim is not None and hist_sim >= max(APPEAR_ESCAPE_HIST, 0.72)) and
                        (int(s.get("age", 0)) >= 5) and
                        (det_confs[di] >= 0.50)
                )
                gate_keep = gate_keep or (strong_appear and (cdist <= ESCAPE_CENTER_MULT * max_dist_px))

            if not gate_keep:
                masks[ti, di] = 0
                cost[ti, di] = 1.0
                continue
            # Adaptive weighting per pair
            w_iou = W_IOU
            w_emb = W_EMB
            w_hist = W_HIST
            if ADAPTIVE_WEIGHT:
                acc_mag = track_accs[ti] if ti < len(track_accs) else 0.0
                if (iou_soft < LOW_IOU_THRESH) or (acc_mag > a_thr):
                    w_emb *= 1.3
                    w_iou *= 0.7
                elif (iou_soft > HIGH_IOU_THRESH) and (acc_mag < 0.5 * a_thr):
                    w_iou *= 1.2
                    w_emb *= 0.8
                # clamp and renorm across all heads
                w_iou = float(np.clip(w_iou, 0.05, 0.90))
                w_emb = float(np.clip(w_emb, 0.05, 0.90))
                w_hist = float(np.clip(w_hist, 0.05, 0.90))
                W_MOT = float(np.clip(W_MOT, 0.05, 0.90))
                _wsum = max(1e-6, (w_iou + w_emb + w_hist + W_MOT))
                w_iou, w_emb, w_hist, W_MOT = w_iou / _wsum, w_emb / _wsum, w_hist / _wsum, W_MOT / _wsum
            # Motion similarity under uncertainty-aware gate
            if ANISO_GATE_ENABLE:
                mot_sim = float(
                    np.exp(-0.5 * max(0.0, (dx * dx) / (max(1.0, sx * sx)) + (dy * dy) / (max(1.0, sy * sy)))))
            else:
                denom = max(1e-6, max_dist_px)
                mot_sim = float(np.exp(-0.5 * (cdist / denom) * (cdist / denom)))
            sim = 0.0
            sim += w_iou * iou_soft
            sim += W_MOT * mot_sim
            if emb_sim is not None:
                sim += w_emb * emb_sim
            if hist_sim is not None:
                sim += w_hist * hist_sim
            sim = max(0.0, min(1.0, sim))

            # hard gate: do not allow ambiguous low-sim pairs into the assignment
            if sim < ASSOC_MIN_SIM:
                masks[ti, di] = 0
                cost[ti, di] = 1.0
                continue

            cost[ti, di] = 1.0 - sim
            masks[ti, di] = 1
            if cost_debug_acc is not None:
                cost_debug_acc["iou_sum"] = cost_debug_acc.get("iou_sum", 0.0) + iou_soft
                cost_debug_acc["emb_sum"] = cost_debug_acc.get("emb_sum", 0.0) + (
                    emb_sim if emb_sim is not None else 0.0)
                cost_debug_acc["hist_sum"] = cost_debug_acc.get("hist_sum", 0.0) + (
                    hist_sim if hist_sim is not None else 0.0)
                cost_debug_acc["pairs"] = cost_debug_acc.get("pairs", 0.0) + 1.0

    # Solve assignment
    matches: List[Tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        if T > 0 and D > 0:
            row_ind, col_ind = linear_sum_assignment(cost[:T, :D])
            for r, c in zip(row_ind, col_ind):
                if masks[r, c] == 1:
                    matches.append((r, c))
    except Exception:
        # Greedy fallback: pick lowest cost pairs
        cand = []
        for r in range(T):
            for c in range(D):
                if masks[r, c] == 1:
                    cand.append((cost[r, c], r, c))
        cand.sort()
        used_r = set();
        used_c = set()
        for v, r, c in cand:
            if r in used_r or c in used_c:
                continue
            matches.append((r, c))
            used_r.add(r);
            used_c.add(c)

    # --- Anti-switch guard (local 2x2 swap veto) ---
    # If two mature tracks appear to swap their detections crosswise, flip them back to
    # preserve temporal continuity. Uses previous centers and current det centers.
    if len(matches) >= 2 and T > 1 and D > 1:
        prev_centers = [pred_centers_w[ti] for ti in range(T)]
        det_centers_arr = [((boxes[c][0] + boxes[c][2]) * 0.5,
                            (boxes[c][1] + boxes[c][3]) * 0.5) for c in range(D)]
        fixed, used = [], set()
        for i in range(len(matches)):
            if i in used:
                continue
            r1, c1 = matches[i]
            swapped = False
            for j in range(i + 1, len(matches)):
                if j in used:
                    continue
                r2, c2 = matches[j]

                # Direct vs. cross sum of distances
                d_direct = _center_distance(prev_centers[r1], det_centers_arr[c1]) + \
                           _center_distance(prev_centers[r2], det_centers_arr[c2])
                d_cross = _center_distance(prev_centers[r1], det_centers_arr[c2]) + \
                          _center_distance(prev_centers[r2], det_centers_arr[c1])

                # Only veto when both tracks are mature and just got updated last frame
                a = state.get(tracks[r1][0], {})
                b = state.get(tracks[r2][0], {})
                a_mature = int(a.get("age", 0)) >= 5 and int(a.get("time_since_update", 0)) == 0
                b_mature = int(b.get("age", 0)) >= 5 and int(b.get("time_since_update", 0)) == 0

                # 4px hysteresis to avoid flapping
                if a_mature and b_mature and (d_cross + 4.0 < d_direct):
                    fixed.append((r1, c2))
                    fixed.append((r2, c1))
                    used.add(i);
                    used.add(j)
                    swapped = True
                    break
            if not swapped:
                fixed.append(matches[i])
        matches = fixed
    # --- /Anti-switch guard ---

    assigned_ids = [-1] * D
    used_d = set()
    used_t = set()
    new_state: Dict[int, Dict[str, Any]] = {}
    recovered_ids = 0
    id_switch_est = 0

    # Helper: heuristic switch estimate: when new track is born near a recently lost track with IoU>=0.3 and appearance high
    def _likely_switch(new_box: Tuple[float, float, float, float], det_emb: Optional[np.ndarray],
                       det_hist: Optional[np.ndarray]) -> bool:
        # find best recent lost track
        best = None
        best_score = 0.0
        for tid, s in state.items():
            miss = frame_idx - int(s.get("last_frame", -999999))
            if miss < 1 or miss > min(60, BT_BUFFER):
                continue
            gb = s.get("last_bbox", None)
            if gb is None:
                continue
            # Scale sanity: avoid counting if size jumps unrealistically
            tw, th = _bbox_wh(s.get("last_bbox", (0, 0, 1, 1)))
            dw, dh = _bbox_wh(new_box)
            if max(dw, dh) / max(1.0, max(tw, th)) > 2.5:
                continue
            iou = _iou_xyxy(gb, new_box)
            if iou < 0.30:
                continue
            emb_sim = _cosine_sim(s.get("emb"), det_emb) if APPEAR_EMB_ENABLE else None
            hist_sim = _hist_similarity(s.get("hist"), det_hist) if APPEAR_ENABLE else None
            score = (emb_sim or 0.0) + 0.3 * (hist_sim or 0.0)
            if score > best_score:
                best_score = score
                best = tid
        return best is not None and best_score >= 0.6

    # Apply matches
    for r, c in matches:
        tid, s = tracks[r]
        db = boxes[c]
        dcx, dcy = _bbox_center(db)
        pcx, pcy = s.get("last_center", (dcx, dcy))
        # instantaneous velocity and EMA update
        vel_inst = (dcx - pcx, dcy - pcy)
        prev_vel = s.get("last_vel", (0.0, 0.0))
        alpha_v = 0.6
        vel = (prev_vel[0] * (1 - alpha_v) + vel_inst[0] * alpha_v,
               prev_vel[1] * (1 - alpha_v) + vel_inst[1] * alpha_v)
        # recompute iou for gating
        try:
            pbw = pred_boxes_w[r]
        except Exception:
            pbw = s.get("last_bbox", db)
        iou_here = _iou_xyxy(pbw, db)
        conf_here = det_confs[c] if det_confs is not None and c < len(det_confs) else 1.0
        slow_ema = (conf_here < (CONFIDENCE_THRESHOLD * 1.2)) or (iou_here < LOW_IOU_THRESH)

        # Avoid learning appearance while boxed-in (prevents drift/steal)
        crowded = False
        for kk in range(len(boxes)):
            if kk == c:
                continue
            if _iou_xyxy(db, boxes[kk]) >= 0.25:
                crowded = True
                break

        can_update_appear = (
                (iou_here >= UPDATE_EMB_IOU) and
                (conf_here >= UPDATE_EMB_CONF) and
                (int(s.get("age", 0)) >= UPDATE_EMB_MIN_AGE) and
                (not crowded)
        )

        # take mask from detection if available
        t_mask = None;
        t_mask_box = None;
        t_vis = None
        if SEG_ENABLE and (det_masks is not None) and (det_mask_boxes is not None):
            if 0 <= c < len(det_masks) and det_masks[c] is not None:
                t_mask = det_masks[c]
                t_mask_box = det_mask_boxes[c]
                if (det_vis is not None) and (0 <= c < len(det_vis)):
                    try:
                        t_vis = float(det_vis[c])
                    except Exception:
                        t_vis = None
        # visibility-gated appearance updates
        vis_ok = True
        if SEG_ENABLE and (t_vis is not None):
            vis_ok = (t_vis >= VIS_MIN_FOR_UPDATE)
        # EMA updates for appearance with gating
        hist = s.get("hist")
        if APPEAR_ENABLE and can_update_appear and vis_ok:
            h_now = det_hists[c]
            if h_now is not None:
                if hist is not None and isinstance(hist, np.ndarray) and hist.shape == h_now.shape:
                    alpha_h = APPEAR_ALPHA * (0.5 if slow_ema else 1.0)
                    hist = (1.0 - alpha_h) * hist + alpha_h * h_now
                    try:
                        cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
                    except Exception:
                        pass
                else:
                    hist = h_now

        # Maintain a short history bank for HSV histograms (mirrors emb_bank)
        hist_bank = list(s.get("hist_bank", [])) if isinstance(s.get("hist_bank", []), list) else []
        try:
            if hist is not None and isinstance(hist, np.ndarray) and hist.size > 0:
                # avoid duplicating near-identical consecutive entries
                if (not hist_bank) or (float(np.dot(hist_bank[-1].ravel(), hist.ravel())) < 0.995):
                    hist_bank.append(hist.astype(np.float32))
                    # keep last N similar to emb bank length
                    hist_bank = hist_bank[-max(8, min(len(hist_bank), EMB_BANK_MAX)):]
        except Exception:
            pass

        emb = s.get("emb")
        emb_bank = list(s.get("emb_bank", [])) if isinstance(s.get("emb_bank", []), list) else []
        emb_init = s.get("emb_init", None)
        if APPEAR_EMB_ENABLE and can_update_appear and vis_ok:
            e_now = det_embs[c]
            if e_now is not None:
                # initialize emb_init once
                if emb_init is None:
                    emb_init = e_now
                # only update EMA if shapes agree
                if emb is not None and isinstance(emb, np.ndarray) and emb.shape == e_now.shape:
                    alpha_e = APPEAR_EMB_ALPHA * (0.5 if slow_ema else 1.0)
                    emb = (1.0 - alpha_e) * emb + alpha_e * e_now
                    n = np.linalg.norm(emb)
                    if n > 1e-12:
                        emb = emb / n
                else:
                    emb = e_now
                # controlled bank update: skip if too dissimilar to current avg or to init (drift guard)
                cs_avg = _cosine_sim(e_now, emb)
                cs_init = _cosine_sim(e_now, emb_init)
                if (cs_avg is None) or (not np.isfinite(cs_avg)):
                    cs_avg = -1.0
                if (cs_init is None) or (not np.isfinite(cs_init)):
                    cs_init = -1.0
                if (cs_avg >= EMB_BANK_MIN_SIM_UPDATE) and (cs_init >= EMB_BANK_DRIFT_INIT_GUARD):
                    try:
                        emb_bank.append(e_now)
                        if len(emb_bank) > EMB_BANK_MAX:
                            emb_bank = emb_bank[-EMB_BANK_MAX:]
                    except Exception:
                        pass
        # Count recovered if this track was missing
        if int(s.get("time_since_update", 0)) > 0:
            recovered_ids += 1
        # update short histories for LSTM
        bh = list(s.get("box_hist", []));
        bh.append(db)
        if len(bh) > LSTM_HISTORY_LEN:
            bh = bh[-LSTM_HISTORY_LEN:]
        ch = list(s.get("conf_hist", []));
        ch.append(conf_here)
        if len(ch) > LSTM_HISTORY_LEN:
            ch = ch[-LSTM_HISTORY_LEN:]
        new_state[tid] = {
            "last_bbox": db,
            "last_center": (dcx, dcy),
            "prev_vel": prev_vel,
            "last_vel": vel,
            "last_frame": frame_idx,
            "age": int(s.get("age", 0)) + 1,
            "hits": int(s.get("hits", 0)) + 1,
            "time_since_update": 0,
            "emb": emb,
            "emb_init": emb_init,
            "emb_bank": emb_bank,
            "hist": hist,
            "mask": t_mask,
            "mask_box": t_mask_box,
            "mask_age": 0,
            "vis": (float(t_vis) if (t_vis is not None) else float(s.get("vis", 1.0))),
            "box_hist": bh,
            "conf_hist": ch,
            "hist_bank": hist_bank,

        }
        assigned_ids[c] = int(tid)
        used_d.add(c);
        used_t.add(r)

    # Unmatched tracks: age and drop if expired
    dropped_pairs: List[Tuple[int, int]] = []
    for ti, (tid, s) in enumerate(tracks):
        if ti in used_t:
            continue
        miss = int(s.get("time_since_update", 0)) + 1
        if miss > BT_BUFFER:
            # archive this track for potential long-gap re-id
            try:
                if ARCHIVE_ENABLE:
                    age_arch = int(s.get("age", 0))
                    bank = s.get("emb_bank", [])
                    emb_arch = []
                    if isinstance(bank, list) and len(bank) > 0:
                        emb_arch = bank[-min(len(bank), 5):]
                    else:
                        if isinstance(s.get("emb", None), np.ndarray):
                            emb_arch = [s.get("emb")]
                    rec = {
                        "tid": int(tid),
                        "embs": emb_arch,
                        "last_frame": int(s.get("last_frame", frame_idx - 1)),
                        "age": age_arch,
                    }
                    # prune old/oversize archive first
                    try:
                        if len(TRACK_ARCHIVE) > 0:
                            TRACK_ARCHIVE[:] = [r for r in TRACK_ARCHIVE if
                                                (frame_idx - int(r.get("last_frame", -999999)) <= ARCHIVE_TTL)]
                    except Exception:
                        pass
                    TRACK_ARCHIVE.append(rec)
                    if len(TRACK_ARCHIVE) > ARCHIVE_MAX:
                        # drop oldest
                        TRACK_ARCHIVE.pop(0)
            except Exception:
                pass
            dropped_pairs.append((int(tid), int(s.get("age", 0))))
            continue
        # keep in state as missing
        new_state[int(tid)] = {
            **s,
            "time_since_update": miss,
            "age": int(s.get("age", 0)) + 1,
        }

    # Unmatched detections: try to reclaim a recent lost ID (AA stitch), else create new
    used_reclaims = set()
    for di, db in enumerate(boxes):
        if di in used_d:
            continue

        reclaimed = False
        if AA_STITCH_ENABLE:
            best_tid = -1
            best_score = 0.0
            dcx, dcy = _bbox_center(db)
            # consider recent ghosts
            for tid, s in state.items():
                miss = frame_idx - int(s.get("last_frame", -999999))
                if miss < 1 or miss > min(REASSIGN_MAX_MISS, ID_REASSIGN_WINDOW):
                    continue
                if tid in used_reclaims:
                    continue

                # predict ghost forward (your existing pcx/pcy/gb code)
                cx, cy = s.get("last_center", (dcx, dcy))
                vx, vy = s.get("last_vel", (0.0, 0.0))
                pcx = cx + vx * float(miss)
                pcy = cy + vy * float(miss)
                gw, gh = _bbox_wh(s.get("last_bbox", (0, 0, 1, 1)))
                gb = _bbox_from_center_wh(pcx, pcy, gw, gh)
                if H_cam is not None and not is_identity_H(H_cam) and warp_box_xyxy is not None:
                    gb = warp_box_xyxy(gb, H_cam)

                iou_g = _iou_xyxy(gb, db)
                dist_g = _center_distance((pcx, pcy), (dcx, dcy))

                ghost_emb = s.get("emb") if APPEAR_EMB_ENABLE else None
                ghost_hist = s.get("hist") if APPEAR_ENABLE else None
                det_emb = det_embs[di] if di < len(det_embs) else None
                det_hist = det_hists[di] if di < len(det_hists) else None

                emb_sim = _cosine_sim(det_emb, ghost_emb)
                hist_sim = _hist_similarity(det_hist, ghost_hist)

                emb_ok = (emb_sim is not None and emb_sim >= max(APPEAR_EMB_MIN_SIM, 0.62))
                hist_ok = (hist_sim is not None and hist_sim >= max(APPEAR_MIN_SIM, 0.75))
                appear_ok = emb_ok or (hist_ok and iou_g >= max(ID_REASSIGN_IOU, 0.20))
                geom_ok = (iou_g >= max(ID_REASSIGN_IOU, 0.20)) or (dist_g <= (ID_REASSIGN_DIST_RATIO * diag))

                if appear_ok and geom_ok:
                    score = (emb_sim or 0.0) + 0.30 * (hist_sim or 0.0) + 0.15 * max(0.0, iou_g)
                    if score > best_score:
                        best_score = score
                        best_tid = int(tid)

            if best_tid > 0:
                dcx, dcy = _bbox_center(db)
                new_state[best_tid] = {
                    "last_bbox": db,
                    "last_center": (dcx, dcy),
                    "prev_vel": state[best_tid].get("last_vel", (0.0, 0.0)),
                    "last_vel": state[best_tid].get("last_vel", (0.0, 0.0)),
                    "last_frame": frame_idx,
                    "age": int(state[best_tid].get("age", 0)) + 1,
                    "hits": int(state[best_tid].get("hits", 0)) + 1,
                    "time_since_update": 0,
                    "emb": det_embs[di] if di < len(det_embs) else state[best_tid].get("emb"),
                    "hist": det_hists[di] if di < len(det_hists) else state[best_tid].get("hist"),
                }
                assigned_ids[di] = best_tid
                used_reclaims.add(best_tid)
                recovered_ids += 1
                reclaimed = True

        if reclaimed:
            continue

        # Try long-gap ID reuse from archive before spawning new
        reused = False
        if ARCHIVE_ENABLE and di < len(det_embs) and det_embs[di] is not None and len(TRACK_ARCHIVE) > 0:
            try:
                best_tid = -1
                best_sim = 0.0
                vec = det_embs[di]
                # prune expired
                try:
                    if len(TRACK_ARCHIVE) > 0:
                        TRACK_ARCHIVE[:] = [r for r in TRACK_ARCHIVE if
                                            (frame_idx - int(r.get("last_frame", -999999)) <= ARCHIVE_TTL)]
                except Exception:
                    pass
                for k, rec in enumerate(TRACK_ARCHIVE):
                    embs = rec.get("embs", [])
                    if not isinstance(embs, list) or len(embs) == 0:
                        continue
                    # skip if tid already revived this frame
                    if int(rec.get("tid", -1)) in new_state:
                        continue
                    # compute max cosine to stored bank
                    local_best = -1.0
                    for vb in embs:
                        cs = _cosine_sim(vec, vb)
                        if cs is not None and cs > local_best:
                            local_best = cs
                    if local_best is not None and local_best >= ARCHIVE_SIM_THR and local_best > best_sim:
                        best_sim = float(local_best)
                        best_tid = int(rec.get("tid"))
                        best_k = k
                if best_tid > 0:
                    dcx, dcy = _bbox_center(db)
                    new_state[best_tid] = {
                        "last_bbox": db,
                        "last_center": (dcx, dcy),
                        "prev_vel": (0.0, 0.0),
                        "last_vel": (0.0, 0.0),
                        "last_frame": frame_idx,
                        "age": int(
                            next((r.get("age", 0) for r in TRACK_ARCHIVE if int(r.get("tid", -1)) == best_tid), 0)) + 1,
                        "hits": 1,
                        "time_since_update": 0,
                        "emb": det_embs[di],
                        "emb_init": det_embs[di],
                        "emb_bank": [det_embs[di]],
                        "hist": det_hists[di] if di < len(det_hists) else None,
                    }
                    assigned_ids[di] = best_tid
                    # remove revived entry from archive
                    try:
                        TRACK_ARCHIVE.pop(best_k)
                    except Exception:
                        pass
                    reused = True
            except Exception:
                reused = False
        if reused:
            continue

        # default: spawn a new ID
        if _likely_switch(db, det_embs[di] if di < len(det_embs) else None,
                          det_hists[di] if di < len(det_hists) else None):
            id_switch_est += 1
        dcx, dcy = _bbox_center(db)
        emb_spawn = det_embs[di] if di < len(det_embs) else None
        hist_spawn = det_hists[di] if di < len(det_hists) else None
        new_state[next_tid] = {
            "last_bbox": db,
            "last_center": (dcx, dcy),
            "last_vel": (0.0, 0.0),
            "last_frame": frame_idx,
            "age": 1,
            "hits": 1,
            "time_since_update": 0,
            "emb": emb_spawn,
            "emb_init": emb_spawn,
            "emb_bank": ([emb_spawn] if isinstance(emb_spawn, np.ndarray) else []),
            "hist": hist_spawn,
        }
        assigned_ids[di] = int(next_tid)
        next_tid += 1

    return assigned_ids, new_state, next_tid, dropped_pairs, recovered_ids, id_switch_est


# -------------
# Slicer helper
# -------------
from typing import Callable as _Callable


def _call_slicer_with_timeout(slicer: Any, frame: np.ndarray, timeout_s: float) -> Tuple[Optional[Any], str]:
    """
    Calls slicer(frame) with a timeout.
    Returns (detections, status) where status is one of: 'ok', 'timeout', 'error'.
    Implementation detail: uses a temporary ThreadPoolExecutor and shuts it down with wait=False
    on timeout to avoid blocking the main thread while the slicer call finishes in the background.
    """
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
        def _job():
            return slicer(frame)

        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(_job)
        try:
            out = fut.result(timeout=timeout_s)
            # Normal path: retrieve result then cleanly shutdown
            ex.shutdown(wait=True, cancel_futures=True)
            return out, 'ok'
        except _Timeout:
            # Timed out: do not wait for worker thread to finish
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            return None, 'timeout'
    except KeyboardInterrupt:
        # propagate to allow clean shutdown
        raise
    except Exception:
        return None, 'error'


# -----------------------------
# Deep ReID: model init, preprocess, embedding & similarity
# -----------------------------
_REID_MODEL = None
_REID_TF = None
_REID_DEV = None
_REID_INPUT_SIZE = (256, 128)  # H, W typical for person/osnet; surfers are vertical-ish


def _select_device_from_env() -> str:
    if REID_DEVICE == "cpu":
        return "cpu"
    if REID_DEVICE == "cuda":
        return "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    if REID_DEVICE == "mps":
        try:
            return "mps" if (
                        HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
        except Exception:
            return "cpu"
    # auto
    try:
        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _init_reid():
    global _REID_MODEL, _REID_TF, _REID_DEV, _REID_PLUG
    if not (HAS_TORCH and APPEAR_EMB_ENABLE):
        return False
    if _REID_MODEL is not None or ('_REID_PLUG' in globals() and _REID_PLUG is not None):
        return True
    # Try pluggable first
    try:
        if _PluggableReID is not None:
            try:
                _REID_PLUG = _PluggableReID(backend=REID_BACKBONE, device=REID_DEVICE, fp16=REID_FP16)
                return True
            except Exception as e:
                logger.warning("[sv-pipeline] warning: ReID backend failed to load, falling back to OSNet")
    except Exception:
        pass
    try:
        dev_str = _select_device_from_env()
        dev = torch.device(dev_str)
        model = None
        tf = None
        # Prefer OSNet via torchreid if available
        if REID_BACKBONE.lower() == "osnet":
            try:
                import torchreid  # type: ignore
                model = torchreid.models.build_model('osnet_x1_0', num_classes=1, pretrained=True)  # type: ignore
                model.classifier = torch.nn.Identity()  # type: ignore[attr-defined]
                tf = _T.Compose([
                    _T.ToTensor(),
                    _T.Resize((_REID_INPUT_SIZE[0], _REID_INPUT_SIZE[1])),
                    _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except Exception:
                # Try torchvision osnet if present in this environment
                try:
                    osnet = getattr(torchvision.models, 'osnet_x1_0', None)
                    if osnet is not None:
                        model = osnet(pretrained=True)
                        if hasattr(model, 'classifier'):
                            model.classifier = torch.nn.Identity()  # type: ignore[attr-defined]
                        elif hasattr(model, 'fc'):
                            model.fc = torch.nn.Identity()
                        tf = _T.Compose([
                            _T.ToTensor(),
                            _T.Resize((_REID_INPUT_SIZE[0], _REID_INPUT_SIZE[1])),
                            _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                except Exception:
                    model = None
        # Fallback to ResNet-50
        if model is None:
            try:
                try:
                    weights = torchvision.models.ResNet50_Weights.DEFAULT  # type: ignore[attr-defined]
                    model = torchvision.models.resnet50(weights=weights)  # type: ignore[arg-type]
                    tf = weights.transforms()  # type: ignore[assignment]
                except Exception:
                    model = torchvision.models.resnet50(pretrained=True)
                    tf = _T.Compose([
                        _T.ToTensor(),
                        _T.Resize((224, 224)),
                        _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                model.fc = torch.nn.Identity()
            except Exception:
                model = None
        if model is None or tf is None:
            return False
        model.eval()
        model.to(dev)
        _REID_MODEL = model
        _REID_TF = tf
        _REID_DEV = dev
        return True
    except Exception:
        _REID_MODEL = None
        _REID_TF = None
        _REID_DEV = None
        return False


def _init_reid_builtin_only():
    """Initialize only the builtin ReID model (OSNet/ResNet), bypassing any pluggable backends."""
    global _REID_MODEL, _REID_TF, _REID_DEV
    if not HAS_TORCH:
        return False
    try:
        dev_str = _select_device_from_env()
        dev = torch.device(dev_str)
        model = None
        tf = None
        # Try OSNet via torchreid
        try:
            import torchreid  # type: ignore
            model = torchreid.models.build_model('osnet_x1_0', num_classes=1, pretrained=True)  # type: ignore
            if hasattr(model, 'classifier'):
                model.classifier = torch.nn.Identity()
            tf = _T.Compose([
                _T.ToTensor(),
                _T.Resize((_REID_INPUT_SIZE[0], _REID_INPUT_SIZE[1])),
                _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception:
            # Try torchvision osnet
            try:
                osnet = getattr(torchvision.models, 'osnet_x1_0', None)
                if osnet is not None:
                    model = osnet(pretrained=True)
                    if hasattr(model, 'classifier'):
                        model.classifier = torch.nn.Identity()
                    elif hasattr(model, 'fc'):
                        model.fc = torch.nn.Identity()
                    tf = _T.Compose([
                        _T.ToTensor(),
                        _T.Resize((_REID_INPUT_SIZE[0], _REID_INPUT_SIZE[1])),
                        _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
            except Exception:
                model = None
        # ResNet-50 fallback
        if model is None:
            try:
                try:
                    weights = torchvision.models.ResNet50_Weights.DEFAULT  # type: ignore[attr-defined]
                    model = torchvision.models.resnet50(weights=weights)  # type: ignore[arg-type]
                    tf = weights.transforms()  # type: ignore[assignment]
                except Exception:
                    model = torchvision.models.resnet50(pretrained=True)
                    tf = _T.Compose([
                        _T.ToTensor(),
                        _T.Resize((224, 224)),
                        _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                model.fc = torch.nn.Identity()
            except Exception:
                model = None
        if model is None or tf is None:
            return False
        model.eval();
        model.to(dev)
        _REID_MODEL = model
        _REID_TF = tf
        _REID_DEV = dev
        return True
    except Exception:
        _REID_MODEL = None
        _REID_TF = None
        _REID_DEV = None
        return False


def _pad_bbox(b: Tuple[float, float, float, float], pad_ratio: float, W: int, H: int) -> Tuple[
    float, float, float, float]:
    x1, y1, x2, y2 = b
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = w * pad_ratio
    py = h * pad_ratio
    nx1 = max(0.0, x1 - px)
    ny1 = max(0.0, y1 - py)
    nx2 = min(float(W - 1), x2 + px)
    ny2 = min(float(H - 1), y2 + py)
    return (nx1, ny1, nx2, ny2)


# ---- Segmentation helpers ----

def _extract_yolo_seg_masks(res: Any, frame_bgr: np.ndarray, det_xyxy: List[Tuple[float, float, float, float]]):
    """
    Inputs:
      - res: Ultralytics result object (may contain .masks)
      - frame_bgr: HxWx3 np.uint8
      - det_xyxy: List[Tuple[float,float,float,float]] for this frame
    Returns:
      - det_masks: List[np.ndarray[H,W] (uint8 0/1)] aligned to the detection ROI
      - det_mask_boxes: List[Tuple[...]] same length as det_xyxy (box in frame coords)
      - det_vis: List[float] in [0,1] visibility (mean mask in the det box)
      OR (None, None, None) if no masks.
    """
    try:
        Hh, Ww = frame_bgr.shape[:2]
        have_yolo_masks = (res is not None and hasattr(res, "masks") and (res.masks is not None))
        if have_yolo_masks:
            mdata = getattr(res.masks, "data", None)
            if mdata is not None:
                try:
                    mm = mdata.detach().cpu().numpy()
                except Exception:
                    mm = np.asarray(mdata)
                if mm is not None and mm.ndim >= 2:
                    # Resize each mask to full frame and then crop per detection
                    det_masks_full: List[np.ndarray] = []
                    for i in range(mm.shape[0]):
                        m = (mm[i] > 0.5).astype("uint8")
                        if m.shape[0] != Hh or m.shape[1] != Ww:
                            m = cv2.resize(m, (Ww, Hh), interpolation=cv2.INTER_NEAREST)
                        det_masks_full.append(m)
                    n = len(det_xyxy)
                    if det_masks_full and len(det_masks_full) >= n:
                        det_masks: List[Optional[np.ndarray]] = [None] * n
                        det_mask_boxes: List[Optional[Tuple[float, float, float, float]]] = [None] * n
                        det_vis: List[float] = [0.0] * n
                        k = 0
                        for i, b in enumerate(det_xyxy):
                            try:
                                x1, y1, x2, y2 = map(float, b)
                                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                                xi2, yi2 = min(Ww, int(x2)), min(Hh, int(y2))
                                if xi2 - xi1 > 1 and yi2 - yi1 > 1:
                                    roi_mask = det_masks_full[i][yi1:yi2, xi1:xi2]
                                    det_masks[i] = roi_mask
                                    det_mask_boxes[i] = (x1, y1, x2, y2)
                                    v = float(roi_mask.mean())
                                    det_vis[i] = v
                                    if v > 0.0:
                                        k += 1
                            except Exception:
                                pass
                        if (k > 0) and (os.getenv("DEBUG", "0") == "1" or os.getenv("DIAG", "1") == "1"):
                            try:
                                logger.debug("[sv] YOLO-SEG masks: %s/%s", k, n)
                            except Exception:
                                pass
                        return det_masks, det_mask_boxes, det_vis
        # Heavy fallback if YOLO masks missing/empty and allowed
        try:
            if (not FORCE_LITE_SEG) and (SEG_BACKEND in ("mask2former", "sam2")) and det_xyxy:
                if not hasattr(_extract_yolo_seg_masks, "_heavy_log"):
                    _extract_yolo_seg_masks._heavy_log = False  # type: ignore[attr-defined]
                masks_h, boxes_h, vis_h = None, None, None
                if SEG_BACKEND == "mask2former":
                    try:
                        from seg_mask2former import infer_roi_masks  # type: ignore
                        masks_h, boxes_h, vis_h = infer_roi_masks(frame_bgr, det_xyxy)
                    except Exception:
                        masks_h, boxes_h, vis_h = None, None, None
                else:
                    try:
                        from seg_sam2 import infer_roi_masks  # type: ignore
                        masks_h, boxes_h, vis_h = infer_roi_masks(frame_bgr, det_xyxy)
                    except Exception:
                        masks_h, boxes_h, vis_h = None, None, None
                if masks_h is not None and boxes_h is not None and vis_h is not None:
                    if not _extract_yolo_seg_masks._heavy_log:  # type: ignore[attr-defined]
                        logger.info("[seg] heavy backend active: %s", SEG_BACKEND)
                        _extract_yolo_seg_masks._heavy_log = True  # type: ignore[attr-defined]
                    return masks_h, boxes_h, vis_h
        except Exception:
            pass
        return None, None, None
    except Exception:
        return None, None, None


def _safe_crop_with_coords(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[
    Optional[np.ndarray], Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    xi1 = max(0, int(np.floor(x1)));
    yi1 = max(0, int(np.floor(y1)))
    xi2 = min(w, int(np.ceil(x2)));
    yi2 = min(h, int(np.ceil(y2)))
    if xi2 - xi1 <= 1 or yi2 - yi1 <= 1:
        return None, (0, 0, 0, 0)
    return frame[yi1:yi2, xi1:xi2], (xi1, yi1, xi2, yi2)


def _apply_mask_to_crop(crop: Optional[np.ndarray], mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if crop is None or mask is None:
        return crop
    try:
        if crop.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = crop.copy()
        out[mask == 0] = 0
        return out
    except Exception:
        return crop


def _mask_iou(track_mask: Optional[np.ndarray],
              track_box: Optional[Tuple[float, float, float, float]],
              det_mask: Optional[np.ndarray],
              det_box: Tuple[float, float, float, float]) -> Optional[float]:
    if track_mask is None or track_box is None or det_mask is None:
        return None
    try:
        dh = max(1, int(round(det_mask.shape[0])))
        dw = max(1, int(round(det_mask.shape[1])))
        tm = cv2.resize(track_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
        dm = det_mask.astype(np.uint8)
        inter = np.logical_and(tm > 0, dm > 0).sum()
        union = np.logical_or(tm > 0, dm > 0).sum()
        if union == 0:
            return 0.0
        return float(inter) / float(union)
    except Exception:
        return None


def _segment_roi(frame: np.ndarray, bbox: Tuple[float, float, float, float], backend: str = "grabcut") -> Tuple[
    Optional[np.ndarray], Optional[Tuple[float, float, float, float]], float]:
    """
    Returns (mask_uint8, refined_box_xyxy_in_frame, visibility_ratio).
    Mask shape == cropped ROI (with SEG_PAD_RATIO).
    """
    if not SEG_ENABLE:
        return None, None, 0.0
    Hh, Ww = frame.shape[:2]
    pb = _pad_bbox(bbox, SEG_PAD_RATIO, Ww, Hh)
    roi, (x1, y1, x2, y2) = _safe_crop_with_coords(frame, pb)
    if roi is None:
        return None, None, 0.0
    # Guard extremely large ROIs to avoid long/blocking segmentation
    roi_h, roi_w = roi.shape[:2]
    roi_area = float(roi_h * roi_w)
    frame_area = float(max(1, Hh * Ww))
    # Environment-configurable caps (with safe defaults if not defined)
    try:
        max_pixels = float(os.getenv("SEG_MAX_ROI_PIXELS", "102400"))  # e.g., 320x320
        max_frac = float(os.getenv("SEG_MAX_ROI_FRAC", "0.12"))  # 12% of frame
        max_side = int(os.getenv("SEG_ROI_MAX_SIDE", "320"))
        gc_big_iters = int(os.getenv("SEG_GC_ITERS_BIG", str(max(1, SEG_GC_ITERS // 2))))
    except Exception:
        max_pixels = 102400.0
        max_frac = 0.12
        max_side = 320
        gc_big_iters = max(1, SEG_GC_ITERS // 2)
    need_downscale = (roi_area > max_pixels) or ((roi_area / frame_area) > max_frac) or (max(roi_h, roi_w) > max_side)
    scale = 1.0
    small = roi
    if need_downscale:
        scale = float(min(1.0, max_side / float(max(1, max(roi_h, roi_w)))))
        if scale < 1.0:
            try:
                new_w = max(32, int(round(roi_w * scale)))
                new_h = max(32, int(round(roi_h * scale)))
                small = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception:
                small = roi
                scale = 1.0
    if backend == "grabcut":
        try:
            m = np.full(small.shape[:2], cv2.GC_PR_BGD, np.uint8)
            rect = (int(0.06 * small.shape[1]), int(0.06 * small.shape[0]), int(0.88 * small.shape[1]),
                    int(0.88 * small.shape[0]))
            iters = max(1, SEG_GC_ITERS if not need_downscale else gc_big_iters)
            cv2.grabCut(small, m, rect, None, None, iters, cv2.GC_INIT_WITH_RECT)
            mask_small = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            # Upscale back if we downscaled
            if scale < 1.0 and (small is not roi):
                mask = cv2.resize(mask_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = mask_small
        except Exception:
            # Fallback: simple rectangular foreground (avoid blocking)
            mask = np.ones((roi_h, roi_w), dtype=np.uint8)
    elif backend == "bgs":
        subtractor = getattr(_segment_roi, "_bgs", None)
        if subtractor is None:
            try:
                if SEG_BGS_KIND.lower() == "gsoc":
                    # type: ignore[attr-defined]
                    subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
                else:
                    subtractor = cv2.createBackgroundSubtractorKNN()
            except Exception:
                subtractor = cv2.createBackgroundSubtractorKNN()
            _segment_roi._bgs = subtractor  # type: ignore[attr-defined]
        raw = subtractor.apply(roi)
        _, mask = cv2.threshold(raw, 200, 1, cv2.THRESH_BINARY)
    else:
        return None, None, 0.0
    if mask is None or (np.sum(mask) <= 0):
        return None, None, 0.0
    try:
        num, labels = cv2.connectedComponents(mask.astype(np.uint8))
        if num > 1:
            cnts = np.bincount(labels.ravel())
            if cnts.size > 0:
                cnts[0] = 0
                mask = (labels == int(np.argmax(cnts))).astype(np.uint8)
    except Exception:
        pass
    if SEG_MORPH_KERNEL >= 3 and (SEG_MORPH_KERNEL % 2 == 1):
        k = np.ones((SEG_MORPH_KERNEL, SEG_MORPH_KERNEL), np.uint8)
        try:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        except Exception:
            pass
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None, None, 0.0
    rx1, ry1, rx2, ry2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rb = (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2)
    vis = float(mask.sum()) / float(mask.shape[0] * mask.shape[1])
    return mask, rb, vis


def _to_tensor_rgb(rgb: np.ndarray, size_hw: _Tuple[int, int]) -> "torch.Tensor":
    ten = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # CHW
    ten = torch.nn.functional.interpolate(
        ten.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False
    ).squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (ten - mean) / std


def _compute_embeddings_for_dets(frame: np.ndarray,
                                 boxes: List[Tuple[float, float, float, float]],
                                 batch_size: int = 32,
                                 masks: Optional[List[Optional[np.ndarray]]] = None,
                                 force_builtin: bool = False
                                 ) -> List[Optional[np.ndarray]]:
    # Pluggable fast path
    if (not force_builtin) and ('_REID_PLUG' in globals()) and (globals().get('_REID_PLUG', None) is not None):
        plug = globals().get('_REID_PLUG')
        if boxes:
            Hh, Ww = frame.shape[:2]
            # Optional SR lazy instance
            sr_inst = None
            if SR_REID_ENABLE and (CropSR is not None):
                try:
                    sr_inst = getattr(_compute_embeddings_for_dets, "_sr", None)
                    if sr_inst is None:
                        sr_inst = CropSR()  # type: ignore[operator]
                        _compute_embeddings_for_dets._sr = sr_inst  # type: ignore[attr-defined]
                except Exception:
                    sr_inst = None
            crops: List[Optional[np.ndarray]] = []
            for i, b in enumerate(boxes):
                pb = _pad_bbox(b, REID_PAD_RATIO, Ww, Hh)
                roi = _safe_crop(frame, pb)
                if MASK_GUIDED_EMB and masks is not None and i < len(masks) and masks[i] is not None:
                    roi = _apply_mask_to_crop(roi, masks[i])
                if roi is not None and sr_inst is not None:
                    try:
                        roi_h, roi_w = roi.shape[:2]
                        if min(roi_h, roi_w) < SR_MIN_SIDE:
                            roi = sr_inst.maybe_upscale(roi, min_side=SR_MIN_SIDE)  # type: ignore[operator]
                    except Exception:
                        pass
                crops.append(roi)
            # Build valid indices and batch only valid crops
            valid_idx = [i for i, c in enumerate(crops) if c is not None]
            outs: List[Optional[np.ndarray]] = [None] * len(boxes)
            if valid_idx:
                valid_crops = [crops[i] for i in valid_idx]
                try:
                    arr = plug.forward(valid_crops, batch_size=max(1, int(batch_size)))  # type: ignore[attr-defined]
                    # arr is expected to be (N_valid, D)
                    if isinstance(arr, np.ndarray):
                        for j, i_idx in enumerate(valid_idx):
                            if j < arr.shape[0]:
                                v = arr[j]
                                if isinstance(v, np.ndarray) and v.size > 0:
                                    v = v.astype(np.float32)
                                    # enforce L2 normalization defensively
                                    n = float(np.linalg.norm(v))
                                    if n > 1e-12:
                                        v = v / n
                                    outs[i_idx] = v
                    # else: keep None for all
                except Exception:
                    # fall back to legacy path if pluggable fails
                    pass
                # If some valid crops still missing, compute them via builtin path as fallback
                missing_idx = [i for i in valid_idx if
                               (outs[i] is None) or (isinstance(outs[i], np.ndarray) and outs[i].size == 0)]
                if missing_idx:
                    sub_boxes = [boxes[i] for i in missing_idx]
                    sub_masks = [masks[i] if (masks is not None and i < len(masks)) else None for i in missing_idx]
                    try:
                        sub_outs = _compute_embeddings_for_dets(frame, sub_boxes, batch_size=batch_size,
                                                                masks=sub_masks, force_builtin=True)
                        for k, i_idx in enumerate(missing_idx):
                            if k < len(sub_outs):
                                v = sub_outs[k]
                                if isinstance(v, np.ndarray) and v.size > 0:
                                    n = float(np.linalg.norm(v))
                                    if n > 1e-12:
                                        v = v / n
                                    outs[i_idx] = v
                    except Exception:
                        pass
            return outs
    # Legacy/built-in path
    if _REID_MODEL is None or _REID_TF is None or _REID_DEV is None:
        ok = _init_reid_builtin_only() if force_builtin else _init_reid()
        if not ok or _REID_MODEL is None or _REID_TF is None or _REID_DEV is None:
            return [None] * len(boxes)
    if not boxes:
        return []
    imgs = []
    valids = []
    H, W = frame.shape[:2]
    for b in boxes:
        pb = _pad_bbox(b, REID_PAD_RATIO, W, H)
        roi = _safe_crop(frame, pb)
        if roi is None:
            imgs.append(None)
            valids.append(False)
            continue
        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            # Use PIL transforms when possible
            try:
                if HAS_PIL and Image is not None:
                    pil = Image.fromarray(rgb)
                    ten = _REID_TF(pil)  # Resize->ToTensor->Normalize if OSNet path
                else:
                    raise RuntimeError("force numpy path")
            except Exception:
                ten = _to_tensor_rgb(rgb, _REID_INPUT_SIZE)
            imgs.append(ten)
            valids.append(True)
        except Exception:
            imgs.append(None)
            valids.append(False)
    # Batch forward
    outs: List[Optional[np.ndarray]] = [None] * len(boxes)
    if PROFILE_EMB:
        import time as _t
        t0 = _t.time()
    try:
        with torch.no_grad():
            cur: List[torch.Tensor] = []
            idxs: List[int] = []
            for i, (ok, ten) in enumerate(zip(valids, imgs)):
                if not ok or ten is None:
                    continue
                cur.append(ten.unsqueeze(0))
                idxs.append(i)
                if len(cur) == max(1, int(batch_size)):
                    batch = torch.cat(cur, dim=0).to(_REID_DEV)
                    emb = _REID_MODEL(batch)  # type: ignore[operator]
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    vecs = emb.detach().cpu().float().numpy()
                    for j, vi in enumerate(idxs):
                        outs[vi] = vecs[j]
                    cur = []
                    idxs = []
            if cur:
                batch = torch.cat(cur, dim=0).to(_REID_DEV)
                emb = _REID_MODEL(batch)  # type: ignore[operator]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                vecs = emb.detach().cpu().float().numpy()
                for j, vi in enumerate(idxs):
                    outs[vi] = vecs[j]
    except Exception:
        pass
    if PROFILE_EMB:
        import time as _t
        logger.debug("[emb] frame batch %s -> took %.3fs", len(boxes), (_t.time() - t0))
    return outs


def _compute_embedding(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    # Single box path, kept for ghost stitcher usage; uses padding
    embs = _compute_embeddings_for_dets(frame, [bbox], batch_size=1)
    return embs[0] if embs else None


def _cosine_sim(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> Optional[float]:
    if v1 is None or v2 is None:
        return None
    try:
        a = np.asarray(v1, dtype=np.float32).reshape(-1)
        b = np.asarray(v2, dtype=np.float32).reshape(-1)
        if a.shape != b.shape or a.size == 0:
            return None
        num = float(np.dot(a, b))
        # vectors should be l2-normalized; still guard denom
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den <= 1e-12:
            return None
        sim = num / den
        # map [-1,1] -> [0,1]
        return max(0.0, min(1.0, 0.5 * (sim + 1.0)))
    except Exception:
        return None


def _parse_roi(poly_str: str) -> Optional[np.ndarray]:
    if not poly_str:
        return None
    pts = []
    for item in poly_str.split(";"):
        if "," not in item:
            continue
        xs, ys = item.split(",")[:2]
        try:
            pts.append([float(xs), float(ys)])
        except Exception:
            pass
    if len(pts) >= 3:
        return np.array(pts, dtype=np.int32)
    return None


ROI_POLY = _parse_roi(ROI_POLY_STR)


def _point_in_roi(x: float, y: float) -> bool:
    if ROI_POLY is None:
        return True
    return cv2.pointPolygonTest(ROI_POLY, (float(x), float(y)), False) >= 0


def _draw_roi_overlay(frame: np.ndarray):
    if ROI_POLY is None:
        return
    overlay = frame.copy()
    cv2.polylines(overlay, [ROI_POLY], isClosed=True, color=(60, 200, 255), thickness=2)
    alpha = 0.15
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [ROI_POLY], color=(60, 200, 255))
    cv2.addWeighted(mask, alpha, overlay, 1.0, 0, overlay)
    cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)


def _render_eval(frame: np.ndarray,
                 detections: "sv.Detections",
                 state: Dict[int, Dict[str, Any]],
                 W: int, H: int,
                 draw_trails: bool = False):
    if detections.xyxy is not None and len(detections) > 0:
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(float).tolist()
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            x1, y1, x2, y2 = bbox
            p1 = (int(max(0, x1)), int(max(0, y1)))
            p2 = (int(min(W - 1, x2)), int(min(H - 1, y2)))
            color = _track_color(tid if tid >= 0 else 0)
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, f"ID {tid}", (p1[0], max(0, p1[1] - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if draw_trails and state:
        for tid, s in state.items():
            tr = s.get("trail", [])
            if len(tr) >= 2:
                pts = np.array([[int(x), int(y)] for (x, y) in tr[-EVAL_TRAIL_LENGTH:]], dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=False, color=_track_color(tid), thickness=2)


# -----------------------------
# Core: YOLO + Supervision ByteTrack (+ optional Slicer & Smoother)
# -----------------------------
def run_tracking_with_supervision() -> Dict[str, Any]:
    if not HAS_YOLO:
        raise RuntimeError("Ultralytics YOLO not installed. `pip install ultralytics`")
    if not HAS_SUPERVISION:
        raise RuntimeError("Supervision not installed. `pip install supervision`")
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if DIAG:
        logger.debug(
            "[sv] cfg | "
            f"ASSOC_APPEAR_ENABLE={int(ASSOC_APPEAR_ENABLE)} "
            f"APPEAR_EMB_ENABLE={int(APPEAR_EMB_ENABLE)} "
            f"REID_BACKBONE={REID_BACKBONE} REID_DEVICE={REID_DEVICE} REID_FP16={int(REID_FP16)} "
            f"REID_EVERY_N={REID_EVERY_N} "
            f"BT_MATCH={BT_MATCH_THRESH} BT_HIGH={BT_TRACK_THRESH} BT_BUFFER={BT_BUFFER} "
            f"SMOOTH_BOXES={int(SMOOTH_BOXES)} SLICER_ENABLE={int(SLICER_ENABLE)} "
            f"ROI_ENFORCE={int(ROI_ENFORCE)} TRACK_CONF={TRACK_CONF} TRACK_IOU={TRACK_IOU}",
        )
    ensure_dir(CAPTURES_DIR)
    model = YOLO(WEIGHTS_PATH)

    # Prefer CUDA and safe fuse (TRT engines will ignore .to('cuda') gracefully)
    try:
        if HAS_TORCH and torch.cuda.is_available():
            _ = model.to('cuda')
        if int(os.getenv("UL_SAFE_FUSE", "0")) == 1 and hasattr(model, "fuse"):
            try:
                model.fuse()
            except Exception:
                pass
    except Exception:
        pass

    # Ensemble and TTA setup
    ensemble_models: List[Any] = [model]
    if DET_ENSEMBLE and DET_MODELS:
        for p in [pp.strip() for pp in DET_MODELS.split(';') if pp.strip()]:
            try:
                if os.path.exists(p):
                    m2 = YOLO(p)
                    try:
                        if HAS_TORCH and torch.cuda.is_available():
                            _ = m2.to('cuda')
                        if int(os.getenv("UL_SAFE_FUSE", "0")) == 1 and hasattr(m2, "fuse"):
                            try:
                                m2.fuse()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    ensemble_models.append(m2)
            except Exception:
                continue
    # Parse TTA scales once
    try:
        _scs = [float(x.strip()) for x in (DET_TTA_SCALES.split(',') if DET_TTA_SCALES else ['1.0'])]
        TTA_SCALES_LIST: List[float] = [s for s in _scs if s > 0]
        if len(TTA_SCALES_LIST) == 0:
            TTA_SCALES_LIST = [1.0]
    except Exception:
        TTA_SCALES_LIST = [1.0]

    # Video info
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Ensure ReID is initialized before reporting status
    try:
        _ = _init_reid()
    except Exception:
        pass
    plug_ok = ('_REID_PLUG' in globals() and _REID_PLUG is not None)
    builtin_ok = (_REID_MODEL is not None)
    dim_val = 0
    try:
        if plug_ok:
            dim_val = int(getattr(_REID_PLUG, "embedding_dim", 0))
        elif builtin_ok and hasattr(_REID_MODEL, "fc"):
            dim_val = int(getattr(_REID_MODEL.fc, "in_features", 0)) or 0
    except Exception:
        dim_val = 0
    _reid_status = {
        "plug": ("yes" if plug_ok else "no"),
        "builtin": ("yes" if builtin_ok else "no"),
        "device": (str(_REID_DEV) if (_REID_DEV is not None) else REID_DEVICE),
        "dim": dim_val
    }
    _diag_emit("reid_init", **_reid_status)

    # Initialize SeqTrack-LSTM predictor
    global traj_predictor
    try:
        if LSTM_MOTION_ENABLE:
            dev = 'cuda' if (HAS_TORCH and torch is not None and torch.cuda.is_available()) else 'cpu'
            variant = LSTM_VARIANT if LSTM_VARIANT in ('A', 'B') else 'A'
            traj_predictor = SeqTrackLSTM(input_size=8, hidden_size=int(LSTM_HIDDEN), num_layers=2,
                                          variant=variant, device=dev, fp16=bool(LSTM_FP16))
            _diag_emit("lstm_init", enable=int(LSTM_MOTION_ENABLE), variant=variant, hidden=int(LSTM_HIDDEN),
                       device=str(dev))
    except Exception as e:
        traj_predictor = None
        _diag_emit("lstm_init_fail", err=str(e))

    base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    out_path = os.path.join(CAPTURES_DIR, f"tracked_{base}_sv_bytetrack.mp4")
    writer = None
    if WRITE_VIDEO and not DRY_RUN:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        # Windows codec fallback
        try:
            if (writer is None) or (not writer.isOpened()):
                fourcc2 = cv2.VideoWriter_fourcc(*'avc1')
                writer = cv2.VideoWriter(out_path, fourcc2, fps, (W, H))
        except Exception:
            pass

    # Follow export setup
    follow_root = None
    follow_writers: Dict[int, cv2.VideoWriter] = {}
    follow_paths: Dict[int, str] = {}
    if FOLLOW_EXPORT and not DRY_RUN:
        try:
            base_dir = os.path.join(CAPTURES_DIR, base)
            follow_root = os.path.join(base_dir, FOLLOW_DIRNAME)
            ensure_dir(follow_root)
        except Exception:
            follow_root = None

    # Supervision pieces
    tracker = sv.ByteTrack(track_activation_threshold=BT_TRACK_THRESH,
                           lost_track_buffer=BT_BUFFER,
                           minimum_matching_threshold=BT_MATCH_THRESH,
                           frame_rate=int(round(fps)))  # ByteTrack params. :contentReference[oaicite:3]{index=3}
    smoother = sv.DetectionsSmoother(
        length=max(3, SMOOTH_LENGTH)) if SMOOTH_BOXES else None  # :contentReference[oaicite:4]{index=4}

    # Optional SAHI/tiling for big frames
    slicer = None
    if SLICER_ENABLE and (W > SLICE_W or H > SLICE_H):
        def _slice_callback(image_slice: np.ndarray) -> sv.Detections:
            use_aug = False  # keep False in slicer callback unless you want per-slice TTA
            res = model(image_slice, verbose=False, conf=TRACK_CONF, iou=TRACK_IOU, augment=use_aug)[0]
            return sv.Detections.from_ultralytics(res)

        overlap_px_w = max(0, int(SLICE_W * SLICER_OVERLAP_RATIO_W))
        overlap_px_h = max(0, int(SLICE_H * SLICER_OVERLAP_RATIO_H))
        slicer = sv.InferenceSlicer(
            callback=_slice_callback,
            slice_wh=(SLICE_W, SLICE_H),
            overlap_ratio_wh=None,
            overlap_wh=(overlap_px_w, overlap_px_h),
            thread_workers=max(1, SLICER_THREADS),
        )  # :contentReference[oaicite:5]{index=5}

    # State for simple predicted overlays & trails
    state: Dict[int, Dict[str, Any]] = {}
    frame_idx = -1
    # GMC state/stats
    prev_gray = None
    gmc_used = 0
    gmc_skips = 0
    # Tracks per frame for MOT export (1-based frame index)
    tracks_per_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    # Metrics and bookkeeping
    created_total = 0
    dropped_total = 0
    recovered_total = 0
    id_switch_est_total = 0
    completed_lengths: List[int] = []
    next_tid = 1
    cost_debug_acc: Optional[Dict[str, float]] = (
        {"iou_sum": 0.0, "emb_sum": 0.0, "hist_sum": 0.0, "pairs": 0.0} if ASSOC_COST_DEBUG else None)
    # Slicer safety state
    slicer_fail_count = 0
    slicer_timeouts = 0
    slicer_errors = 0
    slicer_disabled_logged = False

    # ThresholdManager (adaptive gates/weights)
    tm = ThresholdManager(window=120)

    # Auto-preset probing state
    ema_conf = float(CONFIDENCE_THRESHOLD)
    ema_speed = 0.0  # avg |vel| / frame diag
    ema_density = 0.0  # normalized detection count
    rough_acc_vals: List[float] = []
    gmc_fail_flags: List[int] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx >= MAX_FRAMES:
            break

        if DIAG and frame_idx == 0:
            try:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                test_box = (cx - 32, cy - 64, cx + 32, cy + 64)
                test = _compute_embeddings_for_dets(frame, [test_box], batch_size=1)
                tdim = 0 if (not test or test[0] is None) else int(test[0].size)
                _diag_emit("reid_smoke", emb_dim=tdim, device=_reid_status.get("device"))
            except Exception as e:
                _diag_emit("reid_smoke_fail", err=str(e))

        # reset detector-provided seg masks for this frame
        last_masks_from_yolo = None
        last_mask_boxes = None
        last_mask_vis = None

        # ---- Inference -> Detections
        try:
            if slicer is not None:
                dets, status = _call_slicer_with_timeout(slicer, frame, SLICER_CALL_TIMEOUT)
                if status == 'ok' and dets is not None:
                    detections = dets
                else:
                    slicer_fail_count += 1
                    if status == 'timeout':
                        slicer_timeouts += 1
                        if DEBUG or DRAW_DEBUG:
                            logger.warning("[sv-pipeline] slicer timeout on frame %s (#%s)", frame_idx, slicer_fail_count)
                    else:
                        slicer_errors += 1
                        if DEBUG or DRAW_DEBUG:
                            logger.warning("[sv-pipeline] slicer error on frame %s (#%s)", frame_idx, slicer_fail_count)
                    # Fallback to direct YOLO inference for this frame (with optional TTA/ensemble)
                    if DET_ENSEMBLE or DET_TTA:
                        detections = _infer_ensemble(
                            (ensemble_models if DET_ENSEMBLE else [model]),
                            frame,
                            TRACK_CONF,
                            TRACK_IOU,
                            use_tta=bool(DET_TTA),
                            flip=bool(DET_TTA_FLIP),
                            scales=TTA_SCALES_LIST,
                            wbf_iou=float(DET_TTA_WBF_IOU if DET_TTA else DET_WBF_IOU),
                            skip_boxes=int(DET_WBF_SKIP_BOXES)
                        )
                        last_masks_from_yolo = None
                        last_mask_boxes = None
                        last_mask_vis = None
                    else:
                        use_aug = (DET_TTA_EVERY > 0 and (frame_idx % DET_TTA_EVERY) == 0)
                        res = model(frame, verbose=False, conf=TRACK_CONF, iou=TRACK_IOU, augment=use_aug)[0]
                        detections = sv.Detections.from_ultralytics(res)
                        # ---- Prefer detector-provided masks (YOLO-SEG) when available ----
                        last_masks_from_yolo = None
                        last_mask_boxes = None
                        last_mask_vis = None
                        try:
                            if detections.xyxy is not None and len(detections) > 0:
                                det_xyxy_list = [tuple(map(float, detections.xyxy[i].tolist())) for i in
                                                 range(len(detections))]
                                last_masks_from_yolo, last_mask_boxes, last_mask_vis = _extract_yolo_seg_masks(res,
                                                                                                               frame,
                                                                                                               det_xyxy_list)
                        except Exception:
                            last_masks_from_yolo = None
                            last_mask_boxes = None
                            last_mask_vis = None
                    if slicer_fail_count >= SLICER_TIMEOUT_DISABLE_AFTER:
                        slicer = None
                        if not slicer_disabled_logged:
                            logger.warning(
                                "[sv-pipeline] disabling slicer after %s failures (timeouts=%s, errors=%s)",
                                slicer_fail_count,
                                slicer_timeouts,
                                slicer_errors,
                            )
                            slicer_disabled_logged = True
            else:
                # Normal path (no slicer): run ensemble/TTA if enabled, else single-model
                if DET_ENSEMBLE or DET_TTA:
                    detections = _infer_ensemble(
                        (ensemble_models if DET_ENSEMBLE else [model]),
                        frame,
                        TRACK_CONF,
                        TRACK_IOU,
                        use_tta=bool(DET_TTA),
                        flip=bool(DET_TTA_FLIP),
                        scales=TTA_SCALES_LIST,
                        wbf_iou=float(DET_TTA_WBF_IOU if DET_TTA else DET_WBF_IOU),
                        skip_boxes=int(DET_WBF_SKIP_BOXES)
                    )
                    # Detector-provided masks are not aggregated in TTA/ensemble mode
                    last_masks_from_yolo = None
                    last_mask_boxes = None
                    last_mask_vis = None
                else:
                    use_aug = (DET_TTA_EVERY > 0 and (frame_idx % DET_TTA_EVERY) == 0)
                    res = model(frame, verbose=False, conf=TRACK_CONF, iou=TRACK_IOU, augment=use_aug)[0]
                    detections = sv.Detections.from_ultralytics(res)
                    # ---- Prefer detector-provided masks (YOLO-SEG) when available ----
                    last_masks_from_yolo = None
                    last_mask_boxes = None
                    last_mask_vis = None
                    try:
                        if detections.xyxy is not None and len(detections) > 0:
                            det_xyxy_list = [tuple(map(float, detections.xyxy[i].tolist())) for i in
                                             range(len(detections))]
                            last_masks_from_yolo, last_mask_boxes, last_mask_vis = _extract_yolo_seg_masks(res, frame,
                                                                                                           det_xyxy_list)
                    except Exception:
                        last_masks_from_yolo = None
                        last_mask_boxes = None
                        last_mask_vis = None
        except KeyboardInterrupt:
            logger.info("[sv-pipeline] KeyboardInterrupt: stopping processing loop")
            break

        # Optional class filter BEFORE tracking to reduce noise
        if TRACK_CLASSES is not None and detections.class_id is not None:
            cls_mask = np.isin(detections.class_id, np.array(TRACK_CLASSES, dtype=int))
            detections = detections[cls_mask]

        # ROI filter BEFORE tracking if enforced (keeps tracks only inside polygon)
        if ROI_ENFORCE and detections.xyxy is not None and len(detections) > 0:
            xyxy = detections.xyxy
            cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
            cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
            roi_mask = np.array([_point_in_roi(float(x), float(y)) for x, y in zip(cx, cy)], dtype=bool)
            detections = detections[roi_mask]

        # --- Secondary NMS to remove residual dupes (incl. slicer seams) ---
        detections = _secondary_nms_detections(detections, iou_thr=0.45)

        # ---------- Auto-Flex tuner ----------
        # Computes flexible detection confidence threshold and association rails per frame.
        # EMAs: ema_conf (p30 det confidence), ema_speed (mean |vel|/diag), ema_density (normed det count).
        conf_thr = CONFIDENCE_THRESHOLD  # fallback
        assoc_gates = {
            "ASSOC_MIN_IOU": ASSOC_MIN_IOU,
            "ASSOC_MIN_SIM": ASSOC_MIN_SIM,
            "ASSOC_MAX_CENTER_DIST": ASSOC_MAX_CENTER_DIST,
            "ID_REASSIGN_IOU": ID_REASSIGN_IOU,
            "ID_REASSIGN_DIST_RATIO": ID_REASSIGN_DIST_RATIO,
            "APPEAR_EMB_MIN_SIM": APPEAR_EMB_MIN_SIM,
            "APPEAR_MIN_SIM": APPEAR_MIN_SIM,
        }
        try:
            if FLEX_ENABLE and detections.xyxy is not None and len(detections) > 0:
                diag = float(np.hypot(W, H))

                # 30th percentile confidence for robustness
                conf_arr = detections.confidence if detections.confidence is not None else np.ones((len(detections),),
                                                                                                   dtype=float)
                p30 = float(np.percentile(conf_arr, 30)) if len(conf_arr) else CONFIDENCE_THRESHOLD
                ema_conf = (1.0 - FLEX_ALPHA) * ema_conf + FLEX_ALPHA * p30

                # normalized mean speed of active tracks
                if state:
                    v = [float(np.hypot(*(s.get("last_vel", (0.0, 0.0))))) for s in state.values()]
                    speed = (np.mean(v) / max(1e-6, diag)) if v else 0.0
                else:
                    speed = 0.0
                ema_speed = (1.0 - FLEX_ALPHA) * ema_speed + FLEX_ALPHA * speed

                # density in [0..1] (soft cap ~12 detections)
                density = min(1.0, len(detections) / 12.0)
                ema_density = (1.0 - FLEX_ALPHA) * ema_density + FLEX_ALPHA * density

                # --- Compute flexible thresholds (clamped by rails) ---
                conf_thr = float(np.clip(ema_conf - 0.05, FLEX_CONF_MIN, FLEX_CONF_MAX))

                assoc_min_iou_cur = float(np.clip(
                    ASSOC_MIN_IOU_BASE * (0.6 + 0.8 * (1.0 - ema_speed)),
                    FLEX_IOU_MIN, FLEX_IOU_MAX
                ))

                assoc_min_sim_cur = float(np.clip(
                    ASSOC_MIN_SIM_BASE * (0.6 + 0.8 * ema_conf),
                    FLEX_SIM_MIN, FLEX_SIM_MAX
                ))

                assoc_center_cur = float(np.clip(
                    ASSOC_MAX_CENTER_DIST_BASE * (1.0 + 2.0 * ema_speed + 0.3 * ema_density),
                    0.04, FLEX_CENTER_MAX
                ))

                reassign_iou_cur = float(np.clip(
                    ID_REASSIGN_IOU_BASE * (0.5 + 1.2 * (1.0 - ema_speed)),
                    FLEX_IOU_MIN, 0.40
                ))

                assoc_gates = {
                    "ASSOC_MIN_IOU": assoc_min_iou_cur,
                    "ASSOC_MIN_SIM": assoc_min_sim_cur,
                    "ASSOC_MAX_CENTER_DIST": assoc_center_cur,
                    "ID_REASSIGN_IOU": reassign_iou_cur,
                    "ID_REASSIGN_DIST_RATIO": ID_REASSIGN_DIST_RATIO,
                    "APPEAR_EMB_MIN_SIM": APPEAR_EMB_MIN_SIM,
                    "APPEAR_MIN_SIM": APPEAR_MIN_SIM,
                }

                if FLEX_LOG and (frame_idx % 60 == 0):
                    logger.info(json.dumps({
                        "frame": int(frame_idx),
                        "conf_thr": round(conf_thr, 3),
                        "assoc_min_iou": round(assoc_min_iou_cur, 3),
                        "assoc_min_sim": round(assoc_min_sim_cur, 3),
                        "assoc_center": round(assoc_center_cur, 3),
                        "reassign_iou": round(reassign_iou_cur, 3),
                        "ema_conf": round(ema_conf, 3),
                        "ema_speed": round(ema_speed, 4),
                        "ema_density": round(ema_density, 3),
                    }))
        except Exception:
            conf_thr = CONFIDENCE_THRESHOLD
        # ---------- /Auto-Flex ----------

        # Pre-tracker gating: confidence, area, aspect ratio
        if detections.xyxy is not None and len(detections) > 0:
            xyxy = detections.xyxy
            w_arr = np.clip(xyxy[:, 2] - xyxy[:, 0], 1e-3, None)
            h_arr = np.clip(xyxy[:, 3] - xyxy[:, 1], 1e-3, None)
            area = w_arr * h_arr
            area_ratio = area / float(W * H)
            aspect = w_arr / h_arr
            conf_arr = detections.confidence if detections.confidence is not None else np.ones((len(detections),),
                                                                                               dtype=float)
            mask = (
                    (conf_arr >= conf_thr) &
                    (area_ratio >= MIN_AREA_RATIO) & (area_ratio <= MAX_AREA_RATIO) &
                    (aspect >= MIN_ASPECT_RATIO) & (aspect <= MAX_ASPECT_RATIO)
            )
            detections = detections[mask]

        # ---- Appearance-aware association or ByteTrack path
        # Prepare grayscale and GMC
        H_cam = None
        try:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            curr_gray = None
        if estimate_gmc is not None:
            if prev_gray is None:
                reset_gmc_smoothing()
                H_cam = np.eye(3, dtype=np.float32)
            else:
                try:
                    # Exclude current boxes from keypoints to prefer background features
                    _boxes_for_mask = []
                    if detections.xyxy is not None and len(detections) > 0:
                        _boxes_for_mask = [tuple(map(float, b.tolist())) for b in detections.xyxy]
                    H_cam_est, gstats = estimate_gmc(
                        prev_gray, curr_gray,
                        method=GMC_METHOD,  # orb|oflow|raft
                        nfeatures=ORB_NFEATURES,
                        ransac_thresh=ORB_RANSAC_THRESH,
                        downscale=GMC_DOWNSCALE,
                        mask_exclude_boxes=_boxes_for_mask
                    )
                    okv = 0.0
                    inr = 0.0
                    try:
                        okv = float(gstats.get('ok', 0.0) or 0.0)
                        inr = float(gstats.get('inlier_ratio', 0.0) or 0.0)
                    except Exception:
                        okv, inr = 0.0, 0.0
                    if (okv >= 0.5) or (inr >= GMC_INLIER_MIN):
                        H_cam = H_cam_est
                        gmc_used += 1
                    else:
                        H_cam = np.eye(3, dtype=np.float32)
                        gmc_skips += 1
                except Exception:
                    H_cam = np.eye(3, dtype=np.float32)
                    gmc_skips += 1
            prev_gray = curr_gray
            if DIAG and ((frame_idx + 1) % DIAG_EVERY == 0):
                total = max(1, (gmc_used + gmc_skips))
                _diag_emit("gmc", frame=int(frame_idx), used=int(gmc_used), skips=int(gmc_skips),
                           ratio=float(gmc_used / total))
        recovered_count_frame = 0
        reason = None
        if not ASSOC_APPEAR_ENABLE:
            reason = "flag_off"
        elif detections.xyxy is None or len(detections) == 0:
            reason = "no_detections"

        if reason is not None and DIAG and ((frame_idx + 1) % DIAG_EVERY == 0):
            _diag_emit("branch", path="BT", frame=int(frame_idx),
                       dets=(0 if detections.xyxy is None else int(len(detections))),
                       reason=reason)

        if ASSOC_APPEAR_ENABLE and detections.xyxy is not None and len(detections) > 0:
            xyxy_now = detections.xyxy.astype(float)
            boxes_list: List[Tuple[float, float, float, float]] = [tuple(x) for x in xyxy_now.tolist()]
            # Prefer detector-provided masks (from YOLO-SEG). Fallback to lightweight ROI segmentation.
            if SEG_ENABLE and ('last_masks_from_yolo' in locals()) and (last_masks_from_yolo is not None) and (
            not FORCE_LITE_SEG):
                det_masks = last_masks_from_yolo
                det_mask_boxes = last_mask_boxes
                det_vis = last_mask_vis
            else:
                det_masks = [None] * len(boxes_list)
                det_mask_boxes = [None] * len(boxes_list)
                det_vis = [0.0] * len(boxes_list)
                if SEG_ENABLE and len(boxes_list) > 0:
                    # (Existing overlap/ambiguity selection and _segment_roi(...) loop remain unchanged)
                    # Keep your current budgeted segmentation logic as-is here.
                    overlap_flags = [False] * len(boxes_list)
                    for i in range(len(boxes_list)):
                        for j in range(i + 1, len(boxes_list)):
                            if _iou_xyxy(boxes_list[i], boxes_list[j]) >= SEG_OVERLAP_IOU:
                                overlap_flags[i] = True;
                                overlap_flags[j] = True
                    work_idxs: List[int] = [i for i, f in enumerate(overlap_flags) if f and SEG_ON_OVERLAP]
                    if SEG_ON_AMBIGUITY and state:
                        diag_local = float(np.hypot(W, H))
                        for tid2, s2 in state.items():
                            gb = s2.get("last_bbox", None)
                            if gb is None:
                                continue
                            for di2, db2 in enumerate(boxes_list):
                                if len(work_idxs) >= SEG_MAX_PER_FRAME:
                                    break
                                iou2 = _iou_xyxy(gb, db2)
                                if iou2 < max(ASSOC_MIN_IOU, 0.05):
                                    continue
                                cdist2 = _center_distance(_bbox_center(gb), _bbox_center(db2))
                                if cdist2 > ASSOC_MAX_CENTER_DIST * diag_local:
                                    continue
                                emb_sim2 = _cosine_sim(s2.get("emb"), None)
                                hist_sim2 = _hist_similarity(s2.get("hist"), None)
                                approx = ASSOC_W_IOU * iou2 + ASSOC_W_EMB * ((emb_sim2 or 0.0)) + ASSOC_W_HIST * (
                                (hist_sim2 or 0.0))
                                if ASSOC_MIN_SIM <= approx < (ASSOC_MIN_SIM + SEG_AMBIG_MARGIN):
                                    work_idxs.append(di2)
                            if len(work_idxs) >= SEG_MAX_PER_FRAME:
                                break
                    work_idxs = list(dict.fromkeys(work_idxs))[:SEG_MAX_PER_FRAME]
                    import time as _t
                    _t0 = _t.time()
                    for di2 in work_idxs:
                        try:
                            mk, rb, vs = _segment_roi(frame, boxes_list[di2], backend=SEG_BACKEND)
                            det_masks[di2] = mk
                            det_mask_boxes[di2] = rb
                            det_vis[di2] = float(vs or 0.0)
                        except Exception:
                            pass
                        if (((_t.time() - _t0) * 1000.0) > SEG_MS_BUDGET):
                            break

            # Precompute features once
            det_hists = []
            if APPEAR_ENABLE:
                for i, b in enumerate(boxes_list):
                    if SEG_ENABLE and det_masks[i] is not None and MASK_GUIDED_HIST:
                        det_hists.append(_compute_hsv_hist(frame, b, mask_roi=det_masks[i]))
                    else:
                        det_hists.append(_compute_hsv_hist(frame, b))
            else:
                det_hists = [None] * len(boxes_list)
            det_embs = []
            if APPEAR_EMB_ENABLE:
                # compute embeddings only every N frames to save time
                if REID_EVERY_N <= 1 or (frame_idx % REID_EVERY_N == 0):
                    # masked batch, function will ignore None masks
                    det_embs = _compute_embeddings_for_dets(frame, boxes_list, REID_BATCH_SIZE, masks=(
                        det_masks if (SEG_ENABLE and MASK_GUIDED_EMB) else None))
                else:
                    det_embs = [None] * len(boxes_list)
            else:
                det_embs = [None] * len(boxes_list)

            # Recompute masked features if needed (redundant safe-guard)
            if SEG_ENABLE and any(m is not None for m in det_masks):
                if MASK_GUIDED_HIST and APPEAR_ENABLE:
                    for i, m in enumerate(det_masks):
                        if m is not None:
                            det_hists[i] = _compute_hsv_hist(frame, boxes_list[i], mask_roi=m)
                if MASK_GUIDED_EMB and APPEAR_EMB_ENABLE:
                    det_embs_masked = _compute_embeddings_for_dets(frame, boxes_list, REID_BATCH_SIZE, masks=det_masks)
                    for i, vec in enumerate(det_embs_masked):
                        if det_masks[i] is not None and vec is not None:
                            det_embs[i] = vec

            # After det_embs are computed, emit embedding yield diagnostics
            if DIAG and ((frame_idx + 1) % DIAG_EVERY == 0):
                emb_dim = 0
                valid_emb = 0
                for v in (det_embs or []):
                    if isinstance(v, np.ndarray) and v.size > 0:
                        valid_emb += 1
                        emb_dim = max(emb_dim, int(v.size))
                _diag_emit("branch", path="AA", frame=int(frame_idx),
                           dets=int(len(boxes_list)), emb_ok=int(valid_emb),
                           emb_dim=int(emb_dim), reidN=int(REID_EVERY_N))

            # Associate
            try:
                prev_next = next_tid
                det_confs_list = detections.confidence.tolist() if detections.confidence is not None else [1.0] * len(
                    boxes_list)
                assigned_ids, state, next_tid, dropped_pairs, recovered_ids, id_switch_est = appearance_associate(
                    state, boxes_list, det_embs, det_hists, frame_idx, W, H, next_tid, cost_debug_acc, H_cam=H_cam,
                    fps=float(fps), det_confs=det_confs_list,
                    tm=tm, det_masks=det_masks, det_mask_boxes=det_mask_boxes, det_vis=det_vis
                )
            except Exception as e:
                if DIAG:
                    _diag_emit("aa_exception", frame=int(frame_idx), err_type=type(e).__name__, err=str(e))
                # graceful fallback: mark all as unassigned so downstream pipeline still runs
                assigned_ids = [-1] * len(boxes_list)
                dropped_pairs = []
                recovered_ids = 0
                id_switch_est = 0
            detections.tracker_id = np.array(assigned_ids, dtype=np.int64)
            if smoother is not None:
                detections = smoother.update_with_detections(detections)
            created_total += max(0, next_tid - prev_next)
            dropped_total += len(dropped_pairs)
            completed_lengths.extend([age for (_tid, age) in dropped_pairs if age is not None])
            recovered_total += recovered_ids
            id_switch_est_total += id_switch_est
            present_ids = set([int(x) for x in assigned_ids if int(x) >= 0])
        else:
            # ByteTrack path
            detections = tracker.update_with_detections(detections)  # adds tracker_id
            # ---- Smooth (optional)
            if smoother is not None:
                detections = smoother.update_with_detections(detections)
            # ---- ID Re-assignment (stitching) to keep IDs across short detector dropouts
            try:
                if detections.xyxy is not None and len(detections) > 0 and ID_REASSIGN_WINDOW > 0:
                    xyxy_now = detections.xyxy.astype(float)
                    # Current centers
                    cx_now = (xyxy_now[:, 0] + xyxy_now[:, 2]) * 0.5
                    cy_now = (xyxy_now[:, 1] + xyxy_now[:, 3]) * 0.5
                    # Prepare tracker_id array
                    if detections.tracker_id is None:
                        det_ids = np.full((len(detections),), -1, dtype=int)
                    else:
                        det_ids = detections.tracker_id.astype(int)

                    # Build ghosts: recently missing tracks
                    ghosts: List[Tuple[int, Dict[str, Any]]] = []
                    for gid, s in state.items():
                        miss = frame_idx - int(s.get("last_frame", -999999))
                        if 1 <= miss <= ID_REASSIGN_WINDOW:
                            ghosts.append((gid, s))
                    if ghosts:
                        # Precompute frame diagonal threshold
                        diag = float(np.hypot(W, H))
                        max_dist = float(ID_REASSIGN_DIST_RATIO) * diag

                        # For each ghost, predict bbox after `miss` frames and keep for matching
                        ghost_preds: List[Tuple[
                            int, Tuple[float, float, float, float], Tuple[float, float], Optional[np.ndarray], Optional[
                                np.ndarray]]] = []
                        for gid, s in ghosts:
                            miss = frame_idx - int(s.get("last_frame", frame_idx))
                            cx, cy = s["last_center"]
                            vx, vy = s.get("last_vel", (0.0, 0.0))
                            # Extrapolate center by miss steps
                            pcx = cx + vx * float(miss)
                            pcy = cy + vy * float(miss)
                            w, h = _bbox_wh(s["last_bbox"])  # keep size stable during short miss
                            pb = _bbox_from_center_wh(pcx, pcy, w, h)
                            ghost_hist = s.get("hist") if APPEAR_ENABLE else None
                            ghost_emb = s.get("emb") if APPEAR_EMB_ENABLE else None
                            ghost_preds.append((gid, pb, (pcx, pcy), ghost_hist, ghost_emb))

                        # Pre-compute appearance for current detections (optional)
                        det_hists: List[Optional[np.ndarray]] = []
                        if APPEAR_ENABLE:
                            for di in range(len(detections)):
                                db = tuple(xyxy_now[di].tolist())  # type: ignore
                                det_hists.append(_compute_hsv_hist(frame, db))
                        else:
                            det_hists = [None] * len(detections)

                        det_embs: List[Optional[np.ndarray]] = []
                        if APPEAR_EMB_ENABLE:
                            # batched
                            boxes_list: List[Tuple[float, float, float, float]] = [tuple(x) for x in xyxy_now.tolist()]
                            det_embs = _compute_embeddings_for_dets(frame, boxes_list, REID_BATCH_SIZE)
                        else:
                            det_embs = [None] * len(detections)

                        # Build candidate matches (det_idx, ghost_id, combined_sim, iou)
                        cands: List[Tuple[float, float, int, int]] = []  # (-combined_sim, -iou, det_idx, ghost_idx)
                        for det_idx in range(len(detections)):
                            # Only consider if this det has an ID that likely started anew or is undefined
                            if det_ids[det_idx] >= 0 and det_ids[det_idx] in state and state[det_ids[det_idx]].get(
                                    "last_frame", -1) == frame_idx:
                                # This ID already existed this frame; skip to not fight tracker
                                continue
                            db = tuple(xyxy_now[det_idx].tolist())  # type: ignore
                            dcx, dcy = float(cx_now[det_idx]), float(cy_now[det_idx])
                            det_hist = det_hists[det_idx] if det_hists else None
                            det_emb = det_embs[det_idx] if det_embs else None
                            for ghost_idx, (gid, gb, (gcx, gcy), ghost_hist, ghost_emb) in enumerate(ghost_preds):
                                iou = _iou_xyxy(db, gb)
                                if iou < float(ID_REASSIGN_IOU):
                                    continue
                                dist = float(np.hypot(dcx - gcx, dcy - gcy))
                                if dist > max_dist:
                                    continue
                                # Embedding similarity (preferred)
                                emb_sim = _cosine_sim(det_emb, ghost_emb)
                                if emb_sim is not None and emb_sim < APPEAR_EMB_MIN_SIM:
                                    continue
                                # Histogram similarity (fallback/aux)
                                hist_sim = _hist_similarity(det_hist, ghost_hist)
                                if emb_sim is None and hist_sim is not None and hist_sim < APPEAR_MIN_SIM:
                                    continue
                                # Combine appearance similarities
                                sim_emb_val = emb_sim if emb_sim is not None else 0.0
                                sim_hist_val = hist_sim if hist_sim is not None else 0.0
                                combined = (APPEAR_SIM_W_EMB * sim_emb_val) + (APPEAR_SIM_W_HIST * sim_hist_val)
                                cands.append((-combined, -iou, det_idx, ghost_idx))

                        if cands:
                            cands.sort()
                            used_dets = set()
                            used_ghosts = set()
                            for neg_combined, neg_iou, det_idx, ghost_idx in cands:
                                if det_idx in used_dets or ghost_idx in used_ghosts:
                                    continue
                                gid = ghost_preds[ghost_idx][0]
                                # Assign ghost ID to this detection
                                det_ids[det_idx] = int(gid)
                                used_dets.add(det_idx)
                                used_ghosts.add(ghost_idx)
                                recovered_count_frame += 1

                            # Write back reassigned IDs
                            detections.tracker_id = det_ids.astype(np.int64)
            except Exception as _e:
                # Fail-safe: never break the pipeline due to stitching
                pass
            recovered_total += recovered_count_frame
            # present_ids is rebuilt below per frame, no need to pre-initialize here

        # MOT export accumulation
        if MOT_EXPORT and detections.xyxy is not None and detections.tracker_id is not None:
            fnum = int(frame_idx + 1)
            items: List[Tuple[int, float, float, float, float]] = []
            for i in range(len(detections)):
                tid_i = int(detections.tracker_id[i])
                if tid_i < 0:
                    continue
                x1, y1, x2, y2 = detections.xyxy[i].astype(float).tolist()
                items.append((tid_i, float(x1), float(y1), float(x2), float(y2)))
            tracks_per_frame[fnum] = items

        present_ids = set()

        # Iterate detections (also updates state)
        prev_ids_for_metrics = set(state.keys())
        if detections.xyxy is not None and len(detections) > 0:
            for i in range(len(detections)):
                bbox = detections.xyxy[i].astype(float).tolist()
                conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                label_idx = int(detections.class_id[i]) if detections.class_id is not None else -1
                tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1

                if tid < 0:
                    continue
                # Confidence gate (safety; also applied pre-tracker)
                if conf < conf_thr:
                    continue

                cx, cy = _bbox_center(tuple(bbox))
                # If ROI was not enforced earlier, apply now
                if not ROI_ENFORCE and not _point_in_roi(cx, cy):
                    continue

                present_ids.add(tid)

                # If appearance-aware association already updated state for this frame, only append trail to avoid double EMA/age updates
                if ASSOC_APPEAR_ENABLE:
                    prev = state.get(tid)
                    if prev is not None and int(prev.get("last_frame", -1)) == frame_idx:
                        trail = prev.get("trail", [])
                        trail.append((cx, cy))
                        if len(trail) > max(2, TRAIL_LENGTH):
                            trail = trail[-TRAIL_LENGTH:]
                        prev["trail"] = trail
                        state[tid] = prev
                        # Draw only if enabled
                        if DRAW_DEBUG:
                            x1, y1, x2, y2 = bbox
                            p1 = (int(max(0, x1)), int(max(0, y1)))
                            p2 = (int(min(W - 1, x2)), int(min(H - 1, y2)))
                            color = _track_color(tid if tid >= 0 else 0)
                            cv2.rectangle(frame, p1, p2, color, 2)
                            cv2.putText(frame, f"ID {tid}", (p1[0], max(0, p1[1] - 7)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                        continue

                # --- maintain simple state (trail + extrapolation), and appearance EMA ---
                prev = state.get(tid)
                if prev is None:
                    vel = (0.0, 0.0)
                    trail = [(cx, cy)]
                    age = 1
                    hits = 1
                else:
                    pcx, pcy = prev.get("last_center", (cx, cy))
                    vel_inst = (cx - pcx, cy - pcy)
                    prev_vel = prev.get("last_vel", (0.0, 0.0))
                    alpha = 0.6
                    vel = (
                        prev_vel[0] * (1.0 - alpha) + vel_inst[0] * alpha,
                        prev_vel[1] * (1.0 - alpha) + vel_inst[1] * alpha,
                    )
                    trail = prev.get("trail", [])
                    trail.append((cx, cy))
                    if len(trail) > max(2, TRAIL_LENGTH):
                        trail = trail[-TRAIL_LENGTH:]
                    age = int(prev.get("age", 0)) + 1
                    hits = int(prev.get("hits", 0)) + 1

                # Appearance updates (EMA): histogram and deep embedding
                hist = prev.get("hist") if prev is not None else None
                if APPEAR_ENABLE:
                    if ASSOC_APPEAR_ENABLE and 'det_hists' in locals() and i < len(det_hists):
                        hist_now = det_hists[i]
                    else:
                        hist_now = _compute_hsv_hist(frame, tuple(bbox))
                    if hist_now is not None:
                        if hist is not None and isinstance(hist, np.ndarray) and hist.shape == hist_now.shape:
                            hist = (1.0 - APPEAR_ALPHA) * hist + APPEAR_ALPHA * hist_now
                            try:
                                cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
                            except Exception:
                                pass
                        else:
                            hist = hist_now
                emb = prev.get("emb") if prev is not None else None
                if APPEAR_EMB_ENABLE:
                    if ASSOC_APPEAR_ENABLE and 'det_embs' in locals() and i < len(det_embs):
                        emb_now = det_embs[i]
                    else:
                        emb_now = _compute_embedding(frame, tuple(bbox))
                    if emb_now is not None:
                        if emb is not None and isinstance(emb, np.ndarray) and emb.shape == emb_now.shape:
                            emb = (1.0 - APPEAR_EMB_ALPHA) * emb + APPEAR_EMB_ALPHA * emb_now
                            n = np.linalg.norm(emb)
                            if n > 1e-12:
                                emb = emb / n
                        else:
                            emb = emb_now

                state[tid] = {
                    "last_bbox": tuple(bbox),
                    "last_center": (cx, cy),
                    "last_vel": vel,
                    "last_frame": frame_idx,
                    "trail": trail,
                    "hist": hist,
                    "emb": emb,
                    "age": age,
                    "hits": hits,
                    "time_since_update": 0,
                }

                # Follow export per-ID (lazy init and write)
                if FOLLOW_EXPORT and (follow_root is not None) and (tid >= 0):
                    try:
                        age_now = int(state[tid].get("age", 0))
                        if age_now >= FOLLOW_MIN_AGE:
                            if tid not in follow_writers:
                                # Open writer lazily
                                try:
                                    fpath = os.path.join(follow_root, f"id_{tid:04d}.mp4")
                                    fourcc_f = cv2.VideoWriter_fourcc(*'mp4v')
                                    fw = cv2.VideoWriter(fpath, fourcc_f, fps, (FOLLOW_SIZE, FOLLOW_SIZE))
                                    if (fw is None) or (not fw.isOpened()):
                                        fourcc_f2 = cv2.VideoWriter_fourcc(*'avc1')
                                        fw = cv2.VideoWriter(fpath, fourcc_f2, fps, (FOLLOW_SIZE, FOLLOW_SIZE))
                                    if (fw is not None) and fw.isOpened():
                                        follow_writers[tid] = fw
                                        follow_paths[tid] = fpath
                                except Exception:
                                    pass
                            fw = follow_writers.get(tid, None)
                            if fw is not None and fw.isOpened():
                                try:
                                    pb = _pad_bbox(tuple(bbox), FOLLOW_PAD, W, H)
                                    roi = _safe_crop(frame, pb)
                                    if roi is not None and roi.size > 0:
                                        roi_sq = cv2.resize(roi, (FOLLOW_SIZE, FOLLOW_SIZE))
                                        fw.write(roi_sq)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # Draw only if enabled
                if DRAW_DEBUG:
                    x1, y1, x2, y2 = bbox
                    p1 = (int(max(0, x1)), int(max(0, y1)))
                    p2 = (int(min(W - 1, x2)), int(min(H - 1, y2)))
                    color = _track_color(tid if tid >= 0 else 0)
                    cv2.rectangle(frame, p1, p2, color, 2)
                    cv2.putText(frame, f"ID {tid}", (p1[0], max(0, p1[1] - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Increment time_since_update for missing tracks and drop expired
        to_drop = []
        for tid, s in state.items():
            if tid not in present_ids:
                s["time_since_update"] = int(s.get("time_since_update", 0)) + 1
                if s["time_since_update"] > BT_BUFFER:
                    to_drop.append(tid)
        for tid in to_drop:
            age = int(state[tid].get("age", 0))
            completed_lengths.append(age)
            dropped_total += 1
            del state[tid]

        # Created metric in ByteTrack path: count present IDs not previously seen
        if not (ASSOC_APPEAR_ENABLE and detections.xyxy is not None and len(detections) > 0):
            created_total += sum(1 for tid in present_ids if tid not in prev_ids_for_metrics and tid >= 0)
            id_switch_est_total += 0  # heuristic handled in assoc or could be added here if needed

        # --- Short occlusion drawing: predict for a few frames if a recent ID is missing ---
        if DRAW_DEBUG and DRAW_PRED_FOR > 0:
            for tid, s in state.items():
                if tid in present_ids:
                    continue
                miss = frame_idx - int(s.get("last_frame", frame_idx))
                if 1 <= miss <= DRAW_PRED_FOR:
                    cx, cy = s.get("last_center", (0, 0))
                    vx, vy = s.get("last_vel", (0.0, 0.0))
                    w, h = _bbox_wh(s.get("last_bbox", (0, 0, 1, 1)))
                    pcx = cx + vx
                    pcy = cy + vy
                    px1, py1, px2, py2 = _bbox_from_center_wh(pcx, pcy, w, h)
                    color = _track_color(tid)
                    p1 = (int(max(0, px1)), int(max(0, py1)))
                    p2 = (int(min(W - 1, px2)), int(min(H - 1, py2)))
                    cv2.rectangle(frame, p1, p2, color, 1)
                    cv2.putText(frame, f"ID {tid} (pred)", (p1[0], max(0, p1[1] - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # --- Trails ---
        if DRAW_DEBUG and DRAW_TRAILS:
            for tid, s in state.items():
                tr = s.get("trail", [])
                if len(tr) >= 2:
                    pts = np.array([[int(x), int(y)] for (x, y) in tr], dtype=np.int32)
                    cv2.polylines(frame, [pts], isClosed=False, color=_track_color(tid), thickness=2)

        # ROI overlay
        if DRAW_DEBUG:
            _draw_roi_overlay(frame)

        # HUD (debug only)
        if DRAW_DEBUG:
            hud = f"sv-{'AA' if ASSOC_APPEAR_ENABLE else 'BT'} | frame {frame_idx}"
            modes = []
            has_reid = ((('_REID_PLUG' in globals()) and (globals().get('_REID_PLUG') is not None)) or (
                        _REID_MODEL is not None))
            if APPEAR_EMB_ENABLE and has_reid:
                modes.append("emb")
            if APPEAR_ENABLE:
                modes.append("hist")
            if modes:
                hud += " | reid: " + "+".join(modes)
            else:
                hud += " | reid: none"
            if TRACK_CLASSES is not None:
                hud += f" | classes {TRACK_CLASSES}"
            cv2.putText(frame, hud, (10, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        if writer is not None and not DRY_RUN:
            if EVAL_RENDER:
                _render_eval(frame, detections, state, W, H, draw_trails=EVAL_RENDER_TRAILS)
            writer.write(frame)
        # Periodic compact logging
        if LOG_EVERY > 0 and ((frame_idx + 1) % LOG_EVERY == 0):
            logger.info(
                f"[track] frame={frame_idx + 1} active={len(state)} created={created_total} dropped={dropped_total} recovered={recovered_total} est_id_switches={id_switch_est_total}")

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    # Follow writers cleanup
    try:
        if FOLLOW_EXPORT and follow_writers:
            for _tid, _fw in list(follow_writers.items()):
                try:
                    if _fw is not None:
                        _fw.release()
                except Exception:
                    pass
            follow_writers.clear()
    except Exception:
        pass

    # Compute summary metrics
    total_frames = frame_idx + 1
    # include active tracks' ages
    for tid, s in state.items():
        age = int(s.get("age", 0))
        if age > 0:
            completed_lengths.append(age)
    avg_len = float(np.mean(completed_lengths)) if completed_lengths else 0.0
    mode = "appearance-aware" if ASSOC_APPEAR_ENABLE else "bytetrack"

    # Optional association cost debug dump
    if ASSOC_COST_DEBUG and cost_debug_acc is not None and cost_debug_acc.get("pairs", 0.0) > 0:
        iou_avg = cost_debug_acc["iou_sum"] / cost_debug_acc["pairs"]
        emb_avg = cost_debug_acc["emb_sum"] / cost_debug_acc["pairs"]
        hist_avg = cost_debug_acc["hist_sum"] / cost_debug_acc["pairs"]
        try:
            ensure_dir(CAPTURES_DIR)
            csv_path = os.path.join(CAPTURES_DIR, f"assoc_cost_debug_{base}.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("metric,value\n")
                f.write(f"iou_avg,{iou_avg:.6f}\n")
                f.write(f"emb_avg,{emb_avg:.6f}\n")
                f.write(f"hist_avg,{hist_avg:.6f}\n")
        except Exception:
            pass

    # MOT export
    mot_path = None
    try:
        if MOT_EXPORT and len(tracks_per_frame) > 0:
            ensure_dir(CAPTURES_DIR)
            mot_path = os.path.join(CAPTURES_DIR, f"tracks_{base}.txt")
            write_mot_txt(mot_path, tracks_per_frame, W, H)
    except Exception:
        mot_path = None

    # Compute GMC used ratio
    total_gmc_decisions = max(1, (gmc_used + gmc_skips))
    gmc_used_ratio = float(gmc_used) / float(total_gmc_decisions)

    abs_video = os.path.abspath(out_path) if out_path else None
    abs_mot = os.path.abspath(mot_path) if mot_path else None

    metrics = {
        "video": abs_video if (writer is not None and WRITE_VIDEO and not DRY_RUN) else None,
        "frames": total_frames,
        "tracks_created": created_total,
        "tracks_dropped": dropped_total,
        "avg_track_length": round(avg_len, 2),
        "recovered_ids": recovered_total,
        "est_id_switches": id_switch_est_total,
        "mot_txt": abs_mot,
        "gmc_used_ratio": round(gmc_used_ratio, 4),
        "adaptive_weight": bool(ADAPTIVE_WEIGHT),
        "preset": PRESET,
    }

    # Final single-line minimal log
    video_str = metrics['video'] if metrics['video'] else "None"
    mot_str = metrics['mot_txt'] if metrics['mot_txt'] else "None"
    if DIAG:
        _diag_emit("done", frames=int(metrics["frames"]),
                   created=int(metrics["tracks_created"]),
                   dropped=int(metrics["tracks_dropped"]),
                   recovered=int(metrics["recovered_ids"]),
                   id_switches=int(metrics["est_id_switches"]),
                   mode=("AA" if ASSOC_APPEAR_ENABLE else "BT"))
    try:
        global _diag_fp
        if _diag_fp is not None:
            _diag_fp.close()
            _diag_fp = None
    except Exception:
        pass
    logger.info(
        f"[done] frames={metrics['frames']} created={metrics['tracks_created']} dropped={metrics['tracks_dropped']} avg_len={metrics['avg_track_length']} recovered={metrics['recovered_ids']} est_id_switches={metrics['est_id_switches']} mode={mode} video={video_str} mot={mot_str}")
    return metrics


# -----------------------------
# Main (notebook)
# -----------------------------
def run_pipeline_notebook():
    logger.info(
        f"[✓] YOLO + Supervision ByteTrack (conf={TRACK_CONF}, iou={TRACK_IOU}, slicer={'on' if SLICER_ENABLE else 'off'})")
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    data = run_tracking_with_supervision()
    return data


