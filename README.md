# Seesea1

Seesea1 provides lightweight building blocks for video analytics and multi-object tracking pipelines.  Modules cover global motion compensation, pluggable re-identification backbones, ROI segmentation helpers and crop super-resolution.  Each component is designed to fail open and fall back to simplified versions when heavy dependencies are missing.

## Setup

The core modules only require Python 3.8+, [NumPy](https://numpy.org/) and [OpenCV](https://opencv.org/):

```bash
pip install numpy opencv-python
```

Optional features may need additional packages:

| Feature | Extra dependencies |
| ------- | ------------------ |
| RAFT GMC | `torch`, RAFT weights and `gmc_raft` |
| ReID backbones | `torch`, `timm`, `open_clip`, `torchreid`, `Pillow` depending on backend |
| Mask2Former segmentation | `detectron2` |
| SAM2/Segment-Anything segmentation | `sam2` or `segment-anything` |
| Super-resolution | `realesrgan`, `torch` and model weights |

## Basic usage

```python
from gmc import estimate_gmc
H, stats = estimate_gmc(prev_gray, curr_gray)

from reid_backbones import ReIDExtractor
extractor = ReIDExtractor(backend="osnet")
embeddings = extractor.forward([crop_bgr], batch_size=32)

from seg_sam2 import infer_roi_masks
masks, boxes, vis = infer_roi_masks(frame_bgr, detections)

from sr_corps import CropSR
sr = CropSR(scale=2).maybe_upscale(crop_bgr)
```

## Optional components

### Global Motion Compensation (GMC)
- Default ORB feature matching via OpenCV.
- Set `method="raft"` to use the RAFT optical-flow backend. Requires PyTorch and RAFT weights; falls back to ORB when unavailable.

### ReID backbones
- `ReIDExtractor` supports multiple backends such as `osnet`, `fastreid_r50`, `dinov2_vits14`, `clip_vitl14` and `fusion`.
- All backends require PyTorch; some need `timm` or `open_clip`.
- Choose backend via `ReIDExtractor(backend="...")`.

### Segmentation helpers
- `seg_mask2former.py` tries to use Mask2Former (`detectron2`). Without it, each ROI is processed with a quick GrabCut fallback.
- `seg_sam2.py` attempts SAM2 or Segment-Anything libraries and falls back to GrabCut if missing.

### Super-resolution
- `CropSR` leverages Real-ESRGAN when installed for upscaling small crops. Without it, bicubic interpolation is used.

## Troubleshooting

- Missing optional dependencies trigger console warnings such as `[seg] SAM2 not available; using GrabCut ROI fallback` or `[sr] Real-ESRGAN unavailable; using bicubic fallback`.
- Install the required packages and verify model weights/paths if you need the heavy backends.
- Ensure GPU availability when using PyTorch-based features (`torch.cuda.is_available()`).
- If imports fail, confirm that the module names are correct and that their versions satisfy your Python environment.

