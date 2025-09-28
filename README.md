# seesea tracking

This repository provides a compact object detection and multi-object tracking
pipeline tailored for surf footage.  The goal of the cleanup is to expose a
single, easy-to-follow flow that describes every moving part:

1. **Configuration** lives in `config.py` and is driven by environment
   variables.  The dataclasses defined there document every threshold and what
   happens when it changes.
2. **Detection** is handled in `tracker.py` by a thin wrapper around an
   Ultralytics YOLO model.  There are no fallbacks – if the model cannot be
   loaded an informative exception is raised.
3. **Appearance features** are computed by `reid_backbones.ReIDExtractor`.
   Instead of juggling multiple backends we now rely on a single HSV histogram
   encoder that works with just NumPy and OpenCV.
4. **Tracking** combines the detections and the appearance features inside
   `tracker.SimpleTracker`.  The association logic is intentionally explicit so
   that it is straightforward to reason about its decisions.

The command line entry point (`cli.py`) forwards to
`tracker.run_pipeline_notebook`, which in turn uses the same set of
configurable building blocks.  Tests under `tests/` cover the public APIs and
the most important behavioural guarantees (configuration parsing, ReID feature
consistency, and tracker exports).

## Installation

Install the minimal runtime dependencies before using the library:

```bash
pip install -r requirements.txt
```

Ultralytics and supervision are optional at import time, but are required when
`run_tracking_with_supervision` is executed.  They can be added with:

```bash
pip install ultralytics supervision
```

## Example

```bash
python -m cli --log-level DEBUG --video /path/to/video.mp4
```

## Tests

Run the test-suite with:

```bash
pytest
```
