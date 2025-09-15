# seesea tracking

This package splits the former `supervision_n_yolo.py` monolith into a small
library. Components can be imported individually without running the tracking
pipeline.

## Modules

- `config.py` – environment driven configuration and logging helpers.
- `models/seqtrack.py` – neural network definitions such as `SeqTrackLSTM`.
- `tracker.py` – tracking loop and high level helpers.
- `cli.py` – command line entry point wrapping the tracker.

## Example

```bash
python -m cli --log-level DEBUG
```

## Thread Safety

`ReIDExtractor` maintains a shared PCA cache that is protected by a
threading lock. It is safe to call `forward` or other embedding helpers
from multiple threads within the same process. The cache uses a bounded
buffer to avoid unbounded memory growth.

## Surfer ReID fine-tuning

A lightweight training script is available under `training/train_surfer_reid.py`.
It fine‑tunes one of the backbones defined in `reid_backbones.py` on a surfers
dataset organised as an [ImageFolder](https://pytorch.org/vision/stable/datasets.html#imagefolder).

1. Export the destination path for fine‑tuned weights:

   ```bash
   export REID_WEIGHTS=/path/to/surfer_weights.pth
   ```

2. Run training:

   ```bash
   python training/train_surfer_reid.py --data /path/to/dataset --backbone osnet
   ```

`ReIDExtractor` will automatically load weights from `REID_WEIGHTS` if the file
exists, enabling domain‑specific features without additional code changes.
