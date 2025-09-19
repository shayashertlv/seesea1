# Transformer Integration Roadmap

This package groups everything related to transformer-based ID tracking for the surfing project.  It is intentionally separated from the runtime tracker so that data collection, training, and evaluation workflows stay organized and reproducible.

## Directory Layout

```
transformer/
├── README.md                 # Overview (this file)
├── data_collection/          # Scripts + docs for gathering association data
├── models/                   # Lightweight transformer models used at runtime
└── training/                 # Training & evaluation entry points
```

The runtime tracker (`tracker.py`) already knows how to import the transformer modules when they are available.  You only need to set the relevant environment variables (described below) to enable logging, training-data harvesting, or the learned association module.

## Quick Start

1. **Collect data** – enable the logging hooks in the tracker, then run your normal inference loop.  Detailed instructions live in `data_collection/README.md`.
2. **Train the association transformer** – once you have enough `.npz` samples, train the model by following `training/README.md` (which wraps `train_assoc.py`).
3. **Enable the transformer at runtime** – point `TRANSFORMER_ASSOC_WEIGHTS` at the saved checkpoint and set `TRANSFORMER_ASSOC_ENABLE=1` when launching the tracker.  The model will seamlessly blend its learned costs with the existing appearance+motion solver.

## Relevant Environment Variables

| Variable | Purpose |
| --- | --- |
| `TRANSFORMER_LOG_DIR` | Directory for collected association samples.  Creates `.npz` files + manifest. |
| `TRANSFORMER_LOG_EMBED_DIM` | Optional cap on embedding dimensionality stored in each sample. |
| `TRANSFORMER_ASSOC_ENABLE` | Enables transformer-based cost blending when set to `1`. |
| `TRANSFORMER_ASSOC_WEIGHTS` | Path to a trained `*.pt` checkpoint produced by `train_assoc.py`. |
| `TRANSFORMER_ASSOC_WEIGHT` | Blend factor between classical costs and transformer predictions (0–1 range). |
| `APPEAR_MEMORY_ENABLE` | Keeps the transformer-aware appearance memory active. |

For a detailed walkthrough of the logging/training commands, see the dedicated READMEs in each subdirectory.
