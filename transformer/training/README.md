# Training the Association Transformer

Once you have collected association samples you can train (or fine-tune) the transformer-based cost head.  The `train_assoc.py` script handles dataset loading, batching, validation splits, and checkpoint export.

## 1. Installation

Ensure the repo dependencies are installed and that PyTorch is available.  The default configuration uses GPU acceleration when possible.

```bash
pip install -r requirements.txt
# optional: pip install timm open_clip if you plan to fine-tune ReID backbones separately
```

## 2. Launching a training run

```bash
python -m transformer.training.train_assoc \
    --data /workspace/seesea1/artifacts/assoc_logs \
    --output /workspace/seesea1/weights/assoc_transformer.pt \
    --epochs 12 \
    --batch-size 64 \
    --lr 1e-4
```

Key arguments:

* `--data` – directory created by the logger (must contain `manifest.jsonl`).
* `--output` – path for the `.pt` checkpoint (contains weights + metadata for runtime loading).
* `--val-split` – fraction of samples reserved for validation (default `0.1`).
* `--device` – `cuda`, `cpu`, or `auto` (default `auto`).
* `--inspect` – print dataset statistics and exit (no training).

The script reports loss curves and simple accuracy metrics every epoch.  Validation loss is used to keep the best checkpoint automatically.

## 3. Using the trained weights

1. Copy the resulting `.pt` file to a stable location (e.g., `/workspace/seesea1/weights/assoc_transformer.pt`).
2. Set environment variables before running the tracker:

```bash
export TRANSFORMER_ASSOC_ENABLE=1
export TRANSFORMER_ASSOC_WEIGHTS="/workspace/seesea1/weights/assoc_transformer.pt"
export TRANSFORMER_ASSOC_WEIGHT=0.5   # adjust blend factor if needed
# optional: export TRANSFORMER_ASSOC_EMBED_DIM=256  # must match TRANSFORMER_LOG_EMBED_DIM if set
```

At runtime the tracker will blend the learned association scores with the existing IoU/ReID/HSV costs.  Set `TRANSFORMER_ASSOC_WEIGHT` closer to `1.0` to rely more on the transformer or closer to `0.0` to favour the classical costs.

> **Important:** the checkpoint stores the embedding width that was used during logging/training.  Leave
> `TRANSFORMER_ASSOC_EMBED_DIM` unset (preferred) or set it to the same value as `TRANSFORMER_LOG_EMBED_DIM`.  A mismatch
> now triggers a runtime error so the tracker cannot accidentally drop appearance information.

## 4. Evaluation tips

* Re-run the tracker on validation clips with and without the transformer enabled.  Compare IDF1, HOTA, and ID switch counts.
* Inspect qualitative outputs: the logger can remain active so you can build a second dataset for iterative training.
* Keep an eye on failure modes (e.g., when the transformer disagrees with handcrafted logic) and adjust the blend weight or retrain with more targeted samples.
