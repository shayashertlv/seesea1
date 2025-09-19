# Association Data Collection

This folder documents how to harvest high-quality training samples for the transformer association module.  The runtime tracker can stream its assignment state into `.npz` snapshots without disrupting your normal workflow.

## 1. Configure the tracker

Set the following environment variables before launching `python -m cli` (or your custom entry point):

```bash
export TRANSFORMER_LOG_DIR="/workspace/seesea1/artifacts/assoc_logs"
export TRANSFORMER_LOG_EMBED_DIM=256        # optional cap, matches default model
export APPEAR_MEMORY_ENABLE=1               # enables prototype tracking for better labels
```

The logging directory is created automatically.  Each frame that contains active tracks and detections writes a compressed sample `sample_XXXXXXX.npz` plus an entry in `manifest.jsonl`.

## 2. Quality checklist while collecting

* **Detector confidence** – keep `CONFIDENCE_THRESHOLD` at your usual runtime value so the logged detections reflect production behaviour.
* **Visibility signals** – enable segmentation when available (`SEG_ENABLE=1`).  The logger stores visibility estimates so you can filter low-quality crops during training.
* **Anti-switch diagnostics** – keep `DIAG=1` to monitor ID switch counts.  Skip or relabel segments with high `est_id_switches` before training.
* **Balanced coverage** – record a mix of easy and hard scenes (crowding, occlusion, fast motion).  Aim for at least 5–10k matched pairs before the first training run.

## 3. Inspecting the collected data

The `.npz` schema contains:

* `track_features` – normalized geometry/motion features for each active track.
* `det_features` – geometry + confidence/visibility for each detection.
* `track_embeddings` / `det_embeddings` – padded appearance descriptors (zero when unavailable).
* `cost_matrix` / `mask_matrix` – ByteTrack-style costs and gating masks prior to assignment.
* `assigned_track_ids` – the tracker’s final decisions (track ID or `-1` when unmatched).

Use a quick notebook or the helper in `training/train_assoc.py` (`--inspect` flag) to sanity check a few samples before training.

## 4. Cleaning up

When you are done collecting data you can archive or prune the directory.  The manifest lists every sample so you can subset with standard tooling (e.g., `jq`, `python` scripts) without touching the individual `.npz` files.
