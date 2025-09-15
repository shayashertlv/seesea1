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
