import numpy as np

from appearance_memory import AppearanceMemory


def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _best_assignment(track_vecs, det_vecs):
    sims = np.zeros((len(track_vecs), len(det_vecs)), dtype=np.float32)
    for i, tv in enumerate(track_vecs):
        t = _normalize(tv)
        for j, dv in enumerate(det_vecs):
            sims[i, j] = float(np.clip(np.dot(t, _normalize(dv)), -1.0, 1.0))
    # Two-track scenario: choose between direct and swapped assignment
    score_direct = float(sims[0, 0] + sims[1, 1])
    score_swap = float(sims[0, 1] + sims[1, 0])
    if score_direct >= score_swap:
        return [0, 1]
    return [1, 0]


class SimpleLatestMemory:
    """Baseline memory that keeps only the latest embedding."""

    def __init__(self):
        self._store = {}

    def get(self, tid: int):
        return self._store.get(int(tid))

    def update(self, tid: int, vec: np.ndarray, quality: dict):
        if vec is None:
            return
        self._store[int(tid)] = _normalize(vec)


def _simulate(memory, frames, quality):
    assignments = []
    for idx, (det1, det2) in enumerate(frames):
        m1 = memory.get(1)
        m2 = memory.get(2)
        if m1 is None:
            m1 = det1
        if m2 is None:
            m2 = det2
        assign = _best_assignment([m1, m2], [det1, det2])
        assignments.append(assign)
        q = dict(quality)
        q.update({"frame_idx": idx})
        memory.update(1, det1, q)
        memory.update(2, det2, q)
    return assignments


def test_time_decayed_queue_reduces_switches():
    rng = np.random.default_rng(42)
    v1 = _normalize(np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32))
    v2 = _normalize(np.array([0.0, 1.0, 0.1, 0.0], dtype=np.float32))

    def jitter(base, scale=0.01):
        return _normalize(base + rng.normal(0.0, scale, size=base.shape))

    frames = [
        (jitter(v1), jitter(v2)),
        (jitter(v1), jitter(v2)),
        (jitter(v1), jitter(v2)),
        # Occlusion: embeddings swap appearance
        (jitter(v2, scale=0.02), jitter(v1, scale=0.02)),
        # Recovery
        (jitter(v1), jitter(v2)),
    ]

    quality = {"vis": 0.95, "min_side": 96.0, "conf": 0.90, "blur": 0.5}

    baseline = SimpleLatestMemory()
    queue_mem = AppearanceMemory(
        backend="queue",
        k_protos=4,
        time_decay=60.0,
        max_age=30,
        min_vis=0.5,
        min_side=64,
        max_blur=4.0,
        alpha=0.3,
    )

    baseline_assign = _simulate(baseline, frames, quality)
    queue_assign = _simulate(queue_mem, frames, quality)

    gt = [0, 1]
    baseline_switches = sum(1 for a in baseline_assign if a != gt)
    queue_switches = sum(1 for a in queue_assign if a != gt)

    assert baseline_switches > queue_switches
    assert queue_switches <= 1
