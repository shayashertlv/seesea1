import json
import pathlib

import numpy as np
import pytest

from tools import eval_reid


def test_compute_reid_metrics_simple_case(tmp_path: pathlib.Path) -> None:
    # Two identities observed from different cameras. Query 0 matches gallery 0 and 2.
    q_feats = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    q_ids = np.array([0, 1], dtype=np.int64)
    q_cams = np.array([0, 1], dtype=np.int64)

    g_feats = np.array(
        [
            [1.0, 0.0],  # same as q0
            [0.0, 1.0],  # same as q1
            [1.0, 0.0],  # duplicate identity for q0, different camera
        ],
        dtype=np.float32,
    )
    g_ids = np.array([0, 1, 0], dtype=np.int64)
    g_cams = np.array([1, 0, 2], dtype=np.int64)

    metrics = eval_reid.compute_reid_metrics(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, topk=3)
    assert pytest.approx(metrics["mAP"], rel=1e-5) == 1.0
    assert metrics["valid_queries"] == 2
    assert pytest.approx(metrics["CMC"]["rank-1"], rel=1e-5) == 1.0
    assert pytest.approx(metrics["CMC"]["rank-2"], rel=1e-5) == 1.0

    # Validate serialization when using temporary npz files
    query_path = tmp_path / "query.npz"
    gallery_path = tmp_path / "gallery.npz"
    np.savez(query_path, features=q_feats, labels=q_ids, camids=q_cams)
    np.savez(gallery_path, embeddings=g_feats, ids=g_ids, cameras=g_cams)
    metrics_from_files = eval_reid.compute_reid_metrics(*eval_reid.load_embeddings(query_path), *eval_reid.load_embeddings(gallery_path))
    assert metrics_from_files["valid_queries"] == 2


def test_compute_idf1_simple_case(tmp_path: pathlib.Path) -> None:
    gt_path = tmp_path / "gt.txt"
    pred_path = tmp_path / "pred.txt"

    # Frame 1: perfect match
    # Frame 2: one missed detection
    gt_rows = ["1,1,0,0,10,10", "2,1,0,0,10,10"]
    pred_rows = ["1,10,0,0,10,10", "2,10,100,100,10,10"]

    gt_path.write_text("\n".join(gt_rows))
    pred_path.write_text("\n".join(pred_rows))

    metrics = eval_reid.compute_idf1(gt_path, pred_path, iou_threshold=0.5)
    # IDTP = 1, IDFP = 1 (second frame false positive), IDFN = 1 (missed gt)
    assert pytest.approx(metrics["IDF1"], rel=1e-6) == 0.5
    assert pytest.approx(metrics["IDP"], rel=1e-6) == 0.5
    assert pytest.approx(metrics["IDR"], rel=1e-6) == 0.5

    # CLI smoke test storing JSON output
    output_path = tmp_path / "metrics.json"
    cli_metrics = eval_reid.main(
        [
            "--output",
            str(output_path),
            "idf1",
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--iou-threshold",
            "0.5",
        ]
    )
    assert output_path.exists()
    stored = json.loads(output_path.read_text())
    assert stored["IDF1"] == pytest.approx(cli_metrics["IDF1"], rel=1e-6)
