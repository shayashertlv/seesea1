import pytest

np = pytest.importorskip("numpy")

from global_reid_bank import GlobalReIDBank, _HAS_SCIPY_KDTREE


@pytest.mark.skipif(not _HAS_SCIPY_KDTREE, reason="SciPy KDTree backend is unavailable")
def test_global_reid_bank_scipy_backend_query_returns_cosine_similarity():
    bank = GlobalReIDBank(dim=4, backend="scipy")
    if bank.backend != "scipy":
        pytest.skip("SciPy backend unavailable at runtime")

    vecs = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    ]
    timestamps = [0, 1, 2]

    for idx, (vec, ts) in enumerate(zip(vecs, timestamps), start=1):
        bank.add(idx, vec, ts)

    # Query with relaxed similarity threshold to keep all candidates.
    results = bank.query(vecs[0], top_k=3, sim_threshold=0.0)

    assert [tid for tid, *_ in results] == [1, 2, 3]
    # Cosine similarity for identical vector should be 1.0
    assert results[0][1] == pytest.approx(1.0, rel=1e-6)
    # Cosine similarity between vecs[0] and vecs[1] equals 0.8
    assert results[1][1] == pytest.approx(0.8, rel=1e-6)
