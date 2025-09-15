from models.seqtrack import SeqTrackLSTM


def test_seqtracklstm_instantiation() -> None:
    model = SeqTrackLSTM()
    assert hasattr(model, "predict")
    assert hasattr(model, "device")
