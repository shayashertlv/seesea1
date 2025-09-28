from config import TrackerConfig, load_config


def test_load_config_defaults() -> None:
    cfg = load_config()
    assert isinstance(cfg, TrackerConfig)
    assert cfg.detection.confidence == 0.30
    assert cfg.association.min_iou == 0.35
    explanations = cfg.describe_thresholds()
    assert "detection" in explanations
    assert "confidence" in explanations["detection"]
