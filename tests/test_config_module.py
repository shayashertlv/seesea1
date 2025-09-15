from config import TrackerConfig, load_config


def test_load_config_defaults() -> None:
    cfg = load_config()
    assert isinstance(cfg, TrackerConfig)
    assert cfg.track_conf == 0.35
    assert cfg.track_iou == 0.9
