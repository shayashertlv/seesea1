import os
from pathlib import Path


def test_cli_accepts_video_argument(monkeypatch, tmp_path: Path) -> None:
    import cli

    captured = {}

    def fake_configure_logging(level: str) -> None:
        captured["log_level"] = level

    def fake_run_pipeline() -> None:
        captured["video_path"] = os.environ.get("VIDEO_PATH")

    monkeypatch.setattr(cli, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(cli, "run_pipeline_notebook", fake_run_pipeline)
    monkeypatch.delenv("VIDEO_PATH", raising=False)

    video_file = tmp_path / "clip.mp4"
    cli.main(["--log-level", "DEBUG", "--video", str(video_file)])

    assert captured["log_level"] == "DEBUG"
    assert captured["video_path"] == str(video_file)
