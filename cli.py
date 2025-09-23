import argparse
import os
from typing import Optional, Sequence

from config import configure_logging
from tracker import run_pipeline_notebook


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command line entry point for the tracking pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--video",
        help="Video path to process; overrides the VIDEO_PATH environment variable",
    )
    args = parser.parse_args(argv)
    if args.video is not None:
        os.environ["VIDEO_PATH"] = args.video
    configure_logging(args.log_level)
    run_pipeline_notebook()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
