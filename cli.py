import argparse
import os

from config import configure_logging
from tracker import run_pipeline_notebook


def main() -> None:
    """Command line entry point for the tracking pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    run_pipeline_notebook()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
