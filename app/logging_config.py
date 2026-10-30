"""Basic logging setup for the API."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the process."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return

    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
