from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def scribe(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
