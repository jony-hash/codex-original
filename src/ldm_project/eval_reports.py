"""Compatibility facade; canonical v1 reports live in :mod:`ldm_project.v1`."""

from .v1.eval_reports import *  # noqa: F403
from .v1 import eval_reports as _implementation


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess CLI tests
    _implementation._cli()
