"""Viser + FastAPI web viewer for hhtools."""

from __future__ import annotations

__all__ = ["run_viewer"]


def run_viewer(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Thin re-export that avoids a hard dependency at import time."""
    from hhtools.viewer.app import run_viewer as _run

    return _run(*args, **kwargs)
