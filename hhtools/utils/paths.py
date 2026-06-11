"""Platform-aware cache and config paths for hhtools."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir

HHTOOLS_CACHE_ENV = "HHTOOLS_CACHE_DIR"
HHTOOLS_ROBOT_DIR_ENV = "HHTOOLS_ROBOT_DIR"


def hhtools_cache_dir() -> Path:
    """Return (and create) the hhtools per-user cache directory.

    Honours the ``HHTOOLS_CACHE_DIR`` environment variable when set; otherwise falls back to the
    platform-standard user cache directory (``~/.cache/hhtools`` on Linux, ``%LOCALAPPDATA%``
    on Windows, ``~/Library/Caches/hhtools`` on macOS).
    """
    override = os.environ.get(HHTOOLS_CACHE_ENV)
    p = Path(override) if override else Path(user_cache_dir("hhtools", "hhtools"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_robot_dir() -> Path:
    """Return (and create) the per-user robot preset library.

    Web UI uploads and ``hhtools robot add``-style user installs land here so
    they survive server restarts.  Honours ``HHTOOLS_ROBOT_DIR`` when set;
    otherwise ``$XDG_CONFIG_HOME/hhtools/robots`` (typically
    ``~/.config/hhtools/robots`` on Linux).
    """
    override = os.environ.get(HHTOOLS_ROBOT_DIR_ENV)
    if override:
        p = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        user_cfg = Path(xdg).expanduser() if xdg else Path.home() / ".config"
        p = user_cfg / "hhtools" / "robots"
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    "HHTOOLS_CACHE_ENV",
    "HHTOOLS_ROBOT_DIR_ENV",
    "hhtools_cache_dir",
    "user_robot_dir",
]
