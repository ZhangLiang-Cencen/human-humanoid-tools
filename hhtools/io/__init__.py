"""Import / export adapters for human motion formats.

All loaders return :class:`hhtools.core.motion.Motion`. All exporters write the unified NPZ schema
(or a format-specific output driven by CLI sub-commands).
"""

from __future__ import annotations

from hhtools.io import bvh, npz
from hhtools.io.base import MotionLoader, MotionSaver, load_motion, save_motion

__all__ = ["MotionLoader", "MotionSaver", "bvh", "load_motion", "npz", "save_motion"]
