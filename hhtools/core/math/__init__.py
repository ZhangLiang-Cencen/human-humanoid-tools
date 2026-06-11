"""Vectorised math primitives used throughout hhtools.

Design rules:

- Everything operates on `numpy` arrays (no optional torch dependency in the core).
- Functions are pure, accept arbitrary leading batch dimensions, and return new arrays.
- Quaternions are stored as ``xyzw`` (matching the public NPZ schema and soma-retargeter CSV).
- Rotation matrices are right-multiplied against column vectors: ``R @ v``.
"""

from __future__ import annotations

from hhtools.core.math import quaternion, rotation, transform, vector

__all__ = ["quaternion", "rotation", "transform", "vector"]
