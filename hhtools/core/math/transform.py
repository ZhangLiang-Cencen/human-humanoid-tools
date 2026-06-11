"""Rigid transforms: translation + unit quaternion.

A transform is an ``(..., 7)`` array laid out as ``[tx, ty, tz, qx, qy, qz, qw]``. This mirrors the
``soma-retargeter`` CSV convention and fits naturally into per-joint pose stacks. We also provide
helpers for composition and point application.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q


def identity(*leading: int, dtype: np.dtype | type = np.float32) -> NDArray:
    """Batched identity transform with shape ``(*leading, 7)``."""
    t = np.zeros((*leading, 7), dtype=dtype)
    t[..., 6] = 1.0
    return t


def make(translation: NDArray, quat: NDArray) -> NDArray:
    """Bundle translation ``(..., 3)`` and quaternion ``(..., 4)`` into a transform."""
    translation = np.asarray(translation, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    if translation.shape[:-1] != quat.shape[:-1]:
        raise ValueError(
            f"Leading shape mismatch: translation {translation.shape} vs quaternion {quat.shape}"
        )
    return np.concatenate([translation, quat], axis=-1)


def translation(t: NDArray) -> NDArray:
    """Extract translation component."""
    return t[..., 0:3]


def rotation(t: NDArray) -> NDArray:
    """Extract quaternion (rotation) component."""
    return t[..., 3:7]


def compose(a: NDArray, b: NDArray) -> NDArray:
    """Compose two rigid transforms: ``c = a * b`` (apply ``b`` first, then ``a``)."""
    qa = rotation(a)
    qb = rotation(b)
    pa = translation(a)
    pb = translation(b)
    q = Q.multiply(qa, qb)
    p = pa + Q.rotate(qa, pb)
    return np.concatenate([p, q], axis=-1)


def inverse(t: NDArray) -> NDArray:
    """Invert a rigid transform."""
    q = rotation(t)
    p = translation(t)
    q_inv = Q.conjugate(q)
    p_inv = -Q.rotate(q_inv, p)
    return np.concatenate([p_inv, q_inv], axis=-1)


def apply_point(t: NDArray, p: NDArray) -> NDArray:
    """Apply a rigid transform to a 3-point: ``t * p``."""
    q = rotation(t)
    tr = translation(t)
    return Q.rotate(q, p) + tr
