"""Vector helpers (numpy, batched)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-8


def zeros(*shape: int, dtype: np.dtype | type = np.float32) -> NDArray:
    """Return a zero vector / batch of vectors."""
    return np.zeros(shape, dtype=dtype)


def norm(v: NDArray, axis: int = -1, keepdims: bool = False) -> NDArray:
    """L2 norm along ``axis``."""
    return np.linalg.norm(v, axis=axis, keepdims=keepdims)


def normalize(v: NDArray, axis: int = -1, eps: float = EPS) -> NDArray:
    """Safely normalise a vector along ``axis``; zero-length vectors stay zero."""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return v / n


def distance(a: NDArray, b: NDArray, axis: int = -1) -> NDArray:
    """L2 distance between two (batched) vectors."""
    return np.linalg.norm(a - b, axis=axis)


def dot(a: NDArray, b: NDArray, axis: int = -1, keepdims: bool = False) -> NDArray:
    """Dot product along ``axis``."""
    return np.sum(a * b, axis=axis, keepdims=keepdims)


def cross(a: NDArray, b: NDArray) -> NDArray:
    """Cross product of two 3-vectors (broadcast-friendly)."""
    return np.cross(a, b)
