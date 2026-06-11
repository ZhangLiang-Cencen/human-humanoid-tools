"""Compatibility shims for legacy SMPL/SMPL-H pickle weights.

SMPL and SMPL-H weight files published by MPI are pickled with ``chumpy`` arrays and depend on
a handful of numpy / inspect APIs that have been removed in modern Python and NumPy releases.
This module restores the missing hooks on demand so that :mod:`smplx` can unpickle those files
without users needing a custom numpy/python environment.

The shims are intentionally minimal and side-effect free when the runtime already exposes the
required symbols (e.g. running on a system that still ships ``numpy.bool``).
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np

# (name, fallback value) pairs for ``numpy`` aliases that ``chumpy`` expects to exist.
_NUMPY_ALIASES: tuple[tuple[str, object], ...] = (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode", str),
)


def patch_chumpy_compat() -> None:
    """Patch :mod:`inspect` and :mod:`numpy` so that ``chumpy`` can be imported on Python 3.12+.

    Safe to call multiple times.
    """
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    with warnings.catch_warnings():
        # numpy emits FutureWarnings when probing deprecated scalar aliases; silence them to
        # keep terminal output clean -- the aliases are intentionally re-introduced here.
        warnings.simplefilter("ignore", FutureWarning)
        for name, fallback in _NUMPY_ALIASES:
            if not hasattr(np, name):
                setattr(np, name, fallback)


__all__ = ["patch_chumpy_compat"]
