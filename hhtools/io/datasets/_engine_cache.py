"""Small helper that caches :class:`SmplxEngine` instances across dataset adapter calls.

Constructing the engine takes O(1-3s) because it needs to load weights, so we re-use it
between adjacent calls for the same ``(family, gender, num_betas)`` combination.  This module
is intentionally lazy so that ``import hhtools.io.datasets`` does not pull in ``smplx`` or
``torch``.
"""

from __future__ import annotations

from functools import lru_cache

from hhtools.bodymodels.params import SmplMotionParams


@lru_cache(maxsize=8)
def _get_cached_engine(family: str, gender: str, num_betas: int):  # noqa: ANN202
    from hhtools.bodymodels.engine import SmplxEngine

    return SmplxEngine(family, gender=gender, num_betas=num_betas)


def engine_for_params(params: SmplMotionParams):  # noqa: ANN201
    """Return a cached :class:`SmplxEngine` matching the SMPL family declared by *params*."""
    betas_dim = int(params.betas.reshape(-1).shape[0])
    num_betas = min(betas_dim, 10)  # smplx defaults
    return _get_cached_engine(params.surface_model, params.gender, num_betas)


def clear_engine_cache() -> None:
    _get_cached_engine.cache_clear()


__all__ = ["clear_engine_cache", "engine_for_params"]
