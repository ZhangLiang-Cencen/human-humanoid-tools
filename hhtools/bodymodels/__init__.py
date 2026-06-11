"""SMPL / SMPL-H / SMPL-X utilities: parameter dataclass, joint layouts and forward engine.

Everything except :class:`SmplMotionParams` and :mod:`hhtools.bodymodels.layout` is
intentionally lazy-loaded so that ``import hhtools.core`` does not require ``smplx`` or
``torch`` to be installed.
"""

from __future__ import annotations

from hhtools.bodymodels.layout import (
    SMPL_LAYOUT,
    SMPLH_LAYOUT,
    SMPLX_LAYOUT,
    BodyModelLayout,
    layout_for,
)
from hhtools.bodymodels.params import SmplMotionParams
from hhtools.bodymodels.paths import (
    body_model_search_paths,
    check_body_models,
    default_body_model_root,
    find_body_model,
)

__all__ = [
    "BodyModelLayout",
    "SMPL_LAYOUT",
    "SMPLH_LAYOUT",
    "SMPLX_LAYOUT",
    "SmplMotionParams",
    "body_model_search_paths",
    "check_body_models",
    "default_body_model_root",
    "find_body_model",
    "layout_for",
]
