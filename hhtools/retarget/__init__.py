"""Human-to-humanoid retargeting pipelines.

Two sibling subpackages:

* :mod:`hhtools.retarget.newton_basic` — stock Newton+Warp IK pipeline for
  skeleton-only human motion (``mimic/``).  Stage 1 (pure NumPy, CPU)
  ships the scaler / feet-stabilizer / joint-limit-clamper primitives;
  stage 2 will add the IK solver on top.
* :mod:`hhtools.retarget.interaction_mesh` — Laplacian interaction mesh +
  MPC/SQP backend for ``intermimic/`` and ``meshmimic/`` (same calibration
  and scaler as ``newton_basic``; MuJoCo + QP pieces land incrementally).

Subpackages are **imported lazily** so ``import hhtools.retarget`` does not
require MuJoCo or other stage-2 dependencies until you actually touch
``interaction_mesh`` / ``newton_basic``.
"""

from __future__ import annotations

import importlib

__all__ = ["interaction_mesh", "newton_basic"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
