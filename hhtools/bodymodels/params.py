"""Unified SMPL-family parameter IR.

This dataclass is consumed by :mod:`hhtools.bodymodels.engine` (to forward through a body model
and obtain vertex positions + joint locations) and by :mod:`hhtools.io.datasets` adapters (which
produce instances from heterogeneous source files).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

SurfaceModel = Literal["smpl", "smplh", "smplx"]
Gender = Literal["male", "female", "neutral"]
UpAxis = Literal["X", "Y", "Z"]


@dataclass
class SmplMotionParams:
    """Intermediate representation for an SMPL/SMPL-H/SMPL-X sequence.

    The engine in :mod:`hhtools.bodymodels.engine` accepts instances of this class and produces a
    :class:`hhtools.core.motion.Motion`. Dataset adapters normalise their on-disk schemas into
    instances of this dataclass.
    """

    surface_model: SurfaceModel
    root_orient: NDArray  # (T, 3) axis-angle
    body_pose: NDArray  # (T, 63) for smplh/smplx, (T, 69) for smpl
    betas: NDArray  # (10,) or (16,) or (T, K)
    trans: NDArray  # (T, 3)

    gender: Gender = "neutral"
    framerate: float = 30.0
    hand_pose_left: NDArray | None = None  # (T, 45)
    hand_pose_right: NDArray | None = None  # (T, 45)
    jaw_pose: NDArray | None = None  # (T, 3)
    leye_pose: NDArray | None = None  # (T, 3)
    reye_pose: NDArray | None = None  # (T, 3)
    expression: NDArray | None = None  # (T, K)
    up_axis: UpAxis = "Z"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root_orient = np.asarray(self.root_orient, dtype=np.float32)
        self.body_pose = np.asarray(self.body_pose, dtype=np.float32)
        self.betas = np.asarray(self.betas, dtype=np.float32)
        self.trans = np.asarray(self.trans, dtype=np.float32)
        if self.root_orient.ndim != 2 or self.root_orient.shape[1] != 3:
            raise ValueError(f"root_orient must be (T, 3); got {self.root_orient.shape}")
        if self.trans.shape != self.root_orient.shape:
            raise ValueError(
                f"trans {self.trans.shape} must match root_orient {self.root_orient.shape}"
            )
        if self.framerate <= 0:
            raise ValueError(f"framerate must be positive; got {self.framerate}")

    @property
    def num_frames(self) -> int:
        return int(self.root_orient.shape[0])


__all__ = ["Gender", "SmplMotionParams", "SurfaceModel"]
