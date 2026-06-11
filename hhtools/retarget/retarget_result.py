# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared output data class for all retarget backends (Newton, interaction mesh)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["RetargetedMotion"]


@dataclass
class RetargetedMotion:
    """Output bundle for a single retargeted clip.

    Layout of :attr:`joint_q`: ``(F, root_coord_count + actuated_dof_count)``.
    The first 7 columns are ``(tx, ty, tz, qx, qy, qz, qw)`` — Newton's
    floating-base root joint — matching the hhtools CSV schema.
    """

    name: str
    joint_q: NDArray
    sample_rate: float
    dof_names: tuple[str, ...]
    root_coord_count: int = 7
    meta: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.joint_q.shape[0])

    @property
    def root_trajectory(self) -> NDArray:
        """``(F, 7)`` slice: floating-base root (xyz + xyzw quat)."""
        return self.joint_q[:, : self.root_coord_count]

    @property
    def dof_trajectory(self) -> NDArray:
        """``(F, D)`` slice: actuated DOFs (post root)."""
        return self.joint_q[:, self.root_coord_count :]
