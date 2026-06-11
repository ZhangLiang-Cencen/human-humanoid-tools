# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Bridge :class:`~hhtools.core.motion.Motion` + :class:`~hhtools.core.scene.SceneObject`
to scaled world arrays for interaction-mesh retargeting.

Uses the same :class:`~hhtools.retarget.newton_basic.scaler.HumanToRobotScaler` as
``newton_basic`` so humans and rigid props shrink together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.core.motion import Motion
from hhtools.core.scene import SceneObject
from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler


@dataclass(frozen=True)
class ScaledMotionScene:
    """Per-frame human bone positions and object translations after scaler.

    Attributes:
        human_positions: ``(F, num_bones, 3)`` float32, hierarchy order.
        object_positions: list of ``(F, 3)`` arrays (same length as input objects).
        object_uniform_scales: list of floats — ``SceneObject.scale`` multiplied
            by ``human_height / human_height_assumption`` so mesh draw matches
            the shrunken world frame.
    """

    human_positions: NDArray[np.float32]
    object_positions: list[NDArray[np.float32]]
    object_uniform_scales: list[float]
    object_points: list[NDArray[np.float32]] | None = None


def scatter_scaled_skeleton(
    motion: Motion, scaler: HumanToRobotScaler
) -> NDArray[np.float32]:
    """Build ``(F, N, 3)`` scaled positions in ``motion.hierarchy`` bone order."""
    bones = set(motion.hierarchy.bone_names)
    mapped = set(scaler.joint_names)
    if bones != mapped:
        raise KeyError(
            "interaction_mesh needs ScalerConfig.joint_scales keys to match "
            f"motion hierarchy exactly; only_in_motion={bones - mapped} "
            f"only_in_scaler={mapped - bones}"
        )

    eff = scaler.apply(motion)
    F, n_bones = motion.num_frames, motion.hierarchy.num_bones
    out = np.zeros((F, n_bones, 3), dtype=np.float32)
    name_to_i = {n: i for i, n in enumerate(motion.hierarchy.bone_names)}
    for j, name in enumerate(eff.joint_names):
        idx = name_to_i.get(name)
        if idx is None:
            continue
        out[:, idx, :] = eff.transforms[:, j, :3]
    missing = [n for n in motion.hierarchy.bone_names if n not in eff.joint_names]
    if missing:
        raise KeyError(
            "interaction_mesh requires every hierarchy bone to appear in "
            f"ScalerConfig.joint_scales; missing: {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
        )
    return out


def scale_motion_and_objects(
    motion: Motion,
    scaler: HumanToRobotScaler,
    objects: list[SceneObject] | None = None,
) -> ScaledMotionScene:
    """Apply scaler to full skeleton and each scene object's origin trajectory."""
    human = scatter_scaled_skeleton(motion, scaler)
    objs = objects or []
    ratio = float(scaler.human_height / scaler.config.human_height_assumption)
    obj_pos: list[NDArray[np.float32]] = []
    obj_scales: list[float] = []
    for ob in objs:
        p = ob.positions.astype(np.float32, copy=False)
        # One world origin per frame → shape (F, 1, 3) so the scaler does not
        # interpret (F, 3) as K=F static points (see HumanToRobotScaler doc).
        if p.ndim == 2 and p.shape[1] == 3:
            p = p[:, None, :]
        scaled = scaler.scale_world_points_about_root(motion, p)
        obj_pos.append(scaled[:, 0, :].copy())
        obj_scales.append(float(ob.scale * ratio))
    return ScaledMotionScene(
        human_positions=human,
        object_positions=obj_pos,
        object_uniform_scales=obj_scales,
        object_points=None,
    )


__all__ = ["ScaledMotionScene", "scatter_scaled_skeleton", "scale_motion_and_objects"]
