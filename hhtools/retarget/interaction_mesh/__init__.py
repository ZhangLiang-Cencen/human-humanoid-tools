# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Interaction-mesh retargeting (Laplacian + MPC / SQP).

Sibling to :mod:`hhtools.retarget.newton_basic` for ``intermimic`` / ``meshmimic``
clips. Reuses the same calibration YAML and :class:`~hhtools.retarget.newton_basic.scaler.HumanToRobotScaler`.
"""

from __future__ import annotations

from hhtools.retarget.interaction_mesh.config import InteractionMeshPipelineConfig
from hhtools.retarget.interaction_mesh.laplacian_geometry import (
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
)
from hhtools.retarget.interaction_mesh.motion_bridge import (
    ScaledMotionScene,
    scale_motion_and_objects,
    scatter_scaled_skeleton,
)
from hhtools.retarget.interaction_mesh.mpc_loop import (
    FrameLaplacianTarget,
    iterate_mpc_rti,
    precompute_target_laplacians,
    sqp_step_laplacian,
)
from hhtools.retarget.interaction_mesh.mujoco_jacobians import (
    body_id_or_raise,
    build_T_qdot_to_qpos,
    jacobian_translation_wrt_qpos,
    pack_joint_q_csv,
)
from hhtools.retarget.interaction_mesh.mujoco_scene import MujocoScene, require_mujoco_model
from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline

from . import qp_step

__all__ = [
    "FrameLaplacianTarget",
    "InteractionMeshPipeline",
    "InteractionMeshPipelineConfig",
    "MujocoScene",
    "ScaledMotionScene",
    "body_id_or_raise",
    "build_T_qdot_to_qpos",
    "calculate_laplacian_coordinates",
    "calculate_laplacian_matrix",
    "create_interaction_mesh",
    "get_adjacency_list",
    "iterate_mpc_rti",
    "jacobian_translation_wrt_qpos",
    "pack_joint_q_csv",
    "precompute_target_laplacians",
    "qp_step",
    "require_mujoco_model",
    "scale_motion_and_objects",
    "scatter_scaled_skeleton",
    "sqp_step_laplacian",
]
