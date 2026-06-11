# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
#
# qdot→qpos block for FREE joints adapted from holosoma_retargeting
# (Apache-2.0). See NOTICE.
"""MuJoCo translational Jacobians ∂x/∂qpos for interaction-mesh SQP."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_T_qdot_to_qpos(model, data) -> NDArray[np.float64]:
    """Return ``T`` (nv, nq) with ``v = T @ qdot`` (MuJoCo velocity layout).

    Supports the first joint as ``mjJNT_FREE`` and subsequent ``HINGE`` /
    ``SLIDE`` joints (typical floating-base humanoids). Multiple FREE joints
    (e.g. manipulated object) fill additional 6×7 blocks in order of appearance.
    """
    import mujoco

    nq, nv = model.nq, model.nv
    T = np.zeros((nv, nq), dtype=np.float64)

    def e_world(qw: float, qx: float, qy: float, qz: float) -> NDArray[np.float64]:
        return np.array(
            [
                [-qx, qw, qz, -qy],
                [-qy, -qz, qw, qx],
                [-qz, qy, -qx, qw],
            ],
            dtype=np.float64,
        )

    for j in range(model.njnt):
        jt = model.jnt_type[j]
        if jt != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[j])
        dadr = int(model.jnt_dofadr[j])
        qw, qx, qy, qz = (float(x) for x in data.qpos[qadr + 3 : qadr + 7])
        E = 2.0 * e_world(qw, qx, qy, qz)
        T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)
        T[dadr + 3 : dadr + 6, qadr + 3 : qadr + 7] = E

    for j in range(model.njnt):
        jt = model.jnt_type[j]
        if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qa = int(model.jnt_qposadr[j])
            da = int(model.jnt_dofadr[j])
            T[da, qa] = 1.0
        elif jt == mujoco.mjtJoint.mjJNT_BALL:
            raise NotImplementedError("BALL joints are not supported in T mapping yet")

    return T


def jacobian_translation_wrt_qpos(
    model,
    data,
    *,
    body_id: int,
    point_body: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Translational Jacobian (3, nq): ``dp_W/dt = J @ qdot`` then ``J_q = J @ T``."""
    import mujoco

    point_body = np.asarray(point_body, dtype=np.float64).reshape(3)
    mujoco.mj_forward(model, data)
    R = data.xmat[body_id].reshape(3, 3)
    p_w = (data.xpos[body_id] + R @ point_body).reshape(3, 1)
    Jp = np.zeros((3, model.nv), dtype=np.float64, order="C")
    Jr = np.zeros((3, model.nv), dtype=np.float64, order="C")
    mujoco.mj_jac(model, data, Jp, Jr, p_w, int(body_id))
    T = build_T_qdot_to_qpos(model, data)
    return (Jp @ T).astype(np.float64, copy=False)


def body_id_or_raise(model, name: str) -> int:
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name.removesuffix("_link"))
    if bid < 0:
        raise ValueError(f"MuJoCo body not found: {name!r}")
    return int(bid)


def pack_joint_q_csv(
    model,
    robot_dof_names: tuple[str, ...],
    qpos: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Map MuJoCo ``qpos`` to CSV ``(7 + N,)`` layout: xyz + quat xyzw + actuated.

    Actuated coordinates follow MuJoCo joint declaration order (HINGE/SLIDE
    only), which matches URDF→MJCF compilation order for standard presets.
    """
    import mujoco

    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if qpos.shape[0] != model.nq:
        raise ValueError(f"qpos length {qpos.shape[0]} != model.nq {model.nq}")
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("expected FREE joint at index 0 for root packing")
    qw, qx, qy, qz = (float(x) for x in qpos[3:7])
    root7 = np.array(
        [qpos[0], qpos[1], qpos[2], qx, qy, qz, qw],
        dtype=np.float32,
    )
    hinge_vals: list[float] = []
    for j in range(model.njnt):
        jt = model.jnt_type[j]
        if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            adr = int(model.jnt_qposadr[j])
            hinge_vals.append(float(qpos[adr]))
    n_act = len(robot_dof_names)
    if len(hinge_vals) != n_act:
        raise ValueError(
            f"MuJoCo hinge/slide count {len(hinge_vals)} != len(robot_dof_names)={n_act}"
        )
    return np.concatenate([root7, np.asarray(hinge_vals, dtype=np.float32)], axis=0)


__all__ = [
    "body_id_or_raise",
    "build_T_qdot_to_qpos",
    "jacobian_translation_wrt_qpos",
    "pack_joint_q_csv",
]
