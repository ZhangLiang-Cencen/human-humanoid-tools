# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Derive contact-aware MPC vertices from robot collision geometry."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from hhtools.retarget.interaction_mesh.mpc_loop import RobotMpcPoint


def _body_id(model, name: str) -> int:
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name.removesuffix("_link"))
    return int(bid)


def _geom_candidate_offsets(model, gid: int) -> list[NDArray[np.float64]]:
    """Representative collision points for one MuJoCo geom in its body frame."""
    import mujoco

    gtype = int(model.geom_type[gid])
    c = np.asarray(model.geom_pos[gid], dtype=np.float64).reshape(3)
    R = np.asarray(model.geom_quat[gid], dtype=np.float64).reshape(4)
    gs = np.asarray(model.geom_size[gid], dtype=np.float64)

    # MuJoCo stores geom_quat as wxyz.
    qw, qx, qy, qz = (float(x) for x in R)
    rot = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )

    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(gs[0])
        return [
            c,
            c + np.array([r, 0, 0]), c - np.array([r, 0, 0]),
            c + np.array([0, r, 0]), c - np.array([0, r, 0]),
            c - np.array([0, 0, r]),
        ]

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = float(gs[0]), float(gs[1]), float(gs[2])
        pts = [
            np.array([dx * sx, dy * sy, dz * sz], dtype=np.float64)
            for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)
        ]
        return [c + rot @ p for p in pts]

    if gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, hl = float(gs[0]), float(gs[1])
        pts = [
            np.array([0, 0, hl]), np.array([0, 0, -hl]),
            np.array([r, 0, hl]), np.array([-r, 0, hl]),
            np.array([0, r, hl]), np.array([0, -r, hl]),
            np.array([r, 0, -hl]), np.array([-r, 0, -hl]),
            np.array([0, r, -hl]), np.array([0, -r, -hl]),
        ]
        return [c + rot @ p for p in pts]

    rb = float(model.geom_rbound[gid])
    return [
        c,
        c + np.array([rb, 0, 0]), c - np.array([rb, 0, 0]),
        c + np.array([0, rb, 0]), c - np.array([0, rb, 0]),
        c - np.array([0, 0, rb]),
    ]


def _body_collision_offsets(model, body_name: str) -> NDArray[np.float64]:
    bid = _body_id(model, body_name)
    if bid < 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts: list[NDArray[np.float64]] = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == bid:
            pts.extend(_geom_candidate_offsets(model, gid))
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.vstack(pts).astype(np.float64, copy=False)
    # Remove near-duplicates from repeated primitive samples.
    rounded = np.round(arr, 4)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return arr[np.sort(idx)]


def _pick_extreme(points: NDArray[np.float64], axis: int, sign: float) -> NDArray[np.float64]:
    vals = points[:, axis] * float(sign)
    return points[int(np.argmax(vals))].copy()


def _append_unique(
    out: list[RobotMpcPoint],
    body_name: str,
    source_index: int,
    semantic: str,
    candidates: Sequence[tuple[str, NDArray[np.float64]]],
) -> None:
    seen = {
        (pt.body_name, tuple(np.round(np.asarray(pt.local_offset), 4)), pt.source_index)
        for pt in out
    }
    for tag, off in candidates:
        key = (body_name, tuple(np.round(np.asarray(off, dtype=np.float64), 4)), source_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            RobotMpcPoint(
                body_name=body_name,
                local_offset=np.asarray(off, dtype=np.float64).reshape(3),
                semantic=f"{semantic}:{tag}",
                source_index=int(source_index),
            )
        )


def build_contact_mpc_points(
    model,
    robot_links: Sequence[str],
    source_indices: Sequence[int],
    canonical_names: Sequence[str],
    *,
    hand_effector_weight: float = 1.0,
) -> list[RobotMpcPoint]:
    """Build coarse IK points plus collision-derived contact vertices.

    This is robot-agnostic: contact offsets come from the compiled MuJoCo
    collision geometry and are attached to the same mapped body/source joint.

    ``hand_effector_weight`` scales the position-cost weight of the wrist /
    ``hand_tip`` points.  For hand-less robots (RoboParty RP1 etc.) the wrist
    link's last collision tip stands in for the hand; raising its weight on
    grasping clips (OMOMO chair/box) makes the IK prioritise actually reaching
    the contact instead of letting the feet / pelvis average the arm short.
    """
    out: list[RobotMpcPoint] = []
    for body_name, src_idx, canon in zip(
        robot_links, source_indices, canonical_names, strict=True,
    ):
        out.append(
            RobotMpcPoint(
                body_name=str(body_name),
                local_offset=np.zeros(3, dtype=np.float64),
                semantic=str(canon),
                source_index=int(src_idx),
            )
        )

        lower = str(canon).lower()
        if not any(k in lower for k in ("ankle", "wrist")):
            continue

        pts = _body_collision_offsets(model, str(body_name))
        if pts.shape[0] == 0:
            continue

        candidates: list[tuple[str, NDArray[np.float64]]] = []
        if "ankle" in lower:
            z_low = float(np.min(pts[:, 2]))
            sole = pts[pts[:, 2] <= z_low + 0.03]
            use = sole if sole.shape[0] >= 2 else pts
            candidates.extend(
                [
                    ("toe", _pick_extreme(use, 0, 1.0)),
                    ("heel", _pick_extreme(use, 0, -1.0)),
                ]
            )
        elif "wrist" in lower:
            candidates.extend(
                [
                    ("hand_tip", pts[int(np.argmax(np.linalg.norm(pts, axis=1)))].copy()),
                ]
            )

        _append_unique(out, str(body_name), int(src_idx), lower, candidates)

    if hand_effector_weight != 1.0:
        import dataclasses

        out = [
            dataclasses.replace(pt, weight=float(hand_effector_weight))
            if ("wrist" in str(pt.semantic).lower() or "hand_tip" in str(pt.semantic).lower())
            else pt
            for pt in out
        ]

    return out


__all__ = ["build_contact_mpc_points"]
