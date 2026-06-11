# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""MPC-style windowing + per-frame interaction mesh / Laplacian precompute.

Full ``iterate_mpc`` + holosoma-style foot / penetration constraints will plug
into :mod:`hhtools.retarget.interaction_mesh.qp_step` once MuJoCo Jacobians are
wired.  This module already centralises **target Laplacian** construction from
scaled human + object samples so the MPC horizon can consume a pre-built list.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.retarget.interaction_mesh.laplacian_geometry import (
    calculate_laplacian_coordinates,
    create_interaction_mesh,
    get_adjacency_list,
)
from hhtools.retarget.interaction_mesh.motion_bridge import ScaledMotionScene
from hhtools.retarget.interaction_mesh.qp_step import OsqpUnreliableError

_log = logging.getLogger(__name__)

# Trust-region shrinkage applied when OSQP fails and the SQP falls
# back to a box-only L-BFGS-B solve.  At the default
# ``step_size = 0.2 rad`` this caps the per-iter |Δq| at 0.05 rad —
# small enough that ``smooth_weight`` can absorb it instead of leaving
# the multi-degree single-frame spikes the previous full-trust
# fallback was producing.  See ``sqp_step_laplacian`` docstring,
# section "Failure semantics".
OSQP_FALLBACK_TRUST_SHRINK = 0.25


def count_named_mujoco_bodies(model) -> int:
    """Count MuJoCo bodies (excluding world) that have a non-empty name."""
    import mujoco

    n = 0
    for bid in range(1, model.nbody):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid):
            n += 1
    return n


def _subsample_human_xyz_rows(h: NDArray[np.floating], nh: int) -> NDArray[np.float32]:
    """Pick ``nh`` rows from ``(J, 3)`` human joints (unique indices, roughly uniform)."""

    h = np.asarray(h, dtype=np.float32).reshape(-1, 3)
    j = int(h.shape[0])
    if j <= nh:
        return h.astype(np.float32, copy=False)
    idx = np.unique(np.linspace(0, j - 1, nh).round().astype(np.int64))
    if int(idx.size) < nh:
        idx = np.arange(min(nh, j), dtype=np.int64)
    return h[idx].astype(np.float32, copy=False)


def sample_axis_aligned_box(n: int, extents_xyz: NDArray[np.floating]) -> NDArray[np.float32]:
    """Approximately ``n`` points inside a centred box ``extents`` (full side lengths)."""
    ex = np.asarray(extents_xyz, dtype=np.float64).reshape(3)
    rng = np.random.default_rng(0)
    m = max(8, int(n))
    pts = rng.uniform(-0.5, 0.5, size=(m, 3)) * ex[None, :]
    return pts.astype(np.float32, copy=False)


@dataclass
class FrameLaplacianTarget:
    """One frame's Delaunay topology + target Laplacian coordinates."""

    adj_list: list[list[int]]
    target_laplacian: NDArray[np.float32]
    source_vertices: NDArray[np.float32]
    n_human_vertices: int
    # Source pelvis quaternion for this frame, stored as (qx, qy, qz, qw)
    # to match the rest of the codebase's xyzw convention.  Optional —
    # only populated when the precompute pipeline has access to the
    # source quaternions (the SMPL/SMPL-X path always does).  Read by
    # the SQP frame-0 base-orientation warm-start; leaving it ``None``
    # falls back to keeping whatever quaternion is in the freejoint at
    # solver entry.
    source_root_quat_xyzw: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class RobotMpcPoint:
    """One robot point used as an interaction-mesh vertex.

    ``body_name`` identifies the MuJoCo body, while ``local_offset`` is a
    body-frame point in metres.  The original coarse skeleton uses offset zero;
    contact-aware vertices use offsets derived from collision geometry.
    """

    body_name: str
    local_offset: NDArray[np.float64]
    semantic: str = ""
    source_index: int = -1
    weight: float = 1.0


def build_demo_vertices_frame(
    human_xyz: NDArray[np.floating],
    object_xyz: NDArray[np.floating] | None,
) -> NDArray[np.float32]:
    """Concatenate human joint rows (J,3) with optional object samples (No,3)."""
    h = np.asarray(human_xyz, dtype=np.float32).reshape(-1, 3)
    if object_xyz is None or object_xyz.size == 0:
        v = h
    else:
        o = np.asarray(object_xyz, dtype=np.float32).reshape(-1, 3)
        v = np.vstack([h, o])
    return v


def precompute_target_laplacians(
    scaled: ScaledMotionScene,
    *,
    object_extents: NDArray[np.floating] | list[NDArray[np.floating]] | None = None,
    object_samples: int = 24,
    max_human_vertices: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[FrameLaplacianTarget]:
    """Per-frame target Laplacian δ* from scaled human + optional object box samples.

    Args:
        scaled: Output of :func:`scale_motion_and_objects`.
        object_extents: ``(3,)`` cuboid **full side lengths** (metres). When
            ``scaled.object_positions`` is non-empty and this is ``None``, a
            ``0.6³`` placeholder is used (override with real ``SceneObject.extents``).
        object_samples: Number of random interior box samples per frame.
    max_human_vertices: When set, subsample human joint rows to this count so
        the Laplacian vertex count does not exceed the number of **named**
        MuJoCo bodies on the target robot (see :func:`iterate_mpc_rti`).
        progress_callback: Optional ``cb(frame_done, frame_total)`` for UI
        progress (throttled — at most ~40 calls per sequence).
    """
    F, _, _ = scaled.human_positions.shape
    out: list[FrameLaplacianTarget] = []

    obj_extents: list[NDArray[np.float64]] = []
    if scaled.object_positions:
        if object_extents is None:
            obj_extents = [
                np.array([0.6, 0.6, 0.6], dtype=np.float64)
                for _ in scaled.object_positions
            ]
        elif isinstance(object_extents, list):
            obj_extents = [
                np.asarray(ext, dtype=np.float64).reshape(3)
                for ext in object_extents
            ]
        else:
            ext_arr = np.asarray(object_extents, dtype=np.float64).reshape(3)
            obj_extents = [ext_arr for _ in scaled.object_positions]

    notify_stride = max(1, F // 40)

    # ----------------------------------------------------------------
    # Build vertex sets for every frame first, then tetrahedralise
    # **once** over an aggregate point cloud and reuse the resulting
    # adjacency list for every per-frame target Laplacian.
    #
    # Why a shared adjacency: ``create_interaction_mesh`` runs a
    # Delaunay tetrahedralisation whose output topology depends on
    # the precise vertex coordinates.  When the actor walks, the
    # Delaunay neighbourhood of a given joint (e.g. a hand near a
    # terrain bump in frame 100, then near a different bump in frame
    # 105) flips between frames — a single neighbour swap changes
    # the per-frame ``target_laplacian`` discontinuously, and the
    # SQP downstream reflects that as a step in ``qpos``.  In
    # quantitative terms this is observed as multi-degree per-frame
    # ``|Δq|`` spikes that no amount of ``smooth_weight`` can absorb
    # because the residual being smoothed is itself discontinuous.
    #
    # Holosoma's reference design tetrahedralises once over a
    # representative frame and locks the topology for the whole
    # clip; we go one step further and union vertices from every
    # frame so the adjacency captures every "the actor stood here at
    # some point" relationship.  The per-frame target Laplacian
    # then varies smoothly because the **same** adjacency operator
    # is applied to every frame's vertices.
    # ----------------------------------------------------------------
    per_frame_verts: list[NDArray[np.float32]] = []
    nh_per_frame: list[int] = []
    for f in range(F):
        h = scaled.human_positions[f]
        if max_human_vertices is not None and int(h.shape[0]) > int(max_human_vertices):
            h = _subsample_human_xyz_rows(h, int(max_human_vertices))
        obj_samples: list[NDArray[np.float32]] = []
        if scaled.object_points is not None:
            for pts_traj in scaled.object_points:
                if f < int(pts_traj.shape[0]) and int(pts_traj.shape[1]) > 0:
                    obj_samples.append(pts_traj[f].astype(np.float32, copy=False))
        else:
            for i, obj_traj in enumerate(scaled.object_positions):
                if i >= len(obj_extents):
                    continue
                pts = sample_axis_aligned_box(object_samples, obj_extents[i])
                obj_samples.append((pts + obj_traj[f]).astype(np.float32, copy=False))
        o_world = np.vstack(obj_samples) if obj_samples else None
        verts = build_demo_vertices_frame(h, o_world)
        if verts.shape[0] < 4:
            rng = np.random.default_rng(1000 + f)
            extra = rng.normal(scale=1e-3, size=(4 - verts.shape[0], 3)).astype(np.float32)
            verts = np.vstack([verts, extra])
        per_frame_verts.append(verts)
        nh_per_frame.append(int(h.shape[0]))

    # Use a representative frame for the shared Delaunay topology.
    # The vertex layout (role order: ``nh`` human joints followed
    # by object/terrain samples) is identical across frames, so the
    # adjacency indices computed from any one frame remain valid
    # for every other frame.  We pick the **middle** frame: the
    # actor is most likely to be in a generic configuration there,
    # which yields a tetrahedralisation whose neighbour relations
    # describe the clip as a whole better than a possibly atypical
    # T-pose at frame 0.
    V = int(per_frame_verts[0].shape[0])
    pivot_idx = F // 2
    pivot_pts = per_frame_verts[pivot_idx].astype(np.float64, copy=True)
    # Sub-millimetre isotropic perturbation eliminates any
    # accidentally-coplanar groups (terrain grid + skeletal
    # symmetries) that would otherwise produce a degenerate hull.
    rng = np.random.default_rng(0)
    pivot_pts = pivot_pts + rng.normal(scale=1e-5, size=pivot_pts.shape)
    _, tet = create_interaction_mesh(pivot_pts)
    adj = get_adjacency_list(tet, V)

    for f in range(F):
        verts = per_frame_verts[f]
        target = calculate_laplacian_coordinates(verts, adj, uniform_weight=True)
        out.append(
            FrameLaplacianTarget(
                adj_list=adj,
                target_laplacian=target,
                source_vertices=verts,
                n_human_vertices=nh_per_frame[f],
            )
        )
        if progress_callback is not None and (f % notify_stride == 0 or f == F - 1):
            try:
                progress_callback(f + 1, F)
            except Exception:
                pass
    return out


def _mj_body_names_prefix(model, nh: int) -> list[str]:
    """First ``nh`` named bodies (skip world), MuJoCo body id order."""
    import mujoco

    names: list[str] = []
    for bid in range(1, model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not nm:
            continue
        names.append(nm)
        if len(names) >= nh:
            break
    if len(names) < nh:
        raise ValueError(f"need {nh} named bodies for Laplacian, found {len(names)}")
    return names


def _stack_vertex_jacobians(
    model,
    data,
    robot_points: list[RobotMpcPoint],
    nq: int,
) -> NDArray[np.float64]:
    """``(3 * len(robot_points), nq)`` translational Jacobians at each point.

    ``mj_forward`` + ``build_T_qdot_to_qpos`` only depend on the current
    ``qpos`` (kinematics + the FREE-joint quaternion block of ``T``), so
    they need to run **once per call**, not once per point.  The previous
    implementation ran them ``len(robot_points)`` times via
    :func:`jacobian_translation_wrt_qpos`, which dominated profile in
    high-DOF SQP loops.  We now do one ``mj_forward`` and one ``T``
    construction up front, then loop over points calling ``mj_jac``
    only.  Mirrors holosoma's ``_calc_lap_foot_jacobians_batch``.
    """
    import mujoco

    from hhtools.retarget.interaction_mesh.mujoco_jacobians import (
        body_id_or_raise,
        build_T_qdot_to_qpos,
    )

    nv = model.nv
    mujoco.mj_forward(model, data)
    T = build_T_qdot_to_qpos(model, data)

    Jp = np.zeros((3, nv), dtype=np.float64, order="C")
    Jr = np.zeros((3, nv), dtype=np.float64, order="C")

    rows: list[NDArray[np.float64]] = []
    for pt in robot_points:
        bid = body_id_or_raise(model, pt.body_name)
        off = np.asarray(pt.local_offset, dtype=np.float64).reshape(3)
        R = data.xmat[bid].reshape(3, 3)
        p_w = (data.xpos[bid].astype(np.float64) + R @ off).reshape(3)
        Jp.fill(0.0)
        Jr.fill(0.0)
        mujoco.mj_jac(model, data, Jp, Jr, p_w, int(bid))
        rows.append(Jp @ T)
    return np.vstack(rows).astype(np.float64, copy=False)


def _robot_points_from_body_names(body_names: list[str]) -> list[RobotMpcPoint]:
    return [
        RobotMpcPoint(
            body_name=nm,
            local_offset=np.zeros(3, dtype=np.float64),
            semantic=nm,
        )
        for nm in body_names
    ]


def _current_robot_point_positions(
    model,
    data,
    robot_points: list[RobotMpcPoint],
) -> NDArray[np.float64]:
    from hhtools.retarget.interaction_mesh.mujoco_jacobians import body_id_or_raise

    out: list[NDArray[np.float64]] = []
    for pt in robot_points:
        bid = body_id_or_raise(model, pt.body_name)
        R = data.xmat[bid].reshape(3, 3).astype(np.float64)
        off = np.asarray(pt.local_offset, dtype=np.float64).reshape(3)
        out.append(data.xpos[bid].astype(np.float64) + R @ off)
    return np.vstack(out).astype(np.float64, copy=False)


def _normalize_free_joint_quat(model, qpos: NDArray[np.floating]) -> None:
    import mujoco

    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        return
    qadr = int(model.jnt_qposadr[0])
    qq = q[qadr + 3 : qadr + 7]
    n = float(np.linalg.norm(qq))
    if n > 1e-12:
        qq[:] = qq / n


def _build_joint_limits(model) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract per-qpos joint limits from the MuJoCo model.

    Returns ``(q_lb, q_ub)`` each of shape ``(nq,)``.  Free-joint DOFs and
    unlimited hinges get large bounds (±1e6).
    """
    import mujoco

    nq = model.nq
    q_lb = np.full(nq, -1e6, dtype=np.float64)
    q_ub = np.full(nq, 1e6, dtype=np.float64)
    for j in range(model.njnt):
        jtype = int(model.jnt_type[j])
        qadr = int(model.jnt_qposadr[j])
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            continue
        has_limit = bool(model.jnt_limited[j])
        if has_limit:
            q_lb[qadr] = float(model.jnt_range[j, 0])
            q_ub[qadr] = float(model.jnt_range[j, 1])
    return q_lb, q_ub


def _compute_hard_nonpenetration_rows(
    collision_model,
    collision_data,
    qpos: NDArray[np.float64],
    *,
    threshold: float = 0.05,
    tolerance: float = 0.002,
    fd_epsilon: float = 1e-5,
    max_pairs_per_body: int = 4,
) -> tuple[list[NDArray[np.float64]], list[float]]:
    """Mirror of holosoma's hard non-penetration linearisation.

    Wraps :func:`hhtools.retarget.interaction_mesh.collision.compute_nonpenetration_constraints`
    so that the SQP can call it with a single MuJoCo (model, data)
    pair carrying terrain meshes alongside the robot links.  The
    returned ``J_rows`` / ``rhs`` are ready to feed into OSQP as
    ``A_np · δq ≥ rhs`` rows.

    Each row corresponds to one robot↔scene geom pair whose signed
    distance (from ``mj_geomDistance``) was at most ``threshold`` at
    the linearisation point.  The constraint is **non-violable** —
    whatever the cost gradient suggests, OSQP must keep the body above
    the terrain.  This is the only mechanism strong enough to oppose a
    translation-invariant Laplacian cost driving the floating base
    into the ground.

    ``max_pairs_per_body`` caps inequality rows per
    ``(robot_body, scene_geom)`` bucket.  URDF feet that compile to many
    sub-meshes (RP1 right_ankle_roll_link → 24+ collision primitives,
    G1 ankles similar) generate near-duplicate rows whose witness points
    and normals oscillate frame-to-frame as the foot crosses heightfield
    cell boundaries; left uncapped this chatter (53 rows/frame on
    holosoma parkour_1) is the primary source of OSQP infeasibility on
    contact frames, which then triggers the SQP fallback path and
    surfaces as single-frame pose spikes.  Default ``4`` matches the
    empirical sweep noted in
    :func:`compute_nonpenetration_constraints`'s docstring.
    """
    from hhtools.retarget.interaction_mesh.collision import compute_nonpenetration_constraints

    return compute_nonpenetration_constraints(
        collision_model, collision_data, qpos,
        threshold=float(threshold),
        tolerance=float(tolerance),
        fd_epsilon=float(fd_epsilon),
        max_pairs_per_body=int(max_pairs_per_body),
    )


def sqp_step_laplacian(
    model,
    data,
    qpos: NDArray[np.floating],
    frame: FrameLaplacianTarget,
    body_names: list[str],
    *,
    robot_points: list[RobotMpcPoint] | None = None,
    laplacian_weight: float,
    step_size: float,
    sqp_inner_iters: int = 2,
    q_prev: NDArray[np.floating] | None = None,
    smooth_weight: float = 0.6,
    q_lb: NDArray[np.float64] | None = None,
    q_ub: NDArray[np.float64] | None = None,
    # --- absolute world-position cost (anchors global root motion) ---
    position_weight: float = 0.0,
    # --- home-pose Tikhonov on actuated joints ---
    home_pose_weight: float = 0.0,
    home_qpos: NDArray[np.float64] | None = None,
    actuated_qpos_idx: NDArray[np.int64] | None = None,
    # --- collision (holosoma-style hard non-penetration) ---
    collision_model=None,
    collision_data=None,
    collision_threshold: float = 0.05,
    penetration_tolerance: float = 0.002,
    collision_fd_epsilon: float = 1e-5,
    collision_max_pairs_per_body: int = 4,
    # --- trust region ---
    base_step_size: float | None = None,
) -> NDArray[np.float64]:
    """One SQP frame solve with Laplacian + smoothness cost.

    Holosoma-style **hard non-penetration** is the only collision
    mechanism: when ``collision_model`` is provided, ``mj_geomDistance``
    on every robot↔scene geom pair within ``collision_threshold``
    produces an inequality row ``J · δq ≥ −φ − tol``.  These rows are
    fed straight to OSQP alongside the box trust region — exactly the
    construction in
    ``holosoma_retargeting.src.interaction_mesh_retargeter.solve_mpc_iteration``.

    The hard constraint is the only mechanism that can keep the body
    above the terrain when the cost is translation-invariant: any
    candidate ``δq`` that drops the foot below the terrain is *outside*
    the feasible set, so OSQP cannot return it regardless of how much
    the Laplacian cost would otherwise reward it.

    Failure semantics
    -----------------
    OSQP can occasionally fail (``MAX_ITER_REACHED`` /
    ``PRIMAL_INFEASIBLE`` / …) on contact-rich frames where chattery
    non-penetration rows make the KKT system stiff.  Two extreme
    fallbacks are both wrong:

    1. Silent box-only L-BFGS-B with the *same* trust region drops
       every inequality row and lets a single iter take a 0.2 rad /
       7 cm step — the multi-degree single-frame pose spikes seen on
       holosoma parkour clips.
    2. Returning ``q_prev`` unchanged risks the entire clip locking
       up: if frame 0 fails, every later frame warm-starts from the
       same qpos and can fail for the same reason, producing the
       "OMOMO / holosoma robot doesn't move" failure mode.

    The chosen middle path keeps the SQP making progress while
    bounding the per-iter step:

        OsqpUnreliableError ⇒
            box-only L-BFGS-B with trust region scaled by
            ``OSQP_FALLBACK_TRUST_SHRINK`` (currently 1/4).

    With ``step_size=0.2`` the fallback step is at most ``0.05`` rad
    (and ``base_step_size`` shrinks proportionally).  ``smooth_weight``
    pulls ``δq`` back toward zero so the realised per-frame jump is
    almost always below 0.05 rad — small enough to be invisible to
    downstream training, big enough to keep the clip moving.  The
    fallback is logged at WARNING so it shows up in the retarget log
    instead of being silent.
    """
    import mujoco

    from hhtools.retarget.interaction_mesh.laplacian_geometry import calculate_laplacian_matrix
    from hhtools.retarget.interaction_mesh.qp_step import (
        assemble_laplacian_qp,
        build_kron_laplacian_jacobian,
        solve_qp_box_lbfgsb,
    )

    nh = frame.n_human_vertices
    mpc_points = robot_points or _robot_points_from_body_names(body_names[:nh])
    nq = model.nq
    V = int(frame.source_vertices.shape[0])
    obj_pts = frame.source_vertices[nh:].astype(np.float64, copy=False)
    q_work = np.asarray(qpos, dtype=np.float64).copy()

    has_hard_np = collision_model is not None and collision_data is not None

    for _ in range(sqp_inner_iters):
        data.qpos[:] = q_work
        mujoco.mj_forward(model, data)
        pos_r = _current_robot_point_positions(model, data, mpc_points[:nh])
        verts = np.vstack([pos_r, obj_pts]).astype(np.float64, copy=False)
        if verts.shape[0] != V:
            raise RuntimeError("vertex count mismatch between robot bodies and demo mesh")
        L = calculate_laplacian_matrix(verts, frame.adj_list, uniform_weight=True)
        lap0 = (L @ verts).reshape(-1)
        target_vec = frame.target_laplacian.reshape(-1).astype(np.float64, copy=False)

        J_r = _stack_vertex_jacobians(model, data, mpc_points[:nh], nq)
        J_o = np.zeros((3 * max(0, V - nh), nq), dtype=np.float64)
        J_V = np.vstack([J_r, J_o])
        J_L = build_kron_laplacian_jacobian(L, J_V)
        qp = assemble_laplacian_qp(J_L, lap0, target_vec, laplacian_weight=laplacian_weight)

        if q_prev is not None and smooth_weight > 0:
            sw = float(smooth_weight)
            dq_smooth = np.asarray(q_prev, dtype=np.float64) - q_work
            qp.P[:] += 2.0 * sw * np.eye(nq, dtype=np.float64)
            qp.q_vec[:] += -2.0 * sw * dq_smooth

        # ---- Absolute-position tracking cost ------------------
        # Adds ``½ · pw · Σ_i ‖ pos_robot_i − pos_target_i ‖²`` for
        # the ``nh`` mapped joints.  Linearised at ``q_work`` this is
        # ``½ · pw · ‖J_r δq + (pos_r − target)‖²`` which contributes
        # ``J_rᵀ J_r`` to ``P`` and ``J_rᵀ (pos_r − target)`` to
        # ``q_vec``.  Without this term the Laplacian is purely
        # translation-equivariant: an anatomy-mismatched robot whose
        # leg is longer than the scaled-source pelvis-to-foot can
        # satisfy the Laplacian cost by floating ~Δleg above the
        # source target.  The position cost ties absolute positions
        # to the source so the foot-contact pattern (relative to
        # heightfield) matches the source's.
        if position_weight > 0.0:
            pw = float(position_weight)
            target_pos = frame.source_vertices[:nh].astype(np.float64, copy=False)
            res = (pos_r - target_pos).reshape(-1)
            # Per-point relative weights let a grasping end-effector (the wrist
            # collision tip standing in for a missing hand) be prioritised so it
            # actually reaches the contact, rather than averaging out against the
            # feet / pelvis.  Defaults to 1.0 for every point (uniform = old
            # behaviour).
            w_pts = np.array(
                [float(getattr(p, "weight", 1.0)) for p in mpc_points[:nh]],
                dtype=np.float64,
            )
            if np.allclose(w_pts, 1.0):
                qp.P[:] += 2.0 * pw * (J_r.T @ J_r)
                qp.q_vec[:] += 2.0 * pw * (J_r.T @ res)
            else:
                w3 = np.repeat(w_pts, 3)  # one weight per (x, y, z) residual row
                Jw = J_r * w3[:, None]
                qp.P[:] += 2.0 * pw * (J_r.T @ Jw)
                qp.q_vec[:] += 2.0 * pw * (J_r.T @ (w3 * res))

        # ---- Home-pose Tikhonov on actuated DOFs --------------
        # ``½ · hw · Σ_{j ∈ actuated} (q_j + δq_j − q_home_j)²``.
        # Linearised gradient: ``hw · (q_j − q_home_j + δq_j)`` per
        # actuated DOF; that's a diagonal addition to ``P`` and a
        # linear addition to ``q_vec`` on those DOF rows only.
        # Free-joint quaternion / translation DOFs are deliberately
        # excluded — those are pinned by ``position_weight`` on the
        # pelvis vertex, and applying a Tikhonov to the quaternion
        # would fight per-frame yaw changes.
        if (
            home_pose_weight > 0.0
            and home_qpos is not None
            and actuated_qpos_idx is not None
            and actuated_qpos_idx.size > 0
        ):
            hw = float(home_pose_weight)
            idx = actuated_qpos_idx
            err = (q_work[idx] - home_qpos[idx]).astype(np.float64, copy=False)
            qp.P[idx, idx] += 2.0 * hw
            qp.q_vec[idx] += 2.0 * hw * err

        # --- Box trust region + joint limits ---
        lb = np.full(nq, -float(step_size), dtype=np.float64)
        ub = np.full(nq, float(step_size), dtype=np.float64)
        # Tighter cap on floating-base XYZ DOFs.  Holosoma applies its
        # ``step_size`` uniformly; we keep an extra safety margin on
        # root translation so a single OSQP solve cannot cross a 30 cm
        # step in one iteration.
        if base_step_size is not None and base_step_size > 0:
            try:
                if int(model.jnt_type[0]) == mujoco.mjtJoint.mjJNT_FREE:
                    qadr = int(model.jnt_qposadr[0])
                    bs = float(base_step_size)
                    for j in range(qadr, qadr + 3):
                        lb[j] = max(lb[j], -bs)
                        ub[j] = min(ub[j], bs)
            except Exception:
                pass
        if q_lb is not None and q_ub is not None:
            jl_lb = q_lb - q_work
            jl_ub = q_ub - q_work
            np.maximum(lb, jl_lb, out=lb)
            np.minimum(ub, jl_ub, out=ub)

        # --- Solve QP -----------------------------------------------------
        if has_hard_np:
            J_rows, rhs = _compute_hard_nonpenetration_rows(
                collision_model, collision_data, q_work,
                threshold=float(collision_threshold),
                tolerance=float(penetration_tolerance),
                fd_epsilon=float(collision_fd_epsilon),
                max_pairs_per_body=int(collision_max_pairs_per_body),
            )
            try:
                dq = _solve_qp_with_inequalities(
                    qp.P, qp.q_vec, lb, ub, J_rows, rhs,
                ).astype(np.float64, copy=False)
            except OsqpUnreliableError as exc:
                # Bounded-step box-only fallback — see "Failure
                # semantics" in the docstring.  Trust region is
                # shrunk by OSQP_FALLBACK_TRUST_SHRINK so the solver
                # cannot take more than ~step_size/4 in any single
                # iter even though the inequality rows are dropped;
                # combined with smooth_weight this keeps the realised
                # |Δq| comparable to a healthy frame.
                _log.warning(
                    "SQP frame OSQP fallback (box-only, trust×%.2f): %s",
                    OSQP_FALLBACK_TRUST_SHRINK, exc,
                )
                lb_fb = lb * OSQP_FALLBACK_TRUST_SHRINK
                ub_fb = ub * OSQP_FALLBACK_TRUST_SHRINK
                dq = solve_qp_box_lbfgsb(qp, lb_fb, ub_fb).astype(np.float64, copy=False)
        else:
            dq = solve_qp_box_lbfgsb(qp, lb, ub).astype(np.float64, copy=False)

        q_work = q_work + dq.reshape(-1)
        _normalize_free_joint_quat(model, q_work)
    return q_work


# Quadratic penalty on the per-row non-penetration slack.
#
# Tuned on holosoma parkour_1 (rp1, 80 frames).  This is deliberately a
# *gentle* backstop, not a stiff barrier, for two reasons:
#
#  1. The base + feet are already anchored in absolute world space by the
#     position cost (``position_weight`` = 400), which tracks the scaled
#     source feet whose contact pattern relative to the terrain is correct
#     by construction — so collision only has to stop gross penetration,
#     not reproduce contact.
#  2. Heightfield ``mj_geomDistance`` witness points / normals flip
#     discontinuously as a foot crosses terrain cell boundaries.  A stiff
#     penalty turns that chatter into the whole-robot "flashing" jitter the
#     user reported (sweep: ρ=1e2 → 1.6°/frame² jerk_max; ρ=1e3 → 10.9°;
#     ρ=1e4 → 30.3°, clearly trembling).  ρ=1e2 keeps the trajectory as
#     smooth as the old (collision-dropped) path while still feasible.
#
# Override via ``HHTOOLS_NP_SLACK_PENALTY`` for experiments.
NONPENETRATION_SLACK_PENALTY = float(
    os.environ.get("HHTOOLS_NP_SLACK_PENALTY", "1.0e2")
)


def _solve_qp_with_inequalities(
    P: NDArray[np.float64],
    q_vec: NDArray[np.float64],
    lb: NDArray[np.float64],
    ub: NDArray[np.float64],
    J_rows: list[NDArray[np.float64]],
    rhs: list[float],
    *,
    slack_penalty: float = NONPENETRATION_SLACK_PENALTY,
) -> NDArray[np.float64]:
    """Solve QP with box bounds + **soft** non-penetration constraints via OSQP.

    Combines::

        min  0.5 x'Px + q'x  +  0.5·ρ·Σ sᵢ²
        s.t. lb ≤ x ≤ ub                    (box / trust region + joint limits)
             J_rows[i] · x + sᵢ ≥ rhs[i]    (non-penetration, slack sᵢ ≥ 0)

    **Why soft, not hard.**  The original formulation fed the
    non-penetration rows to OSQP as *hard* inequalities.  On contact-rich
    terrain (holosoma parkour) ``mj_geomDistance`` emits dozens of rows
    whose required recovery ``rhs[i] = −φ − tol`` routinely exceeds what
    the per-iteration box trust region (``±step_size``) can deliver in a
    single step — so the feasible set is empty and OSQP returns
    ``PRIMAL_INFEASIBLE`` *every frame*.  The old code then fell back to a
    box-only solve that dropped **all** collision rows, i.e. the constraints
    were never actually enforced; worse, the active-set thrash that does
    survive shows up as the whole-robot "flashing" jitter.

    The hard formulation was historically justified by the Laplacian
    cost's translation-invariance (a soft penetration penalty could be
    cancelled by lifting the floating base instead of bending the foot).
    That justification no longer holds: ``laplacian_weight`` defaults to 0
    and the base is anchored in absolute world space by ``position_weight``
    (=400), which a slack-absorbed penetration cannot trade against.  So a
    per-row slack with a large quadratic penalty recovers the desired
    behaviour — the foot bends out of penetration whenever reachable — while
    guaranteeing the QP is *always feasible*, eliminating the every-frame
    infeasibility and its fallback jitter.

    Genuine solver failures (``MAX_ITER_REACHED`` etc.) still propagate as
    :class:`hhtools.retarget.interaction_mesh.qp_step.OsqpUnreliableError`
    so the caller can take its bounded box-only step as a last resort.
    """
    from hhtools.retarget.interaction_mesh.qp_step import solve_qp_osqp

    nq = P.shape[0]
    n_ineq = len(J_rows)

    if n_ineq == 0:
        # No collision rows this frame — plain box-constrained QP.
        A = np.eye(nq, dtype=np.float64)
        return solve_qp_osqp(P, q_vec, A, lb.copy(), ub.copy()).astype(
            np.float64, copy=False
        )

    # Augmented variable vector z = [δq (nq); s (n_ineq)].
    n_aug = nq + n_ineq
    rho = float(slack_penalty)

    P_aug = np.zeros((n_aug, n_aug), dtype=np.float64)
    P_aug[:nq, :nq] = P
    # Quadratic slack penalty 0.5·ρ·Σ sᵢ²  → ρ on the slack diagonal.
    P_aug[nq:, nq:] = rho * np.eye(n_ineq, dtype=np.float64)
    q_aug = np.concatenate([q_vec, np.zeros(n_ineq, dtype=np.float64)])

    J_np = np.vstack(J_rows)  # (n_ineq, nq)

    # Constraint blocks (standard OSQP ``l ≤ A z ≤ u``):
    #  1. box on δq:        [I_nq | 0]           ∈ [lb, ub]
    #  2. slack ≥ 0:        [0    | I_m]         ∈ [0, +inf]
    #  3. non-penetration:  [J    | I_m]         ∈ [rhs, +inf]
    A_box = np.hstack([np.eye(nq), np.zeros((nq, n_ineq))])
    A_slack = np.hstack([np.zeros((n_ineq, nq)), np.eye(n_ineq)])
    A_np = np.hstack([J_np, np.eye(n_ineq)])
    A = np.vstack([A_box, A_slack, A_np])

    big = 1e20
    l_full = np.concatenate(
        [lb, np.zeros(n_ineq, dtype=np.float64), np.asarray(rhs, dtype=np.float64)]
    )
    u_full = np.concatenate(
        [ub, np.full(n_ineq, big), np.full(n_ineq, big)]
    )

    z = solve_qp_osqp(P_aug, q_aug, A, l_full, u_full).astype(np.float64, copy=False)
    return z[:nq]


def iterate_mpc_rti(
    model,
    data,
    targets: list[FrameLaplacianTarget],
    *,
    robot_body_names: list[str] | None = None,
    robot_points: list[RobotMpcPoint] | None = None,
    laplacian_weight: float,
    step_size: float,
    smooth_weight: float = 0.6,
    mpc_horizon: int = 1,
    sqp_inner_iters: int = 2,
    # --- absolute world-position cost (anchors global root motion) ---
    position_weight: float = 0.0,
    # --- home-pose Tikhonov on actuated DOFs (breaks null-space yaw drift) ---
    home_pose_weight: float = 0.0,
    # --- holosoma-style hard non-penetration ---
    collision_model=None,
    collision_data=None,
    collision_threshold: float = 0.05,
    penetration_tolerance: float = 0.002,
    collision_fd_epsilon: float = 1e-5,
    collision_max_pairs_per_body: int = 4,
    # --- trust region ---
    base_step_size: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float64]:
    """RTI-style MPC: per frame SQP with Laplacian matching, joint limits,
    temporal smoothing, and (optional) hard non-penetration constraints.

    Returns ``(F, nq)`` generalized coordinates.
    """
    import mujoco

    _ = mpc_horizon
    if not targets:
        return np.zeros((0, model.nq), dtype=np.float64)
    nh = targets[0].n_human_vertices
    if robot_points is not None:
        body_names = [pt.body_name for pt in robot_points[:nh]]
    elif robot_body_names is not None:
        body_names = robot_body_names
    else:
        body_names = _mj_body_names_prefix(model, nh)

    q_lb, q_ub = _build_joint_limits(model)

    traj = np.zeros((len(targets), model.nq), dtype=np.float64)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    q = np.asarray(data.qpos, dtype=np.float64).copy()

    # Snapshot ``home_qpos`` and the actuated-DOF qpos indices once.
    # The home pose is whatever ``mj_resetData`` / keyframe 0 leaves
    # on ``data.qpos`` — i.e. the URDF/MJCF "rest" pose.  We pin all
    # HINGE/SLIDE joints to it as a Tikhonov reference; FREE joint
    # DOFs (XYZ + quat) are excluded by construction.
    home_qpos = q.copy()
    _act_idx: list[int] = []
    for j in range(model.njnt):
        jt = int(model.jnt_type[j])
        if jt in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            _act_idx.append(int(model.jnt_qposadr[j]))
    actuated_qpos_idx = np.asarray(_act_idx, dtype=np.int64)

    Ftot = len(targets)

    # Frame-0 base warm-start: jump the FREE joint **XYZ + quat**
    # straight to the source pelvis pose so the SQP doesn't have to
    # crawl there 5 cm / ≈30° at a time under the per-iteration
    # trust region.  Without this, datasets whose source body yaw
    # differs from the robot's URDF-declared forward (e.g. parc_ms
    # source actor faces +133° while RP1 faces +X) leave the robot
    # rotating roughly 90°/sec while the source sweeps several
    # metres, so the robot's pelvis trajectory ends up rotated
    # ~135° from the source's — the position cost can't recover
    # because each per-frame inner-iter budget is consumed just
    # closing the trans-frame gap.
    #
    # ``targets[0].source_vertices[0]`` is the pelvis joint world
    # position at frame 0 (the first mapped robot point is always
    # the pelvis for SMPL→humanoid maps).  For the orientation we
    # use the corresponding source pelvis quaternion which was
    # stashed on the FrameLaplacianTarget by the precompute step.
    # Subsequent frames warm-start from the previous frame's solve
    # so this one-shot jump only matters at frame 0.
    if Ftot > 0 and int(model.jnt_type[0]) == mujoco.mjtJoint.mjJNT_FREE:
        qadr = int(model.jnt_qposadr[0])
        sv = np.asarray(targets[0].source_vertices, dtype=np.float64)
        if sv.shape[0] > 0:
            q[qadr : qadr + 3] = sv[0, :3]
        sq = getattr(targets[0], "source_root_quat_xyzw", None)
        if sq is not None:
            sq = np.asarray(sq, dtype=np.float64).reshape(4)
            n = float(np.linalg.norm(sq))
            if n > 1e-9:
                sq = sq / n
                q[qadr + 3] = sq[3]
                q[qadr + 4] = sq[0]
                q[qadr + 5] = sq[1]
                q[qadr + 6] = sq[2]

    q_prev: NDArray[np.float64] | None = None
    notify_stride = max(1, Ftot // 40)

    first_iters = max(sqp_inner_iters, 20)

    for f, fr in enumerate(targets):
        iters = first_iters if f == 0 else sqp_inner_iters
        q = sqp_step_laplacian(
            model,
            data,
            q,
            fr,
            body_names,
            robot_points=robot_points,
            laplacian_weight=laplacian_weight,
            step_size=step_size,
            sqp_inner_iters=iters,
            q_prev=q_prev,
            smooth_weight=smooth_weight,
            q_lb=q_lb,
            q_ub=q_ub,
            position_weight=position_weight,
            home_pose_weight=home_pose_weight,
            home_qpos=home_qpos,
            actuated_qpos_idx=actuated_qpos_idx,
            collision_model=collision_model,
            collision_data=collision_data,
            collision_threshold=collision_threshold,
            penetration_tolerance=penetration_tolerance,
            collision_fd_epsilon=collision_fd_epsilon,
            collision_max_pairs_per_body=collision_max_pairs_per_body,
            base_step_size=base_step_size,
        )
        q_prev = q.copy()
        traj[f] = q
        if progress_callback is not None and (f % notify_stride == 0 or f == Ftot - 1):
            try:
                progress_callback(f + 1, Ftot)
            except Exception:
                pass
    return traj


__all__ = [
    "FrameLaplacianTarget",
    "RobotMpcPoint",
    "build_demo_vertices_frame",
    "count_named_mujoco_bodies",
    "iterate_mpc_rti",
    "precompute_target_laplacians",
    "sample_axis_aligned_box",
    "sqp_step_laplacian",
]
