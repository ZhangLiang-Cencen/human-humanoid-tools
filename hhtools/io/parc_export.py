# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""PARC ``MSFileData`` (`.pkl`) exporter.

Pairs a humanoid motion (root pose + per-joint local rotations) with a
heightfield terrain into the same on-disk schema as PARC's SIGGRAPH
dec_release: see ``PARC/PARC/util/file_io.py``.

The motion side accepts either:

1. a meshmimic-style hhtools ``.npz`` (15-bone skeleton with **world**
   joint positions / quaternions), or
2. a pre-built ``MSMotionData``-style dictionary (root_pos, root_rot,
   joint_rot, fps, loop_mode).

The terrain side accepts:

1. a meshmimic ``*_terrain.obj`` together with the per-frame object
   pose taken from the .npz, or
2. a pre-built :class:`hhtools.core.scene.TerrainHeightfield`.

Output is a pickle container::

    {
        "motion_data": pickle.dumps({"root_pos", "root_rot", "joint_rot",
                                     "body_contacts", "fps", "loop_mode"}),
        "terrain_data": pickle.dumps({"hf", "hf_maxmin",
                                      "min_point", "dx"}),
        "misc_data": pickle.dumps({...}) | None,
    }

This file can be opened by ``PARC.util.file_io.load_ms_file`` without
modification — i.e. the user can drop it straight into
``$PARC/data/.../my_clip.pkl`` and run any of the PARC training
scripts that consume MS pkls.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q
from hhtools.core.scene import TerrainHeightfield
from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield

if TYPE_CHECKING:
    from hhtools.core.motion import Motion

_log = logging.getLogger(__name__)

_MOTION_DATA_KEY = "motion_data"
_TERRAIN_DATA_KEY = "terrain_data"
_MISC_DATA_KEY = "misc_data"


# ---------------------------------------------------------------------------
# Motion conversion: hhtools schema-v1 npz → PARC MSMotionData dict
# ---------------------------------------------------------------------------

def _quat_inverse_xyzw(q: NDArray[np.floating]) -> NDArray[np.floating]:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., :3] *= -1.0
    n2 = (out * out).sum(axis=-1, keepdims=True)
    out /= np.maximum(n2, 1e-12)
    return out


def world_quaternions_to_local(
    parent_indices: NDArray[np.integer],
    world_quats_xyzw: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Convert per-joint **world** quaternions to **local-to-parent**.

    PARC's ``joint_rot`` field is per-joint local rotation (the
    rotation of joint ``i`` relative to its parent's frame), but
    hhtools schema-v1 npz stores world quaternions per joint.  This
    routine recomposes the local rotations as ``q_local[i] = q_world[parent[i]]^-1 * q_world[i]``.

    The root joint's parent is conventionally ``-1`` and its local
    rotation is set to identity ``(0, 0, 0, 1)`` (PARC stores the
    actual world rotation of the root in ``root_rot`` separately).

    Parameters
    ----------
    parent_indices
        ``(J,)`` integer array; ``-1`` denotes the root.
    world_quats_xyzw
        ``(T, J, 4)`` xyzw world rotations per frame, per joint.

    Returns
    -------
    NDArray of shape ``(T, J, 4)`` with the same xyzw layout.
    """
    parents = np.asarray(parent_indices, dtype=np.int32)
    wq = np.asarray(world_quats_xyzw, dtype=np.float64)
    T, J, _ = wq.shape
    local = np.zeros_like(wq)

    for j in range(J):
        p = int(parents[j])
        if p < 0:
            local[:, j, :] = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            inv_parent = _quat_inverse_xyzw(wq[:, p, :])
            local[:, j, :] = Q.multiply(inv_parent, wq[:, j, :])

    n = np.linalg.norm(local, axis=-1, keepdims=True)
    local /= np.maximum(n, 1e-12)
    return local


def npz_motion_to_ms_motion_dict(
    npz_data: Mapping[str, Any],
    *,
    body_contacts: NDArray[np.floating] | None = None,
    loop_mode: str = "CLAMP",
) -> dict:
    """Pack an hhtools schema-v1 motion npz into PARC's ``MSMotionData`` layout.

    Parameters
    ----------
    npz_data
        ``np.load``ed npz (or any mapping with the same keys).  Must
        carry ``parent_indices``, ``positions``, ``quaternions`` and
        ``framerate``.
    body_contacts
        Optional ``(T, B)`` per-frame contact mask.  Stored as-is in
        ``motion_data["body_contacts"]``.
    loop_mode
        Either ``"CLAMP"`` (default), ``"WRAP"`` or any other PARC
        loop mode tag.

    Returns
    -------
    dict
        ``{"root_pos", "root_rot", "joint_rot", "body_contacts", "fps",
        "loop_mode"}`` ready to pass to :func:`save_parc_pkl`.
    """
    parent_indices = np.asarray(npz_data["parent_indices"], dtype=np.int32)
    positions = np.asarray(npz_data["positions"], dtype=np.float32)
    quaternions = np.asarray(npz_data["quaternions"], dtype=np.float32)
    framerate = float(np.asarray(npz_data["framerate"]).item())

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"positions must be (T, J, 3); got {positions.shape}"
        )
    if quaternions.shape != positions.shape[:2] + (4,):
        raise ValueError(
            f"quaternions must be (T, J, 4); got {quaternions.shape}"
        )

    root_pos = positions[:, 0, :].astype(np.float32, copy=False)
    root_rot = quaternions[:, 0, :].astype(np.float32, copy=False)

    joint_rot = world_quaternions_to_local(parent_indices, quaternions)
    joint_rot = joint_rot.astype(np.float32, copy=False)

    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "joint_rot": joint_rot,
        "body_contacts": (
            None if body_contacts is None
            else np.asarray(body_contacts, dtype=np.float32)
        ),
        "fps": int(round(framerate)),
        "loop_mode": str(loop_mode),
    }


# ---------------------------------------------------------------------------
# Terrain conversion: meshmimic OBJ + per-frame pose → Heightfield
# ---------------------------------------------------------------------------

def npz_terrain_to_heightfield(
    npz_data: Mapping[str, Any],
    *,
    base_dir: Path | str | None = None,
    dx: float = 0.05,
    padding: float = 0.5,
    rotation_quat_xyzw: NDArray[np.floating] | None = None,
    z_offset: float = 0.0,
    scale: float = 1.0,
) -> TerrainHeightfield | None:
    """Read the terrain entry of a meshmimic NPZ and rasterise it.

    Returns ``None`` when the npz has no ``objects_*`` entries or the
    referenced OBJ file cannot be located.  Uses the FIRST object as
    the static terrain (consistent with hhtools' assumption that
    meshmimic clips have a single fixed terrain).

    One frame of ``objects_positions`` / ``objects_quaternions`` is used
    as the rigid terrain pose.  Default is frame 0; when ``meta_json``
    contains ``terrain_heightfield_frame`` (non-negative int), that
    frame index is used instead (DCC exports often leave frame 0 at the
    origin and place props from frame 1 onward).
    """
    import json

    keys = list(npz_data.keys()) if hasattr(npz_data, "keys") else []
    if "objects_mesh_paths" not in keys:
        return None

    mesh_paths = np.asarray(npz_data["objects_mesh_paths"]).astype(str)
    if mesh_paths.size == 0:
        return None

    obj_rel = str(mesh_paths.flat[0])
    obj_path = Path(obj_rel)
    if not obj_path.is_absolute() and base_dir is not None:
        obj_path = Path(base_dir) / obj_rel
    if not obj_path.is_file():
        _log.warning("terrain OBJ not found: %s (relpath=%s)", obj_path, obj_rel)
        return None

    meta: dict[str, Any] = {}
    if "meta_json" in keys:
        try:
            raw_m = npz_data["meta_json"]
            meta = json.loads(str(raw_m.item() if hasattr(raw_m, "item") else raw_m))
        except (json.JSONDecodeError, TypeError, ValueError):
            meta = {}

    hf_t = 0
    try:
        hf_t = int(meta.get("terrain_heightfield_frame", 0))
    except (TypeError, ValueError):
        hf_t = 0

    obj_pos = None
    obj_quat = None
    if "objects_positions" in keys:
        op = np.asarray(npz_data["objects_positions"], dtype=np.float32)
        if op.size:
            n_frames = int(op.shape[0])
            t = max(0, min(hf_t, n_frames - 1))
            obj_pos = op[t, 0, :].astype(np.float32, copy=False)
    if "objects_quaternions" in keys:
        oq = np.asarray(npz_data["objects_quaternions"], dtype=np.float32)
        if oq.size:
            n_frames = int(oq.shape[0])
            t = max(0, min(hf_t, n_frames - 1))
            obj_quat = oq[t, 0, :].astype(np.float32, copy=False)

    mesh_scale = 1.0
    if "objects_scales" in keys:
        os_ = np.asarray(npz_data["objects_scales"], dtype=np.float32)
        if os_.size:
            mesh_scale = float(os_.flat[0])

    return obj_to_heightfield(
        obj_path,
        dx=float(dx),
        padding=float(padding),
        object_position=obj_pos,
        object_quat_xyzw=obj_quat,
        mesh_scale=mesh_scale,
        rotation_quat_xyzw=rotation_quat_xyzw,
        z_offset=float(z_offset),
        scale=float(scale),
    )


# ---------------------------------------------------------------------------
# Retargeted-motion → split per-component pkl bundle (robot / objects / terrain)
# ---------------------------------------------------------------------------

def save_robot_clip_pkls(
    retargeted: Any,                 # RetargetedMotion (avoid import cycle)
    source_motion: "Motion",          # noqa: F821 — runtime use only
    out_dir: str | Path,
    *,
    smpl_scale: float | None = None,
    z_offset: float | None = None,
) -> dict[str, Path]:
    """Drop one directory containing the retargeted clip's three views.

    The user-facing layout (asked for as
    ``assets/save_npz/meshmimic/<folder>/<clip>/``):

    * ``robot.pkl`` — generalised-coordinate trajectory of the robot
      itself.  Stored as a dict::

            {
              "joint_q": ndarray (F, 7 + nq_act) float32
                # [tx, ty, tz, qw, qx, qy, qz, *dof] per frame
                # NOTE the quaternion is **wxyz** as the user spec
                # requested, NOT the CSV ``xyzw`` layout we use
                # internally.  Conversion happens here.
              "dof_names": list[str]   # nq_act actuated joint names
              "sample_rate": float
              "name": str
              "root_quat_format": "wxyz"
              "meta": dict             # passthrough of retargeted.meta
            }

    * ``object_<i>_<name>.pkl`` (one per scene object, only when
      ``source_motion.objects`` is non-empty) — object 6-DoF
      trajectory in the robot frame::

            {
              "name": str
              "positions": ndarray (F, 3) float32
              "quaternions": ndarray (F, 4) float32      # wxyz!
              "extents": ndarray (3,) float32            # robot-scaled
              "mesh_path": str
              "mesh_scale": float
              "sample_rate": float
              "quat_format": "wxyz"
            }

      Positions follow the retargeting transform
      ``p_robot = (p_source − [0, 0, z_min]) · smpl_scale`` so they
      sit on the same ground level as the retargeted robot.  Extents
      are scaled by both the per-object ``scale`` (from the source
      adapter) and ``smpl_scale`` so the rendered prop matches the
      robot scale.

    * ``terrain.pkl`` (only when ``source_motion.terrain`` is set) —
      scaled-into-robot-frame heightfield in the PARC ``MSFileData``
      schema (``terrain_data`` blob populated, others ``None``); this
      is byte-compatible with ``parkour_1.pkl`` and consumable by
      PARC training and our own viewer/retargeter without conversion.

    All three artefacts share ``smpl_scale`` and ``z_offset`` so a
    downstream consumer can re-assemble the scene exactly as the SQP
    saw it.  The function returns a dict mapping a short label
    (``"robot"`` / ``"object_<i>_<name>"`` / ``"terrain"``) to the
    written :class:`pathlib.Path` so the caller can log each one.

    ``smpl_scale`` and ``z_offset`` default to the values stamped on
    ``retargeted.meta`` by the interaction-mesh pipeline
    (``smpl_scale`` / ``source_z_min``).  Pass them explicitly only
    when consuming a retargeting result whose backend does not
    populate that metadata.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = getattr(retargeted, "meta", {}) or {}
    if smpl_scale is None:
        smpl_scale = float(meta.get("smpl_scale", 1.0))
    if z_offset is None:
        z_offset = float(meta.get("source_z_min", 0.0))

    written: dict[str, Path] = {}

    # ---- robot.pkl --------------------------------------------------
    joint_q_csv = np.asarray(retargeted.joint_q, dtype=np.float32)
    if joint_q_csv.ndim != 2 or joint_q_csv.shape[1] < 7:
        raise ValueError(
            f"retargeted.joint_q has unexpected shape {joint_q_csv.shape}; "
            f"expected (F, 7 + nq_act)"
        )
    # CSV layout is [tx,ty,tz, qx,qy,qz,qw, *dof]; user requested wxyz.
    F = int(joint_q_csv.shape[0])
    joint_q_wxyz = np.empty_like(joint_q_csv)
    joint_q_wxyz[:, :3] = joint_q_csv[:, :3]
    joint_q_wxyz[:, 3] = joint_q_csv[:, 6]   # qw
    joint_q_wxyz[:, 4] = joint_q_csv[:, 3]   # qx
    joint_q_wxyz[:, 5] = joint_q_csv[:, 4]   # qy
    joint_q_wxyz[:, 6] = joint_q_csv[:, 5]   # qz
    if joint_q_csv.shape[1] > 7:
        joint_q_wxyz[:, 7:] = joint_q_csv[:, 7:]

    # Strip the leading 6 freejoint slots in dof_names if present
    # (the CSV writer prepends ``root_*`` placeholders) so the
    # tail-end array maps 1:1 onto actuated DOFs.
    dof_all = list(getattr(retargeted, "dof_names", []) or [])
    nq_act = joint_q_csv.shape[1] - 7
    actuated_dof_names = dof_all[-nq_act:] if len(dof_all) >= nq_act else dof_all

    robot_blob: dict[str, object] = {
                "joint_q": joint_q_wxyz,
                "dof_names": actuated_dof_names,
                "sample_rate": float(retargeted.sample_rate),
                "name": str(getattr(retargeted, "name", "retargeted")),
                "root_quat_format": "wxyz",
                "smpl_scale": float(smpl_scale),
                "z_offset": float(z_offset),
                "meta": {k: str(v) for k, v in meta.items()},
            }

    robot_pkl = out_dir / "robot.pkl"
    with open(robot_pkl, "wb") as f:
        pickle.dump(robot_blob, f)
    _log.info("Robot pkl written to %s (frames=%d, dof=%d)", robot_pkl, F, nq_act)
    written["robot"] = robot_pkl

    # ---- object_<i>_<name>.pkl --------------------------------------
    for idx, ob in enumerate(getattr(source_motion, "objects", []) or []):
        op = np.asarray(ob.positions, dtype=np.float32).copy()
        # Source-frame → robot-frame: identical transform to
        # ``_build_scaled_object_points`` in the retargeting
        # pipeline, so the prop sits relative to the retargeted
        # robot the same way it sat relative to the source actor.
        op[:, 2] -= float(z_offset)
        op *= float(smpl_scale)

        # Quaternion convention swap: hhtools stores xyzw, user
        # asked for wxyz.
        oq_xyzw = np.asarray(ob.quaternions, dtype=np.float32)
        oq_wxyz = np.empty_like(oq_xyzw)
        oq_wxyz[..., 0] = oq_xyzw[..., 3]
        oq_wxyz[..., 1:] = oq_xyzw[..., :3]

        # Extents follow the box-size scaling used by the
        # interaction-mesh sampler so a downstream renderer
        # showing a placeholder cuboid lands on the right cells.
        extents = (
            np.asarray(ob.extents, dtype=np.float32).reshape(3)
            * float(getattr(ob, "scale", 1.0))
            * float(smpl_scale)
        )

        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(ob.name))
        obj_pkl = out_dir / f"object_{idx}_{safe_name}.pkl"
        with open(obj_pkl, "wb") as f:
            pickle.dump(
                {
                    "name": str(ob.name),
                    "positions": op,
                    "quaternions": oq_wxyz,
                    "extents": extents,
                    "mesh_path": str(getattr(ob, "mesh_path", "") or ""),
                    "mesh_scale": float(getattr(ob, "scale", 1.0)),
                    "sample_rate": float(retargeted.sample_rate),
                    "quat_format": "wxyz",
                    "smpl_scale": float(smpl_scale),
                    "z_offset": float(z_offset),
                },
                f,
            )
        _log.info("Object pkl written to %s (frames=%d)", obj_pkl, op.shape[0])
        written[f"object_{idx}_{safe_name}"] = obj_pkl

    # ---- terrain.pkl ------------------------------------------------
    if source_motion.terrain is not None:
        terrain_robot = source_motion.terrain.scaled(
            float(smpl_scale), z_offset=float(z_offset),
        )
        terrain_pkl = out_dir / "terrain.pkl"
        save_parc_pkl(
            terrain_pkl,
            motion_data=None,          # PARC pkl with terrain only,
            terrain_data=terrain_robot,  # mirroring parkour_1.pkl.
            misc_data=None,
        )
        written["terrain"] = terrain_pkl

    return written


# ---------------------------------------------------------------------------
# Retargeted-motion → hhtools schema-v1 Motion (NPZ + sidecar terrain pkl)
# ---------------------------------------------------------------------------

def retargeted_to_motion(
    retargeted: Any,            # RetargetedMotion (avoid import cycle)
    source_motion: "Motion",     # noqa: F821 — used at runtime, not at import
    robot: Any,                 # URDFRobotModel (avoid import cycle)
    *,
    smpl_scale: float | None = None,
    z_offset: float | None = None,
) -> "Motion":
    """Convert a robot retarget result into a hhtools-schema :class:`Motion`.

    The output ``Motion`` shares the schema used everywhere else in
    hhtools (``bone_names`` / ``positions`` / ``quaternions`` / optional
    ``terrain``), so :func:`hhtools.io.npz.save_npz` can serialise it
    to the same NPZ + sidecar PKL format that source clips use.  This
    is the format the user asked for in the export spec:

    * ``<stem>.npz`` — robot link world positions & quaternions, in
      schema parity with the AMASS / GLB clips already shipped.
    * ``<stem>.pkl`` — sidecar PARC-format heightfield terrain
      (written automatically by :func:`save_npz` whenever the Motion
      carries a non-``None`` ``terrain``), in the same layout as
      ``assets/motions/meshmimic/holosoma/parkour_1/parkour_1.pkl``.

    The bone hierarchy is taken from the URDF body topology (worldbody
    skipped).  ``parent_indices[i] = -1`` denotes the root link
    (``floating_base``); all other links point to their MJCF parent
    body's index in the bone list.

    Per-frame world positions and quaternions are obtained by stepping
    MuJoCo through every ``joint_q`` row.  Quaternions are converted
    from MuJoCo's ``wxyz`` convention into hhtools's ``xyzw`` schema.

    The terrain is always emitted in the **same** scale as the robot
    motion: ``source_motion.terrain.scaled(smpl_scale, z_offset)``.
    Callers that already know these values from the pipeline metadata
    can pass them in; otherwise we fall back to ``retargeted.meta``
    (which the interaction-mesh pipeline populates with
    ``smpl_scale`` / ``source_z_min``).

    Parameters
    ----------
    retargeted
        :class:`hhtools.retarget.retarget_result.RetargetedMotion`
        from any backend.  Must have ``joint_q`` of shape
        ``(F, 7 + nq_act)`` with ``[tx, ty, tz, qx, qy, qz, qw, …]``
        layout in CSV/xyzw convention.
    source_motion
        Original :class:`Motion`; only ``terrain`` is consumed here.
    robot
        :class:`hhtools.robot.loader.URDFRobotModel`; provides the
        compiled MuJoCo model used for FK + the link names exposed to
        the hierarchy.
    smpl_scale, z_offset
        ``robot_height / human_height`` and the source motion's
        per-clip ``z_min`` (subtracted before scaling).  When ``None``
        we read them from ``retargeted.meta`` (``smpl_scale`` /
        ``source_z_min``).
    """
    from hhtools.core.hierarchy import Hierarchy
    from hhtools.core.motion import Motion

    import mujoco

    mj_model = getattr(robot, "mujoco_model", None)
    if mj_model is None:
        raise ValueError("robot has no compiled MuJoCo model (mujoco_model is None)")

    joint_q = np.asarray(retargeted.joint_q, dtype=np.float64)
    F = int(joint_q.shape[0])

    # Bone names = URDF link names (skip worldbody at index 0).
    bone_names: list[str] = []
    parents: list[int] = []
    parent_ids = np.asarray(mj_model.body_parentid, dtype=np.int32)
    for bid in range(1, int(mj_model.nbody)):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
        bone_names.append(name)
        pid = int(parent_ids[bid])
        # Worldbody (id=0) is the kinematic root in MJCF — represent
        # this in hhtools schema by setting the parent index to -1.
        # All other parents shift by 1 because we dropped the worldbody.
        parents.append(-1 if pid <= 0 else pid - 1)

    actuated_qadr: list[int] = []
    for j in range(mj_model.njnt):
        jt = int(mj_model.jnt_type[j])
        if jt in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            actuated_qadr.append(int(mj_model.jnt_qposadr[j]))
    nq_act = len(actuated_qadr)

    if joint_q.shape[1] != 7 + nq_act:
        raise ValueError(
            f"retargeted.joint_q has {joint_q.shape[1]} columns but model "
            f"has nq={mj_model.nq} (7 root + {nq_act} actuated)"
        )

    n_bones = len(bone_names)
    positions = np.zeros((F, n_bones, 3), dtype=np.float32)
    quaternions = np.zeros((F, n_bones, 4), dtype=np.float32)
    data = mujoco.MjData(mj_model)

    for f in range(F):
        row = joint_q[f]
        qpos = data.qpos
        qpos[:3] = row[:3]
        # CSV xyzw → MuJoCo wxyz on the freejoint.
        qpos[3] = float(row[6])
        qpos[4] = float(row[3])
        qpos[5] = float(row[4])
        qpos[6] = float(row[5])
        for k, qadr in enumerate(actuated_qadr):
            qpos[qadr] = float(row[7 + k])
        mujoco.mj_forward(mj_model, data)

        for bid in range(1, int(mj_model.nbody)):
            i = bid - 1
            positions[f, i, :] = data.xpos[bid].astype(np.float32)
            xq_w = data.xquat[bid]  # MuJoCo wxyz
            quaternions[f, i, 0] = float(xq_w[1])  # qx
            quaternions[f, i, 1] = float(xq_w[2])  # qy
            quaternions[f, i, 2] = float(xq_w[3])  # qz
            quaternions[f, i, 3] = float(xq_w[0])  # qw

    # Resolve scale / z_offset for the terrain, falling back to the
    # metadata stamped by InteractionMeshPipeline if the caller did
    # not pass them in explicitly.
    meta = getattr(retargeted, "meta", {}) or {}
    if smpl_scale is None:
        smpl_scale = float(meta.get("smpl_scale", 1.0))
    if z_offset is None:
        z_offset = float(meta.get("source_z_min", 0.0))

    terrain_robot: TerrainHeightfield | None = None
    if source_motion.terrain is not None:
        terrain_robot = source_motion.terrain.scaled(
            float(smpl_scale), z_offset=float(z_offset),
        )

    hierarchy = Hierarchy.from_parent_indices(bone_names, parents)
    return Motion(
        name=str(getattr(retargeted, "name", "retargeted")),
        hierarchy=hierarchy,
        positions=positions,
        quaternions=quaternions,
        framerate=float(retargeted.sample_rate),
        up_axis="Z",
        source_format="unknown",
        meta={
            "retarget_meta": {k: str(v) for k, v in meta.items()},
            "robot_preset": getattr(getattr(robot, "preset", None), "name", "unknown"),
        },
        terrain=terrain_robot,
    )


def save_retargeted_motion_npz(
    retargeted: Any,
    source_motion: "Motion",
    robot: Any,
    out_npz: str | Path,
    *,
    smpl_scale: float | None = None,
    z_offset: float | None = None,
    compressed: bool = True,
) -> Path:
    """Write a retargeted clip as ``<stem>.npz`` + ``<stem>.pkl`` (terrain).

    Calls :func:`retargeted_to_motion` for the conversion and then
    :func:`hhtools.io.npz.save_npz` for the serialisation; the
    sidecar PKL is dropped automatically alongside the NPZ when the
    converted motion has terrain attached.

    Returns the resolved NPZ path so callers can log / link it.
    """
    from hhtools.io.npz import save_npz

    out_npz = Path(out_npz)
    motion = retargeted_to_motion(
        retargeted, source_motion, robot,
        smpl_scale=smpl_scale, z_offset=z_offset,
    )
    save_npz(motion, out_npz, compressed=compressed)
    return out_npz


# ---------------------------------------------------------------------------
# Retargeted-motion → PARC pkl
# ---------------------------------------------------------------------------

def retargeted_motion_to_parc_pkl(
    retargeted: Any,            # RetargetedMotion (avoid import cycle)
    source_motion: "Motion",     # noqa: F821
    robot: Any,                 # URDFRobotModel (avoid import cycle)
    out_pkl: str | Path,
    *,
    smpl_scale: float,
    z_offset: float,
    body_contacts: NDArray[np.floating] | None = None,
    loop_mode: str = "CLAMP",
) -> "TerrainHeightfield | None":
    """Write a PARC ``MSFileData`` pkl from a robot retarget result.

    Pairs the retargeted motion (G1 / RP1 / …) with the source clip's
    terrain transformed into the robot frame using the same
    ``(scale, z_offset)`` chain applied during retargeting:

        terrain_robot = source_motion.terrain.scaled(smpl_scale, z_offset)

    so the pkl that ships to PARC training has motion and terrain in
    the same coordinate frame the policy will actually see at run time.

    Per-joint local rotations are computed by stepping MuJoCo through
    every frame's ``qpos`` and recovering ``q_local[bid] = q_world[parent_bid]^-1 * q_world[bid]``.
    The world rotations come from ``data.xquat`` (MuJoCo wxyz),
    converted to xyzw to match the PARC schema.

    Parameters
    ----------
    retargeted
        :class:`hhtools.retarget.retarget_result.RetargetedMotion` from
        any backend.  Must have ``joint_q`` of shape ``(F, 7 + nq_act)``
        with the first 7 columns being ``(tx, ty, tz, qx, qy, qz, qw)``
        and the remaining columns matching the URDF's hinge / slide
        joint declaration order.
    source_motion
        Original :class:`hhtools.core.motion.Motion`; only its
        ``terrain`` field is used.
    robot
        :class:`hhtools.robot.loader.URDFRobotModel`; provides the
        compiled MuJoCo model used for FK.
    out_pkl
        Output path.
    smpl_scale, z_offset
        ``robot_height / human_height`` and the source motion's per-clip
        ``z_min`` (subtracted before scaling) — the values logged by
        :class:`InteractionMeshPipeline` when it produced ``retargeted``.
    body_contacts
        Optional per-frame contact mask, stored as-is in ``MSMotionData``.
    loop_mode
        Stored in ``MSMotionData``; PARC default is ``"CLAMP"``.

    Returns
    -------
    TerrainHeightfield or None
        The robot-frame terrain that was written into the pkl, ``None``
        if the source motion had no terrain.
    """
    import mujoco

    mj_model = getattr(robot, "mujoco_model", None)
    if mj_model is None:
        raise ValueError("robot has no compiled MuJoCo model (mujoco_model is None)")

    joint_q = np.asarray(retargeted.joint_q, dtype=np.float64)
    F = int(joint_q.shape[0])

    nq_act = mj_model.nq - 7
    if joint_q.shape[1] != 7 + nq_act:
        raise ValueError(
            f"retargeted.joint_q has {joint_q.shape[1]} columns but model "
            f"has nq={mj_model.nq} (7 root + {nq_act} actuated)"
        )

    # Build the qpos rebuild order: first 7 from root7 (xyz + xyzw), then
    # actuated joints in the same order as ``pack_joint_q_csv`` emitted them.
    actuated_qadr: list[int] = []
    for j in range(mj_model.njnt):
        jt = int(mj_model.jnt_type[j])
        if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            actuated_qadr.append(int(mj_model.jnt_qposadr[j]))
    if len(actuated_qadr) != nq_act:
        raise ValueError(
            f"actuated qadr count {len(actuated_qadr)} disagrees with nq_act={nq_act}"
        )

    nbody = int(mj_model.nbody)
    # Skip body 0 (worldbody); PARC's joint_rot is one rotation per
    # link.  Order matches MuJoCo body id 1..nbody-1 which itself
    # mirrors URDF link declaration order (modulo the worldbody merge).
    n_joints = nbody - 1
    parent_ids = np.asarray(mj_model.body_parentid, dtype=np.int32)

    data = mujoco.MjData(mj_model)

    root_pos = np.zeros((F, 3), dtype=np.float32)
    root_rot = np.zeros((F, 4), dtype=np.float32)
    joint_rot = np.zeros((F, n_joints, 4), dtype=np.float32)

    for f in range(F):
        # Repack hhtools CSV layout into MuJoCo qpos.
        qpos = data.qpos
        qpos[:3] = joint_q[f, :3]
        # CSV: (qx, qy, qz, qw); MuJoCo: (qw, qx, qy, qz).
        qpos[3] = joint_q[f, 6]
        qpos[4] = joint_q[f, 3]
        qpos[5] = joint_q[f, 4]
        qpos[6] = joint_q[f, 5]
        for k, qadr in enumerate(actuated_qadr):
            qpos[qadr] = joint_q[f, 7 + k]

        mujoco.mj_forward(mj_model, data)

        root_pos[f] = joint_q[f, :3].astype(np.float32)
        root_rot[f] = joint_q[f, 3:7].astype(np.float32)

        # data.xquat is (nbody, 4) wxyz; we want xyzw local-to-parent.
        xq_w = np.asarray(data.xquat, dtype=np.float64)
        xq_xyzw = np.empty_like(xq_w)
        xq_xyzw[:, :3] = xq_w[:, 1:]
        xq_xyzw[:, 3] = xq_w[:, 0]

        for bid in range(1, nbody):
            pid = int(parent_ids[bid])
            q_world = xq_xyzw[bid]
            if pid <= 0:
                # Parent is worldbody — the local rotation IS the world
                # rotation of the root.  PARC stores this redundantly in
                # ``root_rot``; convention here is to mirror that.
                joint_rot[f, bid - 1, :] = q_world.astype(np.float32)
            else:
                q_parent_inv = _quat_inverse_xyzw(xq_xyzw[pid])
                q_local = Q.multiply(
                    q_parent_inv[None, :], q_world[None, :]
                )[0]
                joint_rot[f, bid - 1, :] = q_local.astype(np.float32)

    n = np.linalg.norm(joint_rot, axis=-1, keepdims=True)
    joint_rot /= np.maximum(n, 1e-12)

    motion_dict = {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "joint_rot": joint_rot,
        "body_contacts": (
            None if body_contacts is None
            else np.asarray(body_contacts, dtype=np.float32)
        ),
        "fps": int(round(float(retargeted.sample_rate))),
        "loop_mode": str(loop_mode),
    }

    terrain_robot = None
    if source_motion.terrain is not None:
        terrain_robot = source_motion.terrain.scaled(
            float(smpl_scale), z_offset=float(z_offset),
        )

    save_parc_pkl(
        out_pkl,
        motion_data=motion_dict,
        terrain_data=terrain_robot,
        misc_data=None,
    )
    return terrain_robot


def load_parc_pkl_terrain(pkl_path: str | Path) -> TerrainHeightfield | None:
    """Read just the ``terrain_data`` blob from a PARC-format pkl.

    Returns ``None`` if the pkl is missing the terrain blob (motion-only
    clip) or cannot be parsed.  The motion / misc blobs are skipped to
    keep the loader cheap when the caller already has motion data from
    elsewhere (e.g. an hhtools npz sitting next to the pkl sidecar).
    """
    try:
        with open(pkl_path, "rb") as f:
            container = pickle.load(f)
    except (FileNotFoundError, OSError):
        return None
    blob = container.get(_TERRAIN_DATA_KEY) if isinstance(container, Mapping) else None
    if blob is None:
        return None
    try:
        td = pickle.loads(blob)
    except Exception:
        _log.warning("could not unpickle terrain_data in %s", pkl_path)
        return None
    if td is None:
        return None
    try:
        return TerrainHeightfield.from_ms_terrain_data_dict(td)
    except Exception as exc:
        _log.warning("malformed terrain_data in %s: %s", pkl_path, exc)
        return None


# ---------------------------------------------------------------------------
# Top-level container writer (PARC ``save_ms_file`` clone)
# ---------------------------------------------------------------------------

def save_parc_pkl(
    output_path: str | Path,
    motion_data: Mapping[str, Any] | None,
    terrain_data: Mapping[str, Any] | TerrainHeightfield | None,
    misc_data: Mapping[str, Any] | None = None,
) -> None:
    """Write a PARC-format ``.pkl`` (matches ``PARC.util.file_io.save_ms_file``).

    Each top-level value is individually ``pickle.dumps``ed so PARC
    ``load_ms_file`` can lazily decode them.

    Parameters
    ----------
    output_path
        Destination ``.pkl`` path.  Parent directories are created.
    motion_data
        Dict with the ``MSMotionData`` keys (see
        :func:`npz_motion_to_ms_motion_dict`).  ``None`` → blob is None.
    terrain_data
        Either a :class:`hhtools.core.scene.TerrainHeightfield` or a
        dict with the ``MSTerrainData`` keys.  ``None`` → blob is None.
    misc_data
        Free-form metadata.  PARC stores ``hf_mask_inds`` and
        observation tensors here; we leave it ``None`` by default and
        let downstream tools (``compute_hf_extra_vals``) populate it.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(terrain_data, TerrainHeightfield):
        td_dict = terrain_data.to_ms_terrain_data_dict()
    elif terrain_data is None:
        td_dict = None
    else:
        td_dict = dict(terrain_data)

    md_dict = None if motion_data is None else dict(motion_data)
    misc_dict = None if misc_data is None else dict(misc_data)

    container = {
        _MOTION_DATA_KEY: None if md_dict is None else pickle.dumps(md_dict),
        _TERRAIN_DATA_KEY: None if td_dict is None else pickle.dumps(td_dict),
        _MISC_DATA_KEY: None if misc_dict is None else pickle.dumps(misc_dict),
    }
    with open(out, "wb") as f:
        pickle.dump(container, f)

    _log.info("PARC pkl written to %s", out)


# ---------------------------------------------------------------------------
# High-level batch helper: meshmimic clip directory → PARC pkl
# ---------------------------------------------------------------------------

def meshmimic_clip_to_parc_pkl(
    clip_npz: str | Path,
    output_pkl: str | Path,
    *,
    rotation_quat_xyzw: NDArray[np.floating] | None = None,
    z_offset: float = 0.0,
    scale: float = 1.0,
    dx: float = 0.05,
    padding: float = 0.5,
    loop_mode: str = "CLAMP",
    body_contacts: NDArray[np.floating] | None = None,
) -> TerrainHeightfield | None:
    """Convert a meshmimic clip (``.npz`` + sibling ``*_terrain.obj``) to a PARC pkl.

    Returns the resulting :class:`TerrainHeightfield` (or ``None`` if the
    clip had no terrain) so callers can feed it back into the retargeter
    for visualisation / hard-NP collision without re-rasterising the OBJ.
    """
    clip = Path(clip_npz)
    if not clip.is_file():
        raise FileNotFoundError(clip)
    npz = np.load(clip, allow_pickle=True)
    npz_dict = {k: npz[k] for k in npz.keys()}

    motion_dict = npz_motion_to_ms_motion_dict(
        npz_dict, body_contacts=body_contacts, loop_mode=loop_mode,
    )
    hf = npz_terrain_to_heightfield(
        npz_dict,
        base_dir=clip.parent,
        dx=dx,
        padding=padding,
        rotation_quat_xyzw=rotation_quat_xyzw,
        z_offset=z_offset,
        scale=scale,
    )

    save_parc_pkl(
        output_pkl,
        motion_data=motion_dict,
        terrain_data=hf,
        misc_data=None,
    )
    return hf


__all__ = [
    "TerrainHeightfield",
    "world_quaternions_to_local",
    "npz_motion_to_ms_motion_dict",
    "npz_terrain_to_heightfield",
    "save_parc_pkl",
    "load_parc_pkl_terrain",
    "meshmimic_clip_to_parc_pkl",
    "retargeted_motion_to_parc_pkl",
]
