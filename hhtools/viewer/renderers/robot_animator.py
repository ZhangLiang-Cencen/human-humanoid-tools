"""Animate a URDF robot in the Viser scene from a retargeted joint trajectory.

This renderer owns the ``/robot/<preset>/...`` subtree.  Unlike the
T-pose bake path in the Robot tab (which pre-multiplies the yourdfpy
scene-graph transform into the mesh vertices and drops it as a static
handle), we attach each link-node mesh at its *local* geometry coordinates
and drive its per-frame world pose via ``handle.position`` /
``handle.wxyz`` updates.  That lets a single ``RetargetedMotion`` stream
animate the whole robot without re-uploading geometry every frame.

Two other jobs this class also owns because they're naturally tied to the
scenegraph we build:

1. **Ground alignment.**  URDF authors often place the base link at the
   pelvis (``z=0`` at the hip) which leaves the feet hanging below the
   world ``z=0`` ground plane we draw in the viewer.  On construction we
   compute the minimum ``z`` of the T-pose baked mesh and store an offset
   that's added to every subsequent world transform — so the "loaded"
   T-pose sits flush on the grid, and any later animation inherits the
   same offset.

2. **Idempotent handle management.**  :meth:`clear` removes every mesh
   handle this instance owns; :meth:`set_frame_joint_q` is a pure
   per-frame update.  The viewer's render loop calls it at ~60 Hz when a
   retargeted motion is playing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover — type-only
    import trimesh
    import viser

    from hhtools.robot.loader import URDFRobotModel


_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- math helpers


def _rotmat_to_wxyz(R: NDArray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix → unit quaternion in viser's ``(w, x, y, z)`` order.

    Uses the numerically-stable branch selection from Shepperd 1978 — picking
    the largest of ``w, x, y, z`` to divide by so we never hit the
    near-zero case that naive extraction suffers from.
    """
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) or 1.0
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _xyzw_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> NDArray:
    """``(x, y, z, w)`` quaternion → 3x3 rotation matrix, double-precision."""
    n2 = qx * qx + qy * qy + qz * qz + qw * qw
    if n2 < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n2
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- renderer


class RobotAnimator:
    """Per-link mesh handles + per-frame pose updates for a URDF robot.

    Lifecycle:

    1. ``RobotAnimator(server, model, root_path="/robot")`` — parses the
       model's current trimesh scene, computes the ground alignment offset,
       and drops one mesh handle per geometry node at the current
       configuration (T-pose by default).
    2. ``set_frame_joint_q(joint_q_7_plus_ndof, dof_names)`` — called once
       per frame by the viewer's render loop; updates every handle in-place.
    3. ``clear()`` — removes all handles; call before dropping the
       animator if a different robot will be loaded.
    """

    def __init__(
        self,
        server: "viser.ViserServer",
        model: "URDFRobotModel",
        *,
        root_path: str = "/robot",
        world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        import trimesh

        self._server = server
        self._model = model
        self._root_path = f"{root_path}/{model.preset.name}"
        self._handles: dict[str, object] = {}
        # A planar world offset the caller can use to keep the robot clear of
        # the source-motion skeleton (e.g. +X 1 m so the two figures stand
        # side-by-side in the Robot tab).  Applied on top of ground_offset_z.
        self._world_offset = np.asarray(world_offset, dtype=np.float64).reshape(3)
        # Per-node *static* local-geometry transform baked into the handle's
        # initial pose.  yourdfpy's ``scene.graph[node]`` transform bakes both
        # the link frame AND the visual origin; storing the node-local mesh
        # as-is and using the scene-graph transform verbatim keeps math simple.
        self._scene = model.trimesh_scene(collision=False)

        # Ground alignment: put the lowest mesh vertex at z=0 at T-pose.
        # Read T-pose once into a single world_z_min reduction, then the
        # same offset is reused for every frame's root transform.
        min_z: float | None = None
        for node_name in self._scene.graph.nodes_geometry:
            mat, geom_name = self._scene.graph[node_name]
            if geom_name is None:
                continue
            geom = self._scene.geometry.get(geom_name)
            if not isinstance(geom, trimesh.Trimesh) or geom.is_empty:
                continue
            v = np.asarray(geom.vertices, dtype=np.float64)
            # Only need the min z of ``mat @ v`` — do the dot product column-
            # wise to skip building the full homogeneous-augmented array.
            z = (mat[2, 0] * v[:, 0] + mat[2, 1] * v[:, 1] + mat[2, 2] * v[:, 2] + mat[2, 3]).min()
            if min_z is None or z < min_z:
                min_z = float(z)
        # If the robot sits above z=0 at T-pose already (rare: e.g. URDFs that
        # explicitly model a foot offset on the base link), don't translate
        # down — just leave the offset at 0.
        self._ground_offset_z = max(0.0, -min_z) if min_z is not None else 0.0

        for node_name in self._scene.graph.nodes_geometry:
            mat, geom_name = self._scene.graph[node_name]
            if geom_name is None:
                continue
            geom = self._scene.geometry.get(geom_name)
            if not isinstance(geom, trimesh.Trimesh) or geom.is_empty:
                continue
            pos = (
                float(mat[0, 3] + self._world_offset[0]),
                float(mat[1, 3] + self._world_offset[1]),
                float(mat[2, 3] + self._ground_offset_z + self._world_offset[2]),
            )
            wxyz = _rotmat_to_wxyz(mat[:3, :3])
            handle = server.scene.add_mesh_trimesh(
                f"{self._root_path}/{_sanitize(node_name)}",
                geom,
                position=pos,
                wxyz=wxyz,
            )
            self._handles[node_name] = handle

    # ---- public API ----------------------------------------------------------

    @property
    def ground_offset_z(self) -> float:
        """Vertical lift (in metres) applied so the lowest vertex is at z=0."""
        return self._ground_offset_z

    @property
    def world_offset(self) -> tuple[float, float, float]:
        """The ``(dx, dy, dz)`` planar offset applied on top of ground lift."""
        return (
            float(self._world_offset[0]),
            float(self._world_offset[1]),
            float(self._world_offset[2]),
        )

    def num_handles(self) -> int:
        """Number of link-node mesh handles currently in the scene."""
        return len(self._handles)

    def clear(self) -> None:
        """Remove every handle this animator owns from the viser scene."""
        for handle in self._handles.values():
            try:
                handle.remove()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._handles.clear()

    def set_visible(self, visible: bool) -> None:
        """Toggle the robot meshes without destroying buffers.

        Flipping ``handle.visible`` is cheap (a single websocket message per
        handle) and keeps the per-frame update path working: the render
        loop can still call :meth:`set_frame_joint_q` against hidden
        handles and the pose stays in sync the moment the user toggles
        visibility back on.
        """
        v = bool(visible)
        for handle in self._handles.values():
            try:
                handle.visible = v  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_frame_joint_q(
        self,
        joint_q: NDArray,
        dof_names: tuple[str, ...] | list[str],
        *,
        has_root: bool = True,
    ) -> None:
        """Update every handle from a ``(7 + ndof)`` or ``(ndof,)`` joint_q.

        When ``has_root=True`` the first 7 entries are interpreted as
        ``(tx, ty, tz, qx, qy, qz, qw)`` and composed on top of yourdfpy's
        internal FK.  When ``False`` the whole vector is treated as
        actuated-joint values and the robot is rendered at a fixed base.

        ``dof_names`` must name each actuated joint once (order doesn't have
        to match the URDF's parse order — we build a name→value dict).
        Silently ignores a joint_q longer than 7+``len(dof_names)`` so
        callers can pass a full CSV row without slicing.
        """
        q = np.asarray(joint_q, dtype=np.float64).reshape(-1)
        ndof = len(self._model.actuated_joints)
        if has_root:
            if q.shape[0] < 7:
                raise ValueError(
                    f"joint_q has {q.shape[0]} entries but floating-base root "
                    f"needs at least 7"
                )
            root = q[:7]
            dof_slice = q[7:7 + min(ndof, q.shape[0] - 7)]
        else:
            root = None
            dof_slice = q[:ndof]

        cfg: dict[str, float] = {}
        for i, name in enumerate(dof_names[: dof_slice.shape[0]]):
            cfg[str(name)] = float(dof_slice[i])
        try:
            self._model.apply_configuration(cfg)
        except Exception as err:  # pragma: no cover — upstream yourdfpy raises on bad dof
            if not getattr(self, "_warned_apply_cfg", False):
                self._warned_apply_cfg = True
                _log.warning(
                    "apply_configuration failed (%s); limb FK may be wrong — "
                    "still applying floating-base pose from joint_q",
                    err,
                )

        # ``apply_configuration`` mutates the underlying URDF.  Most yourdfpy
        # versions keep a single persistent ``Scene`` whose node transforms
        # update in place, but we defensively re-fetch the scene so a
        # future yourdfpy release that returns a fresh scene object doesn't
        # silently leave us reading the old T-pose transforms.
        scene = self._model.trimesh_scene(collision=False)
        T_root = np.eye(4, dtype=np.float64)
        if root is None:
            # No IK result — show the T-pose and lift it so the feet sit on
            # the ground plane.  ``_ground_offset_z`` is the correction we
            # measured at init from the lowest mesh vertex.
            T_root[:3, 3] = (
                float(self._world_offset[0]),
                float(self._world_offset[1]),
                float(self._ground_offset_z + self._world_offset[2]),
            )
        else:
            # The IK solver already returns a *world-space* floating-base
            # root position — that root_z is chosen to put the feet on the
            # ground given the robot's actual leg length.  Adding
            # ``_ground_offset_z`` on top would double-lift and leave the
            # robot floating at roughly the human's head height (the bug
            # the user reported: "retarget 出来的机器人在半空中").  Trust
            # the IK root and only layer the planar world offset that keeps
            # the robot side-by-side with the human motion.
            T_root[:3, 3] = (
                float(root[0]) + float(self._world_offset[0]),
                float(root[1]) + float(self._world_offset[1]),
                float(root[2]) + float(self._world_offset[2]),
            )
            T_root[:3, :3] = _xyzw_to_rotmat(
                float(root[3]), float(root[4]), float(root[5]), float(root[6])
            )
        for node_name, handle in self._handles.items():
            mat_local, _ = scene.graph[node_name]
            mat_world = T_root @ np.asarray(mat_local, dtype=np.float64)
            pos = mat_world[:3, 3]
            handle.position = (float(pos[0]), float(pos[1]), float(pos[2]))  # type: ignore[attr-defined]
            handle.wxyz = _rotmat_to_wxyz(mat_world[:3, :3])  # type: ignore[attr-defined]


def _sanitize(node_name: str) -> str:
    """Make a URDF link-node name safe for use inside a viser scene path.

    URDF link names are typically already path-safe, but the ``scene.graph``
    occasionally contains names with spaces or duplicates tagged ``_1``,
    ``_2`` — viser's scene-path segments disallow ``/``, so we replace it
    even though we don't expect to hit it.
    """
    return node_name.replace("/", "_")


__all__ = ["RobotAnimator"]
