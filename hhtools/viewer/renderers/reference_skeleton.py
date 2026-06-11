"""Static skeleton renderer for a :class:`HumanReferencePose`.

Used by the Robot tab's "Retarget calibration" mode to draw the
reference human T-pose alongside the (live-updated) robot so the user
can eyeball alignment while moving joint sliders.  The rendering is
one-shot: the pose is fixed for the lifetime of the renderer, so we
only upload a single line-segment handle + a single merged bead mesh
and never issue per-frame vertex updates (unlike
:class:`ScaledSkeletonRenderer`, which animates through a clip).

Design notes
------------
* We borrow ``ScaledSkeletonRenderer``'s bead-strategy: one merged
  icosphere mesh whose per-joint vertex buffer is *immutable* here,
  which lets us use ``add_mesh_trimesh`` (the lighter path; we don't
  need the writable ``.vertices`` that ``add_mesh_simple`` provides).
* Bone topology comes from the :class:`HumanReferencePose`'s own
  ``parent_names`` — not the hard-coded canonical map — so the SMPL-X
  reference (``pelvis`` / ``spine1`` / ``left_collar`` ... ) renders
  correctly without a second skeleton-schema table.
* Path versioning mirrors the scaled-skeleton renderer so repeated
  open/close of the calibration mode doesn't leave stale vertex
  buffers on the client.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from hhtools.core.grounding import foot_floor_z_in_positions
from hhtools.retarget.calibration.reference import HumanReferencePose


def _unit_icosphere() -> tuple[NDArray, NDArray]:
    """Same icosphere the scaled-skeleton renderer uses — kept in sync by value.

    Duplicated instead of imported to keep this file standalone; it's a
    12-vertex mesh, and the scaled-skeleton copy is considered the
    canonical definition.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float32,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int32,
    )
    return verts, faces


_version_counter = 0


def _next_version() -> int:
    global _version_counter
    _version_counter += 1
    return _version_counter


class ReferenceSkeletonRenderer:
    """Line + bead + optional label renderer for a :class:`HumanReferencePose`.

    Parameters
    ----------
    server:
        Active viser server.
    reference:
        The reference pose to display.  Its ``positions`` array is
        interpreted as hips-relative and anchored at ``world_offset``.
    world_offset:
        Planar offset applied to every vertex before upload — use
        this to place the reference skeleton at the robot's spawn
        point instead of overlapping the human-motion view.
    line_color / bead_color:
        RGB tuples (0..255).  Defaults to a soft cyan so the reference
        skeleton visually distinguishes from the orange scaled preview.
    bead_radius:
        Visual radius of the joint beads in metres.
    show_labels:
        When True, a text label with the joint name is placed at each
        joint position.
    root_name:
        Viser scene-path prefix.  Reusing this across two simultaneous
        renderers collides — each instance appends a monotonic version
        suffix to avoid that.
    exclude_bone_indices:
        If provided (e.g. ``{0}``), skip those joint indices in lines,
        beads, and labels — the same idea as the Motion tab's
        *Hide virtual BVH root* (glTF/FBX placeholder nodes) without
        mutating the underlying :class:`HumanReferencePose` math.
    """

    def __init__(
        self,
        server,  # type: ignore[no-untyped-def]
        reference: HumanReferencePose,
        *,
        world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        heading_rad: float = 0.0,
        line_color: tuple[int, int, int] = (100, 180, 255),
        bead_color: tuple[int, int, int] = (160, 210, 255),
        bead_radius: float = 0.028,
        show_labels: bool = False,
        root_name: str = "/reference_human",
        exclude_bone_indices: set[int] | frozenset[int] | None = None,
    ) -> None:
        self._server = server
        self._reference = reference
        self._version = _next_version()
        self._world_offset = np.asarray(world_offset, dtype=np.float32).reshape(3)

        positions = np.asarray(reference.positions, dtype=np.float32).copy()
        if abs(heading_rad) > 1e-6:
            c, s = float(np.cos(heading_rad)), float(np.sin(heading_rad))
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            positions = (positions @ rot.T).astype(np.float32)
        # Foot-floor align (ankle hubs, not toe/end sites) so the blue
        # reference skeleton meets z=0 beside the ground-aligned robot mesh.
        z_floor = foot_floor_z_in_positions(
            positions, tuple(reference.joint_names),
        )
        if abs(z_floor) > 1e-6:
            positions[:, 2] -= np.float32(z_floor)
        positions = positions + self._world_offset[None, :]

        n_j = int(positions.shape[0])
        excluded = np.zeros(n_j, dtype=bool)
        if exclude_bone_indices:
            for bi in exclude_bone_indices:
                if 0 <= int(bi) < n_j:
                    excluded[int(bi)] = True

        # ---- bone segments (see SkeletonRenderer: both endpoints visible) --
        name_to_idx = {n: i for i, n in enumerate(reference.joint_names)}
        segs_src: list[int] = []
        segs_dst: list[int] = []
        for i, parent in enumerate(reference.parent_names):
            if not parent:
                continue
            if parent not in name_to_idx:
                continue
            pidx = int(name_to_idx[parent])
            if excluded[i] or excluded[pidx]:
                continue
            segs_src.append(pidx)
            segs_dst.append(i)

        self._line_handle: Any = None
        if segs_src:
            lines = np.stack(
                [positions[np.asarray(segs_src, dtype=np.int32)],
                 positions[np.asarray(segs_dst, dtype=np.int32)]],
                axis=1,
            ).astype(np.float32)
            colors = np.tile(
                np.asarray(line_color, dtype=np.uint8)[None, None, :],
                (lines.shape[0], 2, 1),
            )
            self._line_handle = server.scene.add_line_segments(
                f"{root_name}/bones_v{self._version}",
                points=lines,
                colors=colors,
                line_width=3.0,
            )

        # ---- joint beads (only non-excluded) -------------------------------
        sphere_v, sphere_f = _unit_icosphere()
        sphere_v = sphere_v * float(bead_radius)
        nverts_per = int(sphere_v.shape[0])
        nfaces_per = int(sphere_f.shape[0])
        vis_idx = np.where(~excluded)[0].astype(np.int32)
        nbeads = int(vis_idx.size)
        if nbeads == 0:
            self._bead_handle = None
        else:
            all_verts = (
                positions[vis_idx][:, None, :] + sphere_v[None, :, :]
            ).reshape(-1, 3).astype(np.float32)
            all_faces = np.empty((nbeads * nfaces_per, 3), dtype=np.int32)
            for k in range(nbeads):
                all_faces[k * nfaces_per:(k + 1) * nfaces_per] = (
                    sphere_f + k * nverts_per
                )
            self._bead_handle = server.scene.add_mesh_simple(
                name=f"{root_name}/joints_v{self._version}",
                vertices=all_verts,
                faces=all_faces,
                color=bead_color,
                flat_shading=True,
                side="double",
            )

        # ---- joint name labels ----------------------------------------------
        self._label_handles: list[Any] = []
        if show_labels:
            self._create_labels(
                positions,
                root_name,
                tuple(int(x) for x in vis_idx.tolist()),
            )

    def _create_labels(
        self,
        positions: NDArray,
        root_name: str,
        joint_indices: tuple[int, ...],
    ) -> None:
        label_offset = np.array([0.0, 0.0, 0.04], dtype=np.float32)
        for i in joint_indices:
            name = self._reference.joint_names[i]
            pos = positions[i] + label_offset
            try:
                lh = self._server.scene.add_label(
                    f"{root_name}/label_{name}_v{self._version}",
                    text=name,
                    position=tuple(float(x) for x in pos),
                    visible=True,
                )
                self._label_handles.append(lh)
            except Exception:
                pass

    # ------------------------------------------------------------------ public
    def clear(self) -> None:
        """Remove all handles (lines, beads, labels).  Idempotent."""
        for attr in ("_line_handle", "_bead_handle"):
            h = getattr(self, attr, None)
            if h is None:
                continue
            try:
                h.visible = False
            except Exception:
                pass
            try:
                h.remove()
            except Exception:
                pass
            setattr(self, attr, None)
        for lh in self._label_handles:
            try:
                lh.remove()
            except Exception:
                pass
        self._label_handles.clear()

    def set_visible(self, visible: bool) -> None:
        for attr in ("_line_handle", "_bead_handle"):
            h = getattr(self, attr, None)
            if h is None:
                continue
            try:
                h.visible = bool(visible)
            except Exception:
                pass

    def set_labels_visible(self, visible: bool) -> None:
        """Show or hide just the joint name labels."""
        for lh in self._label_handles:
            try:
                lh.visible = bool(visible)
            except Exception:
                pass


__all__ = ["ReferenceSkeletonRenderer"]
