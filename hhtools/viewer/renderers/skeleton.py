"""Viser renderer that draws a :class:`Motion` as line segments + per-joint spheres.

Two buffers are kept: one big line-segment handle for the bones, and one merged mesh handle
for the joint beads. The mesh uses a canonical icosahedron instanced per joint so radii can
vary per bone (finger / toe beads shrink while body joints keep a larger radius). Only the
vertices are rewritten every frame, so per-frame Viser traffic stays small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.core.motion import Motion


def _unit_icosphere() -> tuple[NDArray, NDArray]:
    """Tiny icosahedron (12 verts, 20 tris), unit radius. Shared joint-bead template."""
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


@dataclass
class _RendererState:
    motion: Motion
    segment_src_idx: NDArray  # parent indices for each bone that has a parent
    segment_dst_idx: NDArray  # child indices
    visible_joint_idx: NDArray  # (Jk,) joint indices that get a bead
    joint_radii: NDArray  # (Jk,) radius per visible joint
    line_color: tuple[int, int, int] = (56, 189, 248)


class SkeletonRenderer:
    """Draw a Motion into a viser server's scene at ``root_name``."""

    def __init__(
        self,
        server,  # type: ignore[no-untyped-def]  (viser.ViserServer)
        root_name: str = "/skeleton",
        joint_color: tuple[int, int, int] = (248, 220, 106),
        default_joint_radius: float = 0.02,
        line_color: tuple[int, int, int] = (56, 189, 248),
    ) -> None:
        self._server = server
        self._root = root_name
        self._joint_color = joint_color
        self._default_joint_radius = float(default_joint_radius)
        self._line_color = line_color
        self._state: _RendererState | None = None
        self._joints_handle = None
        self._lines_handle = None
        self._joints_faces_cache: NDArray | None = None
        self._sph_verts, self._sph_faces = _unit_icosphere()
        self._visible = True
        # Version counter appended to handle names so the OLD mesh's name never collides
        # with the newly-added mesh's name during an async-remove-then-add on motion switch.
        # Without this, the Viser client has been observed to apply fresh vertex buffers on
        # top of the previous motion's face topology (visible as spurious "wrong links" or a
        # subset of bones simply not rendering until play resumes).
        self._version = 0

    # ------------------------------------------------------------------ public API

    def clear(self) -> None:
        """Drop all scene nodes owned by this renderer.

        Order: hide *before* remove so any async-queued vertex update from the old
        motion cannot visually "stick around" after the new motion is wired up.
        """
        for handle_name in ("_joints_handle", "_lines_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                try:
                    handle.visible = False
                except Exception:
                    pass
                try:
                    handle.remove()
                except Exception:
                    pass
            setattr(self, handle_name, None)
        self._state = None
        self._joints_faces_cache = None

    def set_visible(self, visible: bool) -> None:
        """Toggle the bone lines + joint beads without destroying buffers.

        Using this to respond to UI checkboxes keeps playback continuous: frames keep
        advancing behind the scenes, only the scene nodes' ``visible`` flag flips.
        """
        self._visible = bool(visible)
        for handle in (self._joints_handle, self._lines_handle):
            if handle is not None:
                try:
                    handle.visible = self._visible
                except Exception:
                    pass

    def set_motion(
        self,
        motion: Motion,
        *,
        exclude_bones: set[int] | None = None,
        bone_radii: NDArray | None = None,
    ) -> None:
        """Switch to a new motion; caches derived indices so per-frame cost is cheap."""
        self.clear()
        parents = motion.hierarchy.parent_indices
        num_bones = int(motion.num_bones)

        excluded = np.zeros(num_bones, dtype=bool)
        if exclude_bones:
            excluded[list(exclude_bones)] = True
        keep_edge = np.array(
            [
                parents[i] >= 0 and not excluded[i] and not excluded[int(parents[i])]
                for i in range(num_bones)
            ],
            dtype=bool,
        )
        src = parents[keep_edge].astype(np.int32)
        dst = np.where(keep_edge)[0].astype(np.int32)

        if bone_radii is not None:
            radii_full = np.asarray(bone_radii, dtype=np.float32).reshape(-1)
            assert radii_full.shape[0] == num_bones, "bone_radii must have one value per bone"
        else:
            radii_full = np.full((num_bones,), self._default_joint_radius, dtype=np.float32)

        keep_joint = ~excluded
        visible_joint_idx = np.where(keep_joint)[0].astype(np.int32)
        # Joint beads 80% of bone radius so they don't engulf finger/toe capsules.
        joint_radii = (radii_full[visible_joint_idx] * 0.8).astype(np.float32)

        self._state = _RendererState(
            motion=motion,
            segment_src_idx=src,
            segment_dst_idx=dst,
            visible_joint_idx=visible_joint_idx,
            joint_radii=joint_radii,
            line_color=self._line_color,
        )
        self._joints_faces_cache = self._build_joint_faces(int(visible_joint_idx.size))
        self._version += 1
        # Note: caller is expected to call ``set_frame`` right after this so the scene picks
        # up the user's current playback cursor (not an implicit jump back to frame 0).

    def set_frame(self, frame: int) -> None:
        state = self._state
        if state is None:
            return
        frame = int(np.clip(frame, 0, state.motion.num_frames - 1))
        positions = state.motion.positions[frame]

        if state.segment_src_idx.size > 0:
            segments = np.stack(
                [positions[state.segment_src_idx], positions[state.segment_dst_idx]],
                axis=1,
            )
        else:
            segments = np.zeros((0, 2, 3), dtype=np.float32)

        self._update_joint_mesh(positions, state.visible_joint_idx, state.joint_radii)
        self._update_segments(segments, state.line_color)

    # --------------------------------------------------------------- viser glue

    def _update_joint_mesh(
        self, positions: NDArray, visible_idx: NDArray, radii: NDArray
    ) -> None:
        if visible_idx.size == 0 or self._joints_faces_cache is None:
            return
        centers = positions[visible_idx]  # (Jk, 3)
        verts = centers[:, None, :] + self._sph_verts[None, :, :] * radii[:, None, None]
        verts = verts.reshape(-1, 3).astype(np.float32)
        if self._joints_handle is None:
            self._joints_handle = self._server.scene.add_mesh_simple(
                name=f"{self._root}/joints_v{self._version}",
                vertices=verts,
                faces=self._joints_faces_cache,
                color=self._joint_color,
                flat_shading=True,
                side="double",
            )
            self._joints_handle.visible = self._visible
        else:
            self._joints_handle.vertices = verts

    def _update_segments(self, segments: NDArray, color: tuple[int, int, int]) -> None:
        if segments.shape[0] == 0:
            if self._lines_handle is not None:
                try:
                    self._lines_handle.remove()
                except Exception:
                    pass
                self._lines_handle = None
            return
        color_arr = np.tile(
            np.asarray(color, dtype=np.uint8)[None, None, :], (segments.shape[0], 2, 1)
        )
        if self._lines_handle is None:
            self._lines_handle = self._server.scene.add_line_segments(
                name=f"{self._root}/bones_v{self._version}",
                points=segments.astype(np.float32),
                colors=color_arr,
                line_width=3.0,
            )
            self._lines_handle.visible = self._visible
        else:
            self._lines_handle.points = segments.astype(np.float32)

    def _build_joint_faces(self, num_joints: int) -> NDArray:
        if num_joints == 0:
            return np.zeros((0, 3), dtype=np.int32)
        verts_per_joint = int(self._sph_verts.shape[0])
        faces = np.concatenate(
            [self._sph_faces + j * verts_per_joint for j in range(num_joints)], axis=0
        )
        return faces.astype(np.int32)


__all__ = ["SkeletonRenderer"]
