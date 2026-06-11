"""Preview the scaler's pre-IK effector targets as a skeleton overlay.

Meant to sit next to the ``RobotAnimator`` in the Robot tab so the user can
eyeball *what the scaler is asking the robot to chase* before paying for
the Newton IK solve.  This mirrors the diagnostic dump that soma-retargeter
produces when its scaler config runs in isolation (e.g.
``lafan_to_rp1_scaler_config.json`` → the rescaled LAFAN figure).

For dense rigs (optional ``source_*`` fields on :class:`ScaledMotionPreview`),
bone segments follow the motion hierarchy and positions come from the scaler
tensor so the yellow figure matches the source skeleton topology, not only the
canonical ``ik_map`` subset.

Draw strategy:

- One line-segment handle per bone so we can update every bone's endpoints
  in a single viser message per frame.
- One merged mesh handle (an instanced icosahedron per visible joint) for
  the beads, so joint positions are visible even when the bone line
  rendering gets culled by a particular client.
- ``NaN`` slots in the scaler's effector tensor are simply skipped — the
  ik_map can reference joints the scaler didn't populate, and we don't
  want those to collapse into an origin-cluster visually.
- Every recreate bumps a ``version`` counter appended to the viser scene
  paths (``.../bones_v3``, ``.../joints_v3``) so a remove-then-add sequence
  never collides with a stale vertex buffer on the client.  Without the
  suffix we observed the second preview after a recreate rendering as
  "beads only" — viser silently reused the old buffer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


# Hard-coded canonical parent map — mirrors ``configs/skeleton_presets/
# canonical_human.yaml``.  Duplicated inline rather than read from disk on
# init so the viewer stays functional in builds that don't ship the YAML.
_CANONICAL_PARENT: dict[str, str | None] = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    "left_shoulder": "chest",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "right_shoulder": "chest",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_hip": "hips",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "left_foot": "left_ankle",
    "right_hip": "hips",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "right_foot": "right_ankle",
}


def _unit_icosphere() -> tuple[NDArray, NDArray]:
    """Tiny icosahedron (12 verts, 20 tris).  Shared joint-bead template."""
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


# Module-level version counter so multiple instances in the same viewer
# session never share a path — even if the caller only creates one
# :class:`ScaledSkeletonRenderer` at a time, the server's GC occasionally
# lags behind client-side caching of removed mesh nodes.
_version_counter = 0


def _next_version() -> int:
    global _version_counter
    _version_counter += 1
    return _version_counter


class ScaledSkeletonRenderer:
    """Line-segment + bead renderer for a :class:`ScaledMotionPreview`."""

    def __init__(
        self,
        server,  # type: ignore[no-untyped-def]
        preview,  # ScaledMotionPreview
        *,
        root_name: str = "/scaled_human",
        world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        line_color: tuple[int, int, int] = (240, 180, 90),
        bead_color: tuple[int, int, int] = (255, 220, 130),
        bead_radius: float = 0.03,
    ) -> None:
        self._server = server
        self._preview = preview
        self._root_name = root_name
        self._bead_color = bead_color
        self._line_color = line_color
        self._version = _next_version()
        self._world_offset = np.asarray(world_offset, dtype=np.float32).reshape(3)

        self._use_source_topology = False
        st = getattr(preview, "source_transforms", None)
        ss = getattr(preview, "source_seg_src", None)
        sd = getattr(preview, "source_seg_dst", None)
        if st is not None and ss is not None and sd is not None:
            st_a = np.asarray(st, dtype=np.float32)
            ss_a = np.asarray(ss, dtype=np.int32)
            sd_a = np.asarray(sd, dtype=np.int32)
            n_frames = int(preview.transforms.shape[0])
            if (
                ss_a.size > 0
                and st_a.ndim == 3
                and int(st_a.shape[0]) == n_frames
                and int(st_a.shape[1]) > 0
            ):
                mx = int(max(ss_a.max(), sd_a.max()))
                if mx < int(st_a.shape[1]):
                    self._use_source_topology = True
                    self._segment_src = ss_a
                    self._segment_dst = sd_a

        if not self._use_source_topology:
            names = preview.joint_names
            name_to_idx = {n: i for i, n in enumerate(names)}
            # Derive bone segments from the canonical parent map restricted to
            # joints the preview actually has.  When an intermediate joint is
            # missing (e.g. ``spine`` absent from the ik_map), walk up the
            # canonical chain until we find an ancestor that IS present so
            # ``chest`` connects directly to ``hips`` instead of leaving a gap.
            segs_src: list[int] = []
            segs_dst: list[int] = []
            for name, parent in _CANONICAL_PARENT.items():
                if parent is None or name not in name_to_idx:
                    continue
                ancestor = parent
                while ancestor is not None and ancestor not in name_to_idx:
                    ancestor = _CANONICAL_PARENT.get(ancestor)
                if ancestor is not None and ancestor in name_to_idx:
                    segs_src.append(name_to_idx[ancestor])
                    segs_dst.append(name_to_idx[name])
            self._segment_src = np.asarray(segs_src, dtype=np.int32)
            self._segment_dst = np.asarray(segs_dst, dtype=np.int32)

        self._num_frames = preview.num_frames
        # Prime with frame 0.
        positions = self._frame_positions(0)
        self._bead_rows = np.arange(int(positions.shape[0]), dtype=np.int32)
        sbidx = getattr(preview, "source_bead_indices", None)
        if sbidx is not None:
            br = np.asarray(sbidx, dtype=np.int32).reshape(-1)
            if br.size > 0:
                self._bead_rows = br

        # Bones: match SkeletonRenderer's tile pattern byte-for-byte so
        # viser's line_segments path picks up the colors uniformly.
        self._line_handle: Any = None
        if len(self._segment_src):
            lines = np.stack(
                [positions[self._segment_src], positions[self._segment_dst]],
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

        # Beads: one icosahedron template offset per joint, flattened into a
        # single mesh handle whose vertex buffer we overwrite on set_frame.
        #
        # Use ``add_mesh_simple`` (not ``add_mesh_trimesh``) because only the
        # former exposes a writeable ``.vertices`` property that actually
        # flushes to the client — the trimesh variant treats the mesh as
        # immutable, which caused the beads to freeze in place even though
        # the bone lines animated correctly.
        sphere_v, sphere_f = _unit_icosphere()
        sphere_v = sphere_v * float(bead_radius)
        nverts_per = sphere_v.shape[0]
        nfaces_per = sphere_f.shape[0]
        nbeads = int(self._bead_rows.shape[0])
        self._bead_nverts_per = nverts_per
        self._bead_verts_template = sphere_v  # (V, 3)
        self._bead_handle: Any = None
        if nbeads > 0:
            all_faces = np.empty((nbeads * nfaces_per, 3), dtype=np.int32)
            for i in range(nbeads):
                all_faces[i * nfaces_per:(i + 1) * nfaces_per] = sphere_f + i * nverts_per
            self._bead_faces = all_faces
            bead_pos = positions[self._bead_rows]
            all_verts = self._compose_beads(bead_pos)
            self._bead_handle = server.scene.add_mesh_simple(
                name=f"{root_name}/joints_v{self._version}",
                vertices=all_verts.astype(np.float32),
                faces=all_faces,
                color=bead_color,
                flat_shading=True,
                side="double",
            )
        else:
            self._bead_faces = np.zeros((0, 3), dtype=np.int32)

    # ---- public API ---------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def set_frame(self, frame: int) -> None:
        """Redraw bones + beads at ``frame`` (0-indexed, clamped)."""
        if self._num_frames <= 0:
            return
        f = int(np.clip(frame, 0, self._num_frames - 1))
        positions = self._frame_positions(f)
        if self._line_handle is not None and len(self._segment_src):
            lines = np.stack(
                [positions[self._segment_src], positions[self._segment_dst]],
                axis=1,
            ).astype(np.float32)
            try:
                self._line_handle.points = lines
            except Exception:
                # One-off client hiccup shouldn't freeze playback; the next
                # frame will try again.
                pass
        if self._bead_handle is not None:
            verts = self._compose_beads(positions[self._bead_rows])
            try:
                self._bead_handle.vertices = verts
            except Exception:
                pass

    def clear(self) -> None:
        """Remove every handle this renderer owns.  Idempotent."""
        for h_attr in ("_line_handle", "_bead_handle"):
            h = getattr(self, h_attr, None)
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
            setattr(self, h_attr, None)

    # ---- internals ----------------------------------------------------------

    def _frame_positions(self, frame: int) -> NDArray:
        """``(M, 3)`` joint positions for a single frame with world offset.

        The upstream tensor uses NaN to flag joints the scaler didn't populate
        — for viz we zero those out so the mesh buffer stays flat; the bone
        list simply doesn't reference them (skipped at construction time).
        World offset is added last so it affects both beads and bones.
        """
        if getattr(self, "_use_source_topology", False):
            arr = self._preview.source_transforms[frame, :, 0:3]
        else:
            arr = self._preview.transforms[frame, :, 0:3]
        out = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
        out = out + self._world_offset[None, :]
        return out

    def _compose_beads(self, positions: NDArray) -> NDArray:
        """Build the merged-mesh vertex buffer: template sphere + joint offset."""
        v = self._bead_verts_template  # (V, 3)
        # Broadcast: (M, 1, 3) + (V, 3) → (M, V, 3), then flatten.
        return (positions[:, None, :] + v[None, :, :]).reshape(-1, 3).astype(np.float32)


__all__ = ["ScaledSkeletonRenderer"]
