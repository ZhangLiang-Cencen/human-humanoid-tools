"""Render a skeleton as per-bone tube meshes (capsule-like), updated in-place each frame.

The renderer pre-computes a canonical cylinder (two rings of ``n_segments`` vertices) and a
canonical sphere (joints), then per frame places one instance of each primitive at every bone
by computing its world-space transform from the parent and child positions. All primitives are
merged into one big mesh handle to keep Viser traffic minimal (single ``vertices`` update per
frame rather than N add_mesh calls).

This gives the user a "pseudo body" visualisation even when no SMPL weights are available, so
BVH / GLB / FBX / CSV motions still render as a 3D volume rather than a stick figure. Per-bone
radii come from :mod:`hhtools.viewer.anatomy` so finger / toe / eye bones render as thin
capsules instead of giant blobs that hide their structure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hhtools.core.motion import Motion


def _unit_cylinder(segments: int) -> tuple[NDArray, NDArray]:
    """Return canonical cylinder vertices and triangle faces.

    The cylinder is axis-aligned: bottom ring at ``z=0`` (radius 1), top ring at ``z=1``
    (radius 1). Side faces only (no caps).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    bottom = ring.copy()
    top = ring.copy()
    top[:, 2] = 1.0
    vertices = np.concatenate([bottom, top], axis=0).astype(np.float32)  # (2S, 3)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, i + segments])
        faces.append([j, j + segments, i + segments])
    return vertices, np.asarray(faces, dtype=np.int32)


def _unit_icosphere() -> tuple[NDArray, NDArray]:
    """A tiny icosahedron (12 verts, 20 tris) with unit radius; used as a joint bead."""
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


def _orthonormal_basis(direction: NDArray) -> tuple[NDArray, NDArray]:
    """Return two unit vectors perpendicular to ``direction`` (and to each other)."""
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ref = np.where(
        np.abs(direction[..., 0:1]) < 0.9,
        fallback,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    ref = np.broadcast_to(ref, direction.shape).copy()
    right = np.cross(direction, ref)
    rnorm = np.linalg.norm(right, axis=-1, keepdims=True)
    rnorm = np.where(rnorm < 1e-8, 1.0, rnorm)
    right = right / rnorm
    up = np.cross(direction, right)
    return right, up


class CapsuleMeshRenderer:
    """Fast per-bone tube + joint-bead renderer that updates a single Viser mesh handle."""

    def __init__(
        self,
        server,  # type: ignore[no-untyped-def]
        root_name: str = "/capsules",
        cylinder_segments: int = 10,
        color: tuple[int, int, int] = (247, 164, 112),
        default_bone_radius: float = 0.035,
        default_joint_radius: float = 0.045,
    ) -> None:
        self._server = server
        self._root = root_name
        self._default_bone_radius = float(default_bone_radius)
        self._default_joint_radius = float(default_joint_radius)
        self._color = color
        self._cyl_verts, self._cyl_faces = _unit_cylinder(cylinder_segments)
        self._sph_verts, self._sph_faces = _unit_icosphere()
        self._handle = None
        self._visible = True
        self._motion: Motion | None = None
        self._parent_idx: NDArray | None = None
        self._child_idx: NDArray | None = None
        self._bone_radii: NDArray | None = None  # (num_edges,)
        self._joint_radii: NDArray | None = None  # (num_visible_joints,)
        self._joint_indices: NDArray | None = None  # which joint index each bead corresponds to
        self._faces_cache: NDArray | None = None
        # Monotonic version tag appended to every handle name. Ensures that when set_motion
        # rebuilds the mesh for a new clip, the new handle does NOT collide with the previous
        # handle's name on the Viser client (removal is async, so for a brief window both
        # exist — same-named handles would then cause "wrong links" where Viser applies the
        # new vertex buffer to the old face topology).
        self._version = 0

    def clear(self) -> None:
        if self._handle is not None:
            try:
                self._handle.visible = False
            except Exception:
                pass
            try:
                self._handle.remove()
            except Exception:
                pass
            self._handle = None
        self._motion = None

    def set_visible(self, visible: bool) -> None:
        """Flip the mesh visibility without destroying buffers (non-blocking for playback)."""
        self._visible = bool(visible)
        if self._handle is not None:
            try:
                self._handle.visible = self._visible
            except Exception:
                pass

    def set_motion(
        self,
        motion: Motion,
        *,
        exclude_bones: set[int] | None = None,
        bone_radii: NDArray | None = None,
    ) -> None:
        """Switch to a new motion.

        ``exclude_bones`` hides specific bones (joint bead) and any edges incident to them.
        ``bone_radii`` is a per-joint float array of length ``num_bones``; the renderer uses
        the radius of the child bone for each edge and the same array for the joint beads.
        """
        self.clear()
        parents = motion.hierarchy.parent_indices
        num_bones = int(motion.num_bones)

        excluded = np.zeros(num_bones, dtype=bool)
        if exclude_bones:
            excluded[list(exclude_bones)] = True

        # Edge mask: keep edge i if it has a parent AND neither its child nor its parent is excluded.
        keep_edge = np.array(
            [
                parents[i] >= 0 and not excluded[i] and not excluded[int(parents[i])]
                for i in range(num_bones)
            ],
            dtype=bool,
        )
        self._parent_idx = parents[keep_edge].astype(np.int32)
        self._child_idx = np.where(keep_edge)[0].astype(np.int32)

        # Radii
        if bone_radii is not None:
            radii_full = np.asarray(bone_radii, dtype=np.float32).reshape(-1)
            assert radii_full.shape[0] == num_bones, "bone_radii must have one value per bone"
        else:
            radii_full = np.full((num_bones,), self._default_bone_radius, dtype=np.float32)

        self._bone_radii = radii_full[self._child_idx].copy()
        keep_joint = ~excluded
        self._joint_indices = np.where(keep_joint)[0].astype(np.int32)
        self._joint_radii = radii_full[self._joint_indices].copy() * 1.15  # beads slightly larger

        self._motion = motion
        self._faces_cache = self._build_faces_once()
        self._version += 1
        # Caller must call ``set_frame`` next — keeps playback cursor continuous across toggles.

    def set_frame(self, frame: int) -> None:
        motion = self._motion
        if motion is None or self._faces_cache is None:
            return
        frame = int(np.clip(frame, 0, motion.num_frames - 1))
        positions = motion.positions[frame]
        vertices = self._build_vertices(positions)
        if vertices.shape[0] == 0:
            return
        if self._handle is None:
            self._handle = self._server.scene.add_mesh_simple(
                name=f"{self._root}/mesh_v{self._version}",
                vertices=vertices,
                faces=self._faces_cache,
                color=self._color,
                flat_shading=False,
                side="double",
            )
            self._handle.visible = self._visible
        else:
            self._handle.vertices = vertices

    def _build_vertices(self, positions: NDArray) -> NDArray:
        bone_verts = self._bone_tube_vertices(positions)
        joint_verts = self._joint_bead_vertices(positions)
        return np.concatenate([bone_verts, joint_verts], axis=0).astype(np.float32)

    def _bone_tube_vertices(self, positions: NDArray) -> NDArray:
        if self._parent_idx is None or self._child_idx is None:
            return np.zeros((0, 3), dtype=np.float32)
        if self._parent_idx.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        n = positions.shape[0]
        pidx, cidx = self._parent_idx, self._child_idx
        if pidx.size > 0 and (int(pidx.max()) >= n or int(cidx.max()) >= n):
            mask = (pidx < n) & (cidx < n)
            pidx, cidx = pidx[mask], cidx[mask]
            if pidx.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
        starts = positions[pidx]  # (B, 3)
        ends = positions[cidx]  # (B, 3)
        vec = ends - starts
        length = np.linalg.norm(vec, axis=-1, keepdims=True)  # (B, 1)
        length_safe = np.where(length < 1e-6, 1.0, length)
        dir_ = vec / length_safe
        right, up = _orthonormal_basis(dir_)
        radii = self._bone_radii[:, None, None] if self._bone_radii is not None else 0.04  # (B,1,1)
        base = self._cyl_verts  # (2S, 3)
        length_b = length[:, None, :]
        pos = (
            starts[:, None, :]
            + right[:, None, :] * (base[None, :, 0:1] * radii)
            + up[:, None, :] * (base[None, :, 1:2] * radii)
            + dir_[:, None, :] * (base[None, :, 2:3] * length_b)
        )
        return pos.reshape(-1, 3)

    def _joint_bead_vertices(self, positions: NDArray) -> NDArray:
        if self._joint_indices is None or self._joint_radii is None:
            return np.zeros((0, 3), dtype=np.float32)
        idx = self._joint_indices
        n = positions.shape[0]
        if idx.size > 0 and int(idx.max()) >= n:
            mask = idx < n
            idx = idx[mask]
            if idx.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
            radii_local = self._joint_radii[mask]
        else:
            radii_local = self._joint_radii
        base = self._sph_verts  # (K, 3)
        centers = positions[idx]  # (Jk, 3)
        radii = self._joint_radii[:, None, None]  # (Jk, 1, 1)
        pos = centers[:, None, :] + base[None, :, :] * radii
        return pos.reshape(-1, 3)

    def _build_faces_once(self) -> NDArray:
        if self._parent_idx is None or self._child_idx is None or self._joint_indices is None:
            return np.zeros((0, 3), dtype=np.int32)
        num_bones = int(self._parent_idx.size)
        num_joints = int(self._joint_indices.size)
        verts_per_bone = int(self._cyl_verts.shape[0])
        verts_per_joint = int(self._sph_verts.shape[0])
        total_bone_verts = num_bones * verts_per_bone

        bone_faces = (
            np.concatenate(
                [self._cyl_faces + i * verts_per_bone for i in range(num_bones)], axis=0
            )
            if num_bones > 0
            else np.zeros((0, 3), dtype=np.int32)
        )
        joint_faces = (
            np.concatenate(
                [
                    self._sph_faces + total_bone_verts + j * verts_per_joint
                    for j in range(num_joints)
                ],
                axis=0,
            )
            if num_joints > 0
            else np.zeros((0, 3), dtype=np.int32)
        )
        return np.concatenate([bone_faces, joint_faces], axis=0).astype(np.int32)


__all__ = ["CapsuleMeshRenderer"]
