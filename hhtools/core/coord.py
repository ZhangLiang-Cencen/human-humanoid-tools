"""Coordinate-system utilities for the internal motion representation.

The project's canonical coordinate system is **Z-up, right-handed, +X forward**, matching
common robotics conventions (MuJoCo, Isaac Lab, Newton). Different motion capture sources
and SMPL-based datasets use Y-up by convention; this module provides a single conversion
primitive so all downstream code (viewer, retargeting, analytics) can assume Z-up.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q
from hhtools.core.motion import Motion, UpAxis

# Rotation that maps Y-up to Z-up: rotate +90 degrees about the +X axis.
# Under this rotation: (x, y, z) -> (x, -z, y). In xyzw quaternion form:
# q = [sin(45 deg), 0, 0, cos(45 deg)].
_Y_TO_Z_QUAT = np.array(
    [np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0)], dtype=np.float32
)
# Rotation matrix equivalent (pre-multiplied onto column vectors).
_Y_TO_Z_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def rotate_y_up_to_z_up_positions(positions: NDArray) -> NDArray:
    """Rotate a ``(..., 3)`` position array from Y-up to Z-up."""
    p = np.asarray(positions, dtype=np.float32)
    return p @ _Y_TO_Z_MATRIX.T


def rotate_y_up_to_z_up_quaternions(quats: NDArray) -> NDArray:
    """Left-multiply each xyzw quaternion by the Y->Z rotation so world-frame rotations align."""
    q = np.asarray(quats, dtype=np.float32)
    rot = np.broadcast_to(_Y_TO_Z_QUAT, q.shape).copy()
    return Q.normalize(Q.multiply(rot, q)).astype(np.float32)


def to_up_axis(motion: Motion, target: UpAxis = "Z") -> Motion:
    """Return a new :class:`Motion` rotated so its up-axis matches ``target``.

    Only the Y <-> Z case is currently handled (X-up is extremely uncommon for body motion
    data). The conversion is applied as a rigid rotation to every frame, every bone, and
    also to any attached :class:`SceneObject` trajectories so human + prop stay in sync.
    """
    if motion.up_axis == target:
        return motion
    if motion.up_axis == "Y" and target == "Z":
        positions = rotate_y_up_to_z_up_positions(motion.positions)
        quats = rotate_y_up_to_z_up_quaternions(motion.quaternions)
        rot_mat = _Y_TO_Z_MATRIX
        rot_quat = _Y_TO_Z_QUAT
    elif motion.up_axis == "Z" and target == "Y":
        rot_mat = _Y_TO_Z_MATRIX.T
        positions = motion.positions @ rot_mat.T
        rot_quat = np.array([-_Y_TO_Z_QUAT[0], 0.0, 0.0, _Y_TO_Z_QUAT[3]], dtype=np.float32)
        rot = np.broadcast_to(rot_quat, motion.quaternions.shape).copy()
        quats = Q.normalize(Q.multiply(rot, motion.quaternions)).astype(np.float32)
    else:
        raise NotImplementedError(
            f"Up-axis conversion {motion.up_axis!r} -> {target!r} is not implemented"
        )

    new_objects = []
    for obj in motion.objects:
        from hhtools.core.scene import SceneObject  # local import to avoid cycle on first import
        new_pos = (obj.positions @ rot_mat.T).astype(np.float32)
        rot_b = np.broadcast_to(rot_quat, obj.quaternions.shape).copy()
        new_q = Q.normalize(Q.multiply(rot_b, obj.quaternions)).astype(np.float32)
        new_objects.append(
            SceneObject(
                name=obj.name,
                positions=new_pos,
                quaternions=new_q,
                extents=obj.extents,
                mesh_path=obj.mesh_path,
                scale=obj.scale,
            )
        )

    # Rotate attached mesh geometry so the body stays upright. Without this the
    # skeleton would flip to Z-up but a ``BakedMesh`` cache would still hold Y-up
    # vertices, visibly knocking the character onto its side.
    new_meta: dict = {**motion.meta, "up_axis_original": motion.up_axis}
    _rotate_mesh_meta(new_meta, rot_mat)
    # Heightfield terrain is axis-aligned by construction — a 90 degree
    # rotation would no longer be representable as a heightfield (a Z-up
    # ramp becomes a vertical wall under Y->Z).  In practice every dataset
    # carrying TerrainHeightfield ships in Z-up already, so we pass the
    # field through unchanged; if a future caller needs to flip up_axis
    # while terrain is present, it should pre-rotate the OBJ before
    # rasterisation.
    return Motion(
        name=motion.name,
        hierarchy=motion.hierarchy,
        positions=positions.astype(np.float32),
        quaternions=quats.astype(np.float32),
        framerate=motion.framerate,
        up_axis=target,
        source_format=motion.source_format,
        meta=new_meta,
        objects=new_objects,
        terrain=motion.terrain,
    )


def _rotate_mesh_meta(meta: dict, rot_mat: NDArray) -> None:
    """In-place rotate ``meta["baked_mesh"]`` vertices by ``rot_mat``.

    :class:`~hhtools.core.skinning.BakedMesh` stores pre-computed world-space vertex
    positions (one set per frame, from the SMPL forward pass), so they have to move
    along with the joints when we re-orient the up-axis — otherwise the character
    would keep its head pointing +Y while the skeleton rotated into Z-up.

    :class:`~hhtools.core.skinning.SkinnedMesh` is deliberately *not* rotated here.
    LBS evaluates ``v_world = G · B⁻¹ · v_rest`` per frame, and we already rotated
    every joint's global transform ``G`` above.  Since ``B⁻¹`` was authored against
    the untransformed rest frame and ``v_rest`` lives in that same untransformed
    frame, leaving both alone means ``v_world`` naturally comes out in the new
    frame: ``v_world_new = R · G · B⁻¹ · v_rest = G_new · B⁻¹ · v_rest``.
    """
    from hhtools.core.skinning import BakedMesh  # lazy, avoid import cycles

    baked = meta.get("baked_mesh")
    if isinstance(baked, BakedMesh):
        new_verts = (baked.vertices @ rot_mat.T).astype(np.float32)
        new_normals = None
        if baked.normals is not None:
            new_normals = (baked.normals @ rot_mat.T).astype(np.float32)
        meta["baked_mesh"] = BakedMesh(
            vertices=new_verts,
            triangles=baked.triangles,
            normals=new_normals,
        )


__all__ = [
    "rotate_y_up_to_z_up_positions",
    "rotate_y_up_to_z_up_quaternions",
    "to_up_axis",
]
