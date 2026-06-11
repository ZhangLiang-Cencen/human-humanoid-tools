"""Euler-angle helpers and up-axis utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q


def euler_xyz_to_quat(euler: NDArray, degrees: bool = False) -> NDArray:
    """Convert intrinsic Euler angles in XYZ order to an xyzw quaternion.

    ``euler`` has shape ``(..., 3)``. When ``degrees`` is ``True`` the angles are interpreted in
    degrees.
    """
    e = np.asarray(euler, dtype=np.float32)
    if degrees:
        e = np.deg2rad(e)
    cx = np.cos(e[..., 0] * 0.5)
    sx = np.sin(e[..., 0] * 0.5)
    cy = np.cos(e[..., 1] * 0.5)
    sy = np.sin(e[..., 1] * 0.5)
    cz = np.cos(e[..., 2] * 0.5)
    sz = np.sin(e[..., 2] * 0.5)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return np.stack([qx, qy, qz, qw], axis=-1).astype(np.float32)


def bvh_euler_to_quat(angles: NDArray, order: str, degrees: bool = True) -> NDArray:
    """Convert BVH-style Euler angles to a quaternion.

    BVH rotation orders are usually given as three uppercase characters (e.g. ``"ZYX"``) and
    applied as intrinsic rotations in that sequence. We compose axis quaternions in the same
    order.
    """
    if len(order) != 3:
        raise ValueError(f"Invalid BVH rotation order: {order!r}")

    e = np.asarray(angles, dtype=np.float32)
    if degrees:
        e = np.deg2rad(e)

    axis_map = {"X": 0, "Y": 1, "Z": 2}
    quats = []
    for i, axis_char in enumerate(order):
        axis_idx = axis_map[axis_char.upper()]
        angle = e[..., i]
        half = 0.5 * angle
        s = np.sin(half)
        c = np.cos(half)
        q = np.zeros((*e.shape[:-1], 4), dtype=np.float32)
        q[..., axis_idx] = s
        q[..., 3] = c
        quats.append(q)

    out = quats[0]
    for q in quats[1:]:
        out = Q.multiply(out, q)
    return Q.normalize(out)


def up_axis_rotation(src_up: str, dst_up: str) -> NDArray:
    """Return a 3x3 rotation matrix that maps ``src_up`` to ``dst_up``.

    Inputs are single characters in ``{"X", "Y", "Z"}`` with the convention that the axis points
    in the positive direction.
    """
    src_up = src_up.upper()
    dst_up = dst_up.upper()
    if src_up == dst_up:
        return np.eye(3, dtype=np.float32)

    axis_map = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    a = axis_map[src_up]
    b = axis_map[dst_up]
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-8:
        # parallel (same or opposite). Handle the opposite case by flipping one axis.
        if c > 0:
            return np.eye(3, dtype=np.float32)
        # 180-degree rotation around any axis orthogonal to both
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, ortho)
        v = v / np.linalg.norm(v)
        vx, vy, vz = v
        kmat = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=np.float32)
        return np.eye(3, dtype=np.float32) + 2 * kmat @ kmat

    vx, vy, vz = v
    kmat = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=np.float32)
    return (np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))).astype(np.float32)
