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


def quat_to_bvh_euler(
    quat: NDArray,
    order: str,
    *,
    degrees: bool = True,
) -> NDArray:
    """Convert an xyzw quaternion to BVH-style intrinsic Euler angles.

    Prefer :func:`quat_to_bvh_euler_for_write` when encoding BVH files that
    will be read back through :func:`bvh_euler_to_quat`.
    """
    if len(order) != 3:
        raise ValueError(f"Invalid BVH rotation order: {order!r}")

    from scipy.spatial.transform import Rotation as SciRot

    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    # SciPy expects scalar-last; hhtools uses xyzw.
    r = SciRot.from_quat(q)
    euler = r.as_euler(order.lower(), degrees=degrees)
    return euler.astype(np.float32)


def _quat_to_euler_zyx_legacy(quat: NDArray, *, degrees: bool) -> NDArray:
    """Intrinsic ZYX Euler matching :func:`bvh_euler_to_quat` (``Zrotation Yrotation Xrotation``)."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    if degrees:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)
    return np.stack([yaw, pitch, roll], axis=-1).astype(np.float32)


def quat_to_bvh_euler_for_write(
    quat: NDArray,
    order: str,
    *,
    degrees: bool = True,
) -> NDArray:
    """Encode a local quaternion for :func:`hhtools.io.bvh.save_bvh`.

    ``load_bvh`` decodes MOTION channels with :func:`bvh_euler_to_quat` (legacy
    axis composition).  SciPy ``as_euler`` is **not** inverse of that decoder
    for common orders such as ``ZYX``, which broke bundled ``*_zero_frame0.bvh``
    round-trips for LAFAN.  ``ZYX`` uses the closed-form legacy inverse; other
    orders fall back to a small least-squares fit against ``bvh_euler_to_quat``.
    """
    if len(order) != 3:
        raise ValueError(f"Invalid BVH rotation order: {order!r}")
    if order.upper() == "ZYX":
        return _quat_to_euler_zyx_legacy(quat, degrees=degrees)

    from scipy.optimize import least_squares

    q = np.asarray(quat, dtype=np.float64).reshape(4)
    x0 = quat_to_bvh_euler(q, order, degrees=degrees).reshape(3)

    def _residual(x: NDArray) -> NDArray:
        q2 = np.asarray(bvh_euler_to_quat(x, order, degrees=degrees), dtype=np.float64).reshape(4)
        d1 = q - q2
        d2 = q + q2
        return d1 if float(np.dot(d1, d1)) < float(np.dot(d2, d2)) else d2

    result = least_squares(_residual, x0, method="lm", max_nfev=100)
    return result.x.astype(np.float32)


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
