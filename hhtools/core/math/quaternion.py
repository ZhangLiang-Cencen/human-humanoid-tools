"""Quaternion helpers (xyzw convention, batched numpy).

Conventions:

- Storage order is ``[x, y, z, w]`` (xyzw). This matches ``scipy.spatial.transform.Rotation``
  and the soma-retargeter CSV output.
- Rotations act on column vectors: ``v' = q * v`` with the quaternion Hamilton product.
- Identity quaternion is ``[0, 0, 0, 1]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-8


def identity(*leading: int, dtype: np.dtype | type = np.float32) -> NDArray:
    """Batched identity quaternion with shape ``(*leading, 4)``."""
    q = np.zeros((*leading, 4), dtype=dtype)
    q[..., 3] = 1.0
    return q


def normalize(q: NDArray, eps: float = EPS) -> NDArray:
    """Normalise to unit quaternion; zero-length inputs become identity."""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    bad = n < eps
    n = np.where(bad, 1.0, n)
    out = q / n
    if np.any(bad):
        out = np.where(bad, np.array([0.0, 0.0, 0.0, 1.0], dtype=out.dtype), out)
    return out


def conjugate(q: NDArray) -> NDArray:
    """Quaternion conjugate (inverse for unit quaternions)."""
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def multiply(a: NDArray, b: NDArray) -> NDArray:
    """Hamilton product ``a * b`` (xyzw)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)


def rotate(q: NDArray, v: NDArray) -> NDArray:
    """Rotate 3-vector(s) ``v`` by quaternion(s) ``q`` (xyzw)."""
    u = q[..., :3]
    w = q[..., 3:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def from_matrix(mat: NDArray) -> NDArray:
    """Convert a rotation matrix ``(..., 3, 3)`` to an xyzw quaternion ``(..., 4)``.

    Uses the shepperd's method with branching on the largest trace / diagonal entry for numerical
    stability. Input matrices should already be orthonormal.
    """
    m = np.asarray(mat, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.empty((m.shape[0], 4), dtype=np.float64)

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    case_trace = trace > 0.0

    # Case 1: trace positive
    if np.any(case_trace):
        t = trace[case_trace] + 1.0
        s = 0.5 / np.sqrt(t)
        q[case_trace, 3] = 0.25 / s  # w
        q[case_trace, 0] = (m[case_trace, 2, 1] - m[case_trace, 1, 2]) * s
        q[case_trace, 1] = (m[case_trace, 0, 2] - m[case_trace, 2, 0]) * s
        q[case_trace, 2] = (m[case_trace, 1, 0] - m[case_trace, 0, 1]) * s

    remaining = ~case_trace
    if np.any(remaining):
        mr = m[remaining]
        # Pick the largest diagonal
        dxx = mr[:, 0, 0]
        dyy = mr[:, 1, 1]
        dzz = mr[:, 2, 2]
        idx = np.argmax(np.stack([dxx, dyy, dzz], axis=-1), axis=-1)

        out = np.empty((mr.shape[0], 4), dtype=np.float64)
        for k in range(3):
            sel = idx == k
            if not np.any(sel):
                continue
            mk = mr[sel]
            if k == 0:
                t = 1.0 + mk[:, 0, 0] - mk[:, 1, 1] - mk[:, 2, 2]
                s = 2.0 * np.sqrt(t)
                out[sel, 0] = 0.25 * s
                out[sel, 1] = (mk[:, 0, 1] + mk[:, 1, 0]) / s
                out[sel, 2] = (mk[:, 0, 2] + mk[:, 2, 0]) / s
                out[sel, 3] = (mk[:, 2, 1] - mk[:, 1, 2]) / s
            elif k == 1:
                t = 1.0 - mk[:, 0, 0] + mk[:, 1, 1] - mk[:, 2, 2]
                s = 2.0 * np.sqrt(t)
                out[sel, 0] = (mk[:, 0, 1] + mk[:, 1, 0]) / s
                out[sel, 1] = 0.25 * s
                out[sel, 2] = (mk[:, 1, 2] + mk[:, 2, 1]) / s
                out[sel, 3] = (mk[:, 0, 2] - mk[:, 2, 0]) / s
            else:  # k == 2
                t = 1.0 - mk[:, 0, 0] - mk[:, 1, 1] + mk[:, 2, 2]
                s = 2.0 * np.sqrt(t)
                out[sel, 0] = (mk[:, 0, 2] + mk[:, 2, 0]) / s
                out[sel, 1] = (mk[:, 1, 2] + mk[:, 2, 1]) / s
                out[sel, 2] = 0.25 * s
                out[sel, 3] = (mk[:, 1, 0] - mk[:, 0, 1]) / s
        q[remaining] = out

    q = q.reshape((*shape, 4))
    return normalize(q).astype(np.float32)


def to_matrix(q: NDArray) -> NDArray:
    """Convert an xyzw quaternion ``(..., 4)`` to a rotation matrix ``(..., 3, 3)``."""
    q = normalize(np.asarray(q))
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    m = np.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    )
    return m.reshape((*q.shape[:-1], 3, 3)).astype(np.float32)


def from_axis_angle(axis_angle: NDArray) -> NDArray:
    """Convert axis-angle ``(..., 3)`` to xyzw quaternion.

    The magnitude of ``axis_angle`` encodes the rotation angle in radians; the direction is the
    rotation axis. Zero vectors map to the identity quaternion.
    """
    aa = np.asarray(axis_angle, dtype=np.float32)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = angle < EPS
    safe_angle = np.where(small, 1.0, angle)
    axis = aa / safe_angle
    half = 0.5 * angle
    sin_half = np.sin(half)
    cos_half = np.cos(half)
    xyz = axis * sin_half
    q = np.concatenate([xyz, cos_half], axis=-1)
    q = np.where(small, np.array([0.0, 0.0, 0.0, 1.0], dtype=q.dtype), q)
    return q.astype(np.float32)


def to_axis_angle(q: NDArray) -> NDArray:
    """Convert xyzw quaternion ``(..., 4)`` to axis-angle ``(..., 3)``."""
    q = normalize(np.asarray(q, dtype=np.float32))
    w = np.clip(q[..., 3:4], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    safe = sin_half < EPS
    sin_half = np.where(safe, 1.0, sin_half)
    axis = q[..., :3] / sin_half
    out = axis * angle
    return np.where(safe, np.zeros_like(out), out).astype(np.float32)


def slerp(q0: NDArray, q1: NDArray, t: NDArray | float) -> NDArray:
    """Spherical linear interpolation between two unit quaternions.

    ``q0`` and ``q1`` have shape ``(..., 4)``; ``t`` broadcasts against ``(...,)``.
    """
    q0 = normalize(np.asarray(q0, dtype=np.float32))
    q1 = normalize(np.asarray(q1, dtype=np.float32))
    t = np.asarray(t, dtype=np.float32)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    # Take the shorter path
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    # When the angle is very small fall back to linear interpolation (plus renormalisation)
    close = dot > 1.0 - 1e-6
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)

    t = t[..., None] if t.ndim < q0.ndim else t
    theta = theta_0 * t

    sin_theta = np.sin(theta)
    s0 = np.where(close, 1.0 - t, np.cos(theta) - dot * sin_theta / np.maximum(sin_theta_0, EPS))
    s1 = np.where(close, t, sin_theta / np.maximum(sin_theta_0, EPS))
    out = s0 * q0 + s1 * q1
    return normalize(out)


def ensure_continuous(q: NDArray) -> NDArray:
    """Flip quaternion signs so consecutive frames stay in the same hemisphere.

    Quaternions ``q`` and ``-q`` represent the same rotation, but sign
    discontinuities break SLERP interpolation, IK target tracking, and
    direction-offset computations that depend on continuous quaternion
    trajectories.  This function walks the leading axis (assumed to be
    the frame/time axis) and negates any frame whose 4-D dot product
    with its predecessor is negative.

    Args:
        q: ``(F, ..., 4)`` quaternion array where the first axis is time.

    Returns:
        A copy with per-frame signs adjusted for continuity.
    """
    out = np.array(q, dtype=q.dtype, copy=True)
    flat = out.reshape(out.shape[0], -1, 4)
    for f in range(1, flat.shape[0]):
        dots = np.sum(flat[f] * flat[f - 1], axis=-1)
        flip = dots < 0.0
        flat[f][flip] *= -1.0
    return out


def delta(q_from: NDArray, q_to: NDArray) -> NDArray:
    """Quaternion delta ``q_to * inv(q_from)`` as an xyzw quaternion."""
    return multiply(q_to, conjugate(q_from))
