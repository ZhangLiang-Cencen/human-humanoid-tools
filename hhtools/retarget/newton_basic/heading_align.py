# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Apply ``conj(source_body_quat)`` to stacked effector transforms (preview / UI)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q


def align_effector_tensor_to_source_heading(
    transforms: NDArray[np.floating],
    *,
    source_body_quat: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Counter-rotate every finite ``(pos, quat)`` row to match source motion heading.

    ``transforms`` has shape ``(F, M, 7)`` with ``(x,y,z, qx,qy,qz,qw)`` per joint.
    Returns a float32 copy with the same layout.
    """

    out = np.asarray(transforms, dtype=np.float32, order="C").copy()
    sbq = np.asarray(source_body_quat, dtype=np.float32).reshape(4)
    if np.allclose(sbq, [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        return out
    inv_q = Q.conjugate(sbq[None, :])[0]
    F, M = out.shape[:2]
    n = F * M
    q_bc = np.broadcast_to(inv_q[None, :], (n, 4))
    pos = out[:, :, :3].reshape(n, 3)
    quat = out[:, :, 3:7].reshape(n, 4)
    valid = np.isfinite(pos).all(axis=1)
    if valid.any():
        pos[valid] = Q.rotate(q_bc[valid], pos[valid]).astype(np.float32)
        quat[valid] = Q.multiply(q_bc[valid], quat[valid]).astype(np.float32)
    out[:, :, :3] = pos.reshape(F, M, 3)
    out[:, :, 3:7] = quat.reshape(F, M, 4)
    return out


__all__ = ["align_effector_tensor_to_source_heading"]
