# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""GPU batch-size guards for Newton multi-env IK.

Newton's tiled LM solver allocates one ``DOF × DOF`` Cholesky tile *per IK
problem*, where ``DOF`` is the **single-robot** dof count.  When
:meth:`NewtonBasicPipeline.run_batch` passes a single-robot model with
``n_problems=N`` (matching soma-retargeter), per-block shared memory is sized
by one robot's ``DOF`` and is **independent of N** — so N is bounded only by a
soft cap (grid / device memory), not by shared memory.

The historical failure::

    Failed to compile LTO 'potrf_300_300_...'
    Estimated shared memory requirement is 720000B, but the device-reported
    limit is 101376B.

came from feeding an *N-robot merged* model to the solver, which made each tile
``(N · dof)²`` (e.g. ``300 = N · 35``).  That bug is fixed; this module now only
guards the (rare) case where even a *single* robot's tile would not fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hhtools.robot.loader import URDFRobotModel

_log = logging.getLogger(__name__)

# Newton ``wp.tile_cholesky`` scratch ≈ DOF² × 8 bytes (float32 tiles).
_CHOLESKY_BYTES_PER_DOF_SQ = 8

# Soft upper bound on parallel IK problems.  Shared memory no longer scales
# with N (one tile per problem, sized by single-robot dof), so this cap exists
# only to keep host-side pre/post-processing and device memory reasonable.
# soma-retargeter ships ``batch_size=100``; we leave headroom above that.
_MAX_BATCH_ENVS = 256


def ik_cholesky_smem_bytes(total_dof: int) -> int:
    """Estimated shared-memory bytes for one tiled Cholesky factorization."""
    dof = max(0, int(total_dof))
    return dof * dof * _CHOLESKY_BYTES_PER_DOF_SQ


def is_ik_shared_memory_error(err: BaseException) -> bool:
    """True when *err* looks like a Newton/Warp IK shared-memory compile failure."""
    msg = str(err).lower()
    needles = (
        "shared memory requirement",
        "device-reported limit",
        "tile_cholesky",
        "potrf_",
        "tile size(s) may be too large",
    )
    return any(n in msg for n in needles)


def _device_shared_memory_limit() -> int:
    try:
        import warp as wp

        wp.init()
        dev = wp.get_device()
        if not dev.is_cuda:
            return 0
        return int(dev.max_shared_memory_per_block or 0)
    except Exception:  # noqa: BLE001
        return 0


def max_gpu_batch_envs(
    joint_dof_per_env: int,
    *,
    device_smem_limit: int | None = None,
    headroom: float = 0.8,
) -> int:
    """Upper bound on safe ``num_envs`` for :meth:`NewtonBasicPipeline.run_batch`.

    Each parallel IK problem factorises its own ``dof_per_env × dof_per_env``
    Cholesky tile, so per-block shared memory depends only on the single-robot
    dof — **not** on the batch size.  The result is therefore a soft cap
    (:data:`_MAX_BATCH_ENVS`) as long as one robot's tile fits in shared
    memory; only a robot whose single tile already overflows is clamped to 1.

    Returns ``1`` on CPU or when the device limit is unknown (conservative).
    """
    per_env = max(1, int(joint_dof_per_env))
    limit = device_smem_limit if device_smem_limit is not None else _device_shared_memory_limit()
    if limit <= 0:
        return 1

    budget = int(limit * headroom)
    # A single robot's Cholesky tile must fit in one block's shared memory.
    if ik_cholesky_smem_bytes(per_env) > budget:
        return 1
    # Shared memory does not grow with N; bound only by the soft cap.
    return _MAX_BATCH_ENVS


def clamp_gpu_batch_size(robot: URDFRobotModel, requested: int) -> int:
    """Clamp Web/CLI batch size to what the current Warp device can compile."""
    requested = max(1, min(_MAX_BATCH_ENVS, int(requested)))
    try:
        from hhtools.retarget.newton_basic.robot_model import build_newton_model

        ctx = build_newton_model(robot, num_envs=1)
        per_env = int(ctx.joint_dof_count or ctx.model.joint_dof_count)
        safe = max_gpu_batch_envs(per_env)
        if safe < requested:
            _log.info(
                "Clamping GPU batch_size %d → %d (robot dof/env=%d, smem limit=%s)",
                requested,
                safe,
                per_env,
                _device_shared_memory_limit(),
            )
        return min(requested, safe)
    except Exception as err:  # noqa: BLE001 — never block batch on probe failure
        _log.debug("batch_size clamp probe failed: %s", err)
        return requested


def shared_memory_error_hint(joint_dof_per_env: int | None = None) -> str:
    """User-facing hint appended to batch failure reasons."""
    extra = ""
    if joint_dof_per_env:
        extra = f"（本机器人约 {joint_dof_per_env} 自由度/条）"
    return (
        "单个机器人的 Newton IK 内核就超出了 CUDA 共享内存（per-block SMEM，约 99KB）"
        f"上限{extra}，与 nvidia-smi 里的全局显存无关。"
        "批量模式每条子问题独立分配 dof×dof 的 Cholesky tile（与批量大小无关），"
        "因此这通常意味着该机器人自由度过高或标定异常。"
        "服务会自动逐条回退重试；若仍失败请检查标定与机器人配置。"
    )
