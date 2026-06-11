# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Newton batch-size guard: per-problem Cholesky tile is sized by single-robot dof.

These exercise the corrected shared-memory model where ``run_batch`` feeds a
single-robot model with ``n_problems=N`` (soma-retargeter layout).  The
per-block shared-memory budget then depends only on one robot's dof and is
**independent of N**, so the safe batch size is a soft cap rather than the old
``(N · dof)²`` clamp that collapsed batching to ~1.
"""

from __future__ import annotations

from hhtools.retarget.newton_basic.batch_limits import (
    _MAX_BATCH_ENVS,
    ik_cholesky_smem_bytes,
    is_ik_shared_memory_error,
    max_gpu_batch_envs,
)

# A typical consumer GPU per-block shared-memory limit (bytes).
_CONSUMER_SMEM = 101376


def test_smem_bytes_scale_with_single_robot_dof() -> None:
    # 35-dof humanoid (G1 29-dof + 6-dof floating base) → 35² × 8.
    assert ik_cholesky_smem_bytes(35) == 35 * 35 * 8


def test_batch_envs_independent_of_n_for_normal_humanoid() -> None:
    # A 35-dof robot's single tile (9800 B) fits easily in 99 KB, so the
    # guard returns the soft cap — NOT a small (N·dof)²-derived number.
    assert max_gpu_batch_envs(35, device_smem_limit=_CONSUMER_SMEM) == _MAX_BATCH_ENVS


def test_batch_envs_cpu_or_unknown_is_conservative() -> None:
    assert max_gpu_batch_envs(35, device_smem_limit=0) == 1


def test_batch_envs_clamps_only_when_single_tile_overflows() -> None:
    # A robot so large that even one Cholesky tile overflows shared memory.
    # 200² × 8 = 320000 B > 0.8 × 101376 ≈ 81100 B → must clamp to 1.
    assert max_gpu_batch_envs(200, device_smem_limit=_CONSUMER_SMEM) == 1


def test_shared_memory_error_detection() -> None:
    err = RuntimeError(
        "Failed to compile LTO 'potrf_300_300_...': Estimated shared memory "
        "requirement is 720000B, but the device-reported limit is 101376B."
    )
    assert is_ik_shared_memory_error(err)
    assert not is_ik_shared_memory_error(ValueError("unrelated"))
