"""Pre-IK effector-target constraints (pure NumPy, CPU).

Upstream soma-retargeter splits its "feet stabilization" concern across two
modules:

* ``soma_retargeter.robotics.human_to_robot_scaler`` owns the *pre-IK*
  constraint chain (``_enforce_ground_contact``, ``_enforce_foot_planting``,
  ``_enforce_min_lateral_separation``, ``_smooth_corrections`` …) that fixes
  effector targets *before* they're fed to IK.
* ``soma_retargeter.pipelines.feet_stabilizer`` runs a *post-IK* Warp
  two-bone-IK solve on the robot rig to snap ankles to those targets.

Stage-1 of the hhtools port only ships the *pre-IK* half here — purely numpy,
no Newton/Warp required.  The two-bone-IK solve will land in a follow-up
stage that introduces the actual IK solver.  Splitting these responsibilities
keeps the constraint logic trivially unit-testable (synthetic effector
trajectories in, constrained trajectories out) while we wait on the solver.

Attribution:
  Portions of the constraint formulas are adapted from soma-retargeter
  (Apache-2.0).
  https://github.com/NVlabs/SOMA-Retargeter
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.retarget.newton_basic.config import FeetStabilizerConfig


__all__ = ["FeetStabilizer", "StabilizationStats"]


_AXIS_TO_IDX = {"X": 0, "Y": 1, "Z": 2}


@dataclass(frozen=True)
class StabilizationStats:
    """Diagnostics so callers can surface per-clip tuning signal.

    Attributes:
        ground_corrections: ``(F,)`` float array — per-frame metres the
            effector block was lifted or pushed to satisfy ``ground_contact_z``.
        planted_frames: ``(num_foot_pairs, F)`` bool array — ``True`` on
            frames that were detected as "foot planted" and locked.
        smoothed_frames: Count of frames that had at least one correction
            clamped by the smoothing rate limit.
    """

    ground_corrections: NDArray
    planted_frames: NDArray
    smoothed_frames: int


class FeetStabilizer:
    """Apply foot-planting / ground-contact / lateral-separation / smoothing
    constraints to an effector trajectory.

    The input/output layout matches :class:`~hhtools.retarget.newton_basic
    .scaler.ScaledEffectors.transforms`: ``(F, M, 7)`` with
    ``(x, y, z, qx, qy, qz, qw)`` per mapped joint.  Quaternions are passed
    through untouched (we only constrain positional targets at this stage).
    """

    def __init__(
        self,
        config: FeetStabilizerConfig,
        *,
        joint_names: tuple[str, ...],
    ) -> None:
        self._config = config
        self._joint_names = tuple(joint_names)
        self._up = _AXIS_TO_IDX[config.up_axis]
        # Lateral / forward axes are whatever remains after up
        axes = {0, 1, 2} - {self._up}
        self._fwd = _AXIS_TO_IDX[config.forward_axis]
        if self._fwd not in axes:
            # forward == up is invalid — fall back to first non-up axis so the
            # stabilizer is still usable when misconfigured.
            self._fwd = sorted(axes)[0]
        self._lat = sorted(axes - {self._fwd})[0]

        self._foot_indices = self._resolve_foot_indices()
        self._foot_toe_pairs = self._resolve_foot_toe_pairs()
        self._hips_idx = self._resolve("hips_name")
        self._lateral_pairs = self._resolve_lateral_pairs()

    # --------------------------------------------------------------- properties

    @property
    def config(self) -> FeetStabilizerConfig:
        return self._config

    @property
    def up_axis_idx(self) -> int:
        return self._up

    @property
    def foot_indices(self) -> tuple[int, ...]:
        """Indices (into the mapped-joint list) of the left / right foot."""
        return self._foot_indices

    # --------------------------------------------------------------- apply

    def apply(
        self, effectors: NDArray, *, return_stats: bool = False
    ) -> NDArray | tuple[NDArray, StabilizationStats]:
        """Run the full constraint chain on ``(F, M, 7)`` effectors.

        Returns a *new* array.  The original upstream returns an in-place
        mutated buffer; we copy up-front because modern numpy pipelines lean
        on immutability for cache friendliness.

        Order of operations mirrors soma's ``_postprocess_scaled_effectors_batched``:

        1. Foot planting       — lock horizontal positions when a foot is still.
        2. Min lateral sep     — push L/R pairs apart.
        3. Ground contact      — lift blocks that float in upright poses.
        4. Smoothing           — rate-limit the deltas we just added.
        """
        arr = np.asarray(effectors, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 7:
            raise ValueError(
                f"effectors must be (F, M, 7); got {arr.shape}"
            )
        if arr.shape[1] != len(self._joint_names):
            raise ValueError(
                f"effectors M={arr.shape[1]} does not match "
                f"len(joint_names)={len(self._joint_names)}"
            )

        pre_constraints_pos = arr[..., 0:3].copy()
        out = arr.copy()

        planted = self._apply_foot_planting(out)
        self._apply_min_lateral_separation(out)
        ground_corr = self._apply_ground_contact(out)
        smoothed_frames = self._smooth_corrections(out, pre_constraints_pos)

        if not return_stats:
            return out

        stats = StabilizationStats(
            ground_corrections=ground_corr.astype(np.float32, copy=False),
            planted_frames=planted,
            smoothed_frames=int(smoothed_frames),
        )
        return out, stats

    # --------------------------------------------------------------- planting

    def _apply_foot_planting(self, effectors: NDArray) -> NDArray:
        """Horizontal lock when a foot's horizontal velocity stays below threshold.

        Returns an ``(num_pairs, F)`` bool mask of planted frames (for stats).
        """
        cfg = self._config
        pairs = self._foot_toe_pairs
        if cfg.foot_planting_velocity_threshold <= 0.0 or not pairs:
            return np.zeros((len(pairs), effectors.shape[0]), dtype=bool)

        F = effectors.shape[0]
        if F < 3:
            return np.zeros((len(pairs), F), dtype=bool)

        vel_thresh = cfg.foot_planting_velocity_threshold
        height_limit = cfg.ground_contact_z + cfg.foot_planting_height_margin
        release_n = max(1, cfg.foot_planting_release_frames)
        h_axes = [self._fwd, self._lat]
        up = self._up

        planted_all = np.zeros((len(pairs), F), dtype=bool)

        for p_idx, (fi, ti) in enumerate(pairs):
            horiz = effectors[:, fi, h_axes].copy()
            h = effectors[:, fi, up]

            vel_h = np.zeros(F)
            vel_h[1:] = np.linalg.norm(np.diff(horiz, axis=0), axis=1)
            planted = (vel_h < vel_thresh) & (h < height_limit)
            planted_all[p_idx] = planted

            lock_pos = None
            toe_delta = np.zeros(2) if ti is not None else None

            for f in range(F):
                if planted[f]:
                    if lock_pos is None:
                        lock_pos = horiz[f].copy()
                        if ti is not None:
                            toe_delta = effectors[f, ti, h_axes] - horiz[f]
                    effectors[f, fi, h_axes[0]] = lock_pos[0]
                    effectors[f, fi, h_axes[1]] = lock_pos[1]
                    if ti is not None:
                        effectors[f, ti, h_axes[0]] = lock_pos[0] + toe_delta[0]
                        effectors[f, ti, h_axes[1]] = lock_pos[1] + toe_delta[1]
                else:
                    if lock_pos is not None:
                        for k in range(release_n):
                            target_f = f + k
                            if target_f >= F:
                                break
                            blend = (k + 1) / (release_n + 1)
                            orig = horiz[target_f]
                            blended = lock_pos * (1.0 - blend) + orig * blend
                            effectors[target_f, fi, h_axes[0]] = blended[0]
                            effectors[target_f, fi, h_axes[1]] = blended[1]
                            if ti is not None:
                                effectors[target_f, ti, h_axes[0]] = blended[0] + toe_delta[0]
                                effectors[target_f, ti, h_axes[1]] = blended[1] + toe_delta[1]
                        lock_pos = None

        return planted_all

    # --------------------------------------------------------- lateral separation

    def _apply_min_lateral_separation(self, effectors: NDArray) -> None:
        cfg = self._config
        if cfg.min_lateral_separation <= 0.0 or not self._lateral_pairs:
            return
        la = self._lat
        min_dist = cfg.min_lateral_separation

        for f in range(effectors.shape[0]):
            for li, ri in self._lateral_pairs:
                ll = effectors[f, li, la]
                rl = effectors[f, ri, la]
                gap = ll - rl
                if gap < min_dist:
                    mid = (ll + rl) * 0.5
                    half = min_dist * 0.5
                    effectors[f, li, la] = mid + half
                    effectors[f, ri, la] = mid - half

    # --------------------------------------------------------- ground contact

    def _apply_ground_contact(self, effectors: NDArray) -> NDArray:
        """Lift uprightly-posed effector blocks so feet don't float.

        Returns ``(F,)`` the per-frame correction magnitude (0 when skipped).
        """
        cfg = self._config
        corrections = np.zeros(effectors.shape[0], dtype=np.float32)
        if cfg.ground_contact_z <= 0.0 or not self._foot_indices:
            return corrections

        ref_h = cfg.ground_contact_z
        max_correction = cfg.max_ground_correction
        blend_range = cfg.ground_uprightness_range
        up = self._up
        hips_idx = self._hips_idx

        for f in range(effectors.shape[0]):
            min_foot_h = min(effectors[f, idx, up] for idx in self._foot_indices)

            uprightness = 1.0
            if hips_idx >= 0:
                uprightness = float(np.clip(
                    (effectors[f, hips_idx, up] - min_foot_h) / max(blend_range, 1e-6),
                    0.0, 1.0,
                ))
            if uprightness < 0.01:
                continue

            excess = min_foot_h - ref_h
            if excess > 0.002:
                correction = min(excess, max_correction) * uprightness
                effectors[f, :, up] -= correction
                corrections[f] = -correction

        return corrections

    # --------------------------------------------------------- smoothing

    def _smooth_corrections(
        self, constrained: NDArray, unconstrained: NDArray,
    ) -> int:
        """Rate-limit per-effector position corrections across frames.

        Args:
            constrained: ``(F, M, 7)`` effectors we just constrained.  Only
                ``[..., 0:3]`` is mutated; quats stay untouched.
            unconstrained: ``(F, M, 3)`` pre-constraint positions.

        Returns:
            Number of frames that had at least one DOF clamped.
        """
        max_rate = self._config.smoothing_max_rate
        if max_rate <= 0.0 or constrained.shape[0] < 2:
            return 0

        F = constrained.shape[0]
        corrections = constrained[:, :, 0:3] - unconstrained
        clamped_frames = 0

        for f in range(1, F):
            delta = corrections[f] - corrections[f - 1]
            magnitudes = np.linalg.norm(delta, axis=1, keepdims=True)
            exceeded = magnitudes > max_rate
            if not np.any(exceeded):
                continue
            clamped_frames += 1
            scale = np.where(
                exceeded, max_rate / np.maximum(magnitudes, 1e-8), 1.0,
            )
            corrections[f] = corrections[f - 1] + delta * scale

        constrained[:, :, 0:3] = unconstrained + corrections
        return clamped_frames

    # --------------------------------------------------------- name resolution

    def _resolve(self, attr: str) -> int:
        name = getattr(self._config, attr)
        if name is None:
            return -1
        try:
            return self._joint_names.index(name)
        except ValueError:
            return -1

    def _resolve_foot_indices(self) -> tuple[int, ...]:
        out = []
        for attr in ("left_foot_name", "right_foot_name"):
            idx = self._resolve(attr)
            if idx >= 0:
                out.append(idx)
        return tuple(out)

    def _resolve_foot_toe_pairs(self) -> tuple[tuple[int, int | None], ...]:
        pairs = []
        for foot_attr, toe_attr in (
            ("left_foot_name", "left_toe_name"),
            ("right_foot_name", "right_toe_name"),
        ):
            foot_idx = self._resolve(foot_attr)
            if foot_idx < 0:
                continue
            toe_idx = self._resolve(toe_attr)
            pairs.append((foot_idx, toe_idx if toe_idx >= 0 else None))
        return tuple(pairs)

    def _resolve_lateral_pairs(self) -> tuple[tuple[int, int], ...]:
        pairs = []
        for left, right in self._config.lateral_pairs:
            li = self._joint_names.index(left) if left in self._joint_names else -1
            ri = self._joint_names.index(right) if right in self._joint_names else -1
            if li >= 0 and ri >= 0:
                pairs.append((li, ri))
        return tuple(pairs)
