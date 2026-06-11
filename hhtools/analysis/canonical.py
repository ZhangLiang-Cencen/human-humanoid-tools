# SPDX-License-Identifier: Apache-2.0
"""Cross-source projection onto a canonical motion-feature representation.

Both human source clips (:class:`~hhtools.core.motion.Motion`, any rig) and
retargeted robot trajectories (:class:`~hhtools.io.robot_csv.RobotCSV`) are
projected onto the same :class:`CanonicalMotionFeatures` so that downstream
metrics, tags and embeddings can be computed by one code path and mixed in one
scatter plot.

The human path reuses :func:`hhtools.retarget.newton_basic.human_aliases.auto_source_to_canonical`
to rename arbitrary bone names (SMPL / BVH / Mixamo / holosoma / parc_ms) into
the canonical hhtools skeleton (``configs/skeleton_presets/canonical_human.yaml``).

The robot path is intentionally lighter: it reads the floating-base root from the
first 7 CSV columns and exposes the actuated DOF angles directly (``dof_q``),
which is closer to the LIMMT joint-velocity signal than FK-derived positions and
needs no URDF.  Foot positions are unavailable without FK, so terrain-aware (L2)
metrics gracefully degrade for robot clips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Canonical joint slots we care about for analysis (subset of canonical_human.yaml).
CANONICAL_JOINTS: tuple[str, ...] = (
    "hips",
    "chest",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
)

_LEG_JOINTS: tuple[str, ...] = (
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)
_ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)
_FOOT_JOINTS: tuple[str, ...] = ("left_ankle", "right_ankle", "left_foot", "right_foot")


@dataclass
class CanonicalMotionFeatures:
    """A source-agnostic per-frame kinematic summary used by all analysis code.

    Attributes:
        fps: frames per second.
        source_kind: ``"human"`` or ``"robot"``.
        root_pos: ``(F, 3)`` world root (hips/pelvis/base) position in metres,
            up-axis Z.
        root_quat: ``(F, 4)`` xyzw world root orientation (may be identity for
            position-only sources such as holosoma).
        joint_pos: canonical-name -> ``(F, 3)`` world position. Only present
            joints are stored.
        dof_q: optional ``(F, N)`` actuated joint angles (robot CSV) used for the
            LIMMT-style ``q̇`` complexity signal. ``None`` for human clips.
        up_axis: always ``"Z"`` (hhtools internal convention).
    """

    fps: float
    source_kind: str
    root_pos: NDArray
    root_quat: NDArray
    joint_pos: dict[str, NDArray] = field(default_factory=dict)
    dof_q: NDArray | None = None
    up_axis: str = "Z"

    # ----------------------------------------------------------------- shape

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def duration(self) -> float:
        if self.num_frames < 2:
            return 0.0
        return (self.num_frames - 1) / max(self.fps, 1e-6)

    @property
    def delta_time(self) -> float:
        return 1.0 / max(self.fps, 1e-6)

    def has(self, name: str) -> bool:
        return name in self.joint_pos

    # ----------------------------------------------------------------- queries

    def joint_velocity(self, name: str) -> NDArray | None:
        """Finite-difference velocity (m/s) of canonical joint ``name``."""
        pos = self.joint_pos.get(name)
        if pos is None:
            return None
        return _finite_diff(pos, self.delta_time)

    def present_joints(self, names: tuple[str, ...]) -> list[str]:
        return [n for n in names if n in self.joint_pos]

    def leg_joint_names(self) -> list[str]:
        return self.present_joints(_LEG_JOINTS)

    def arm_joint_names(self) -> list[str]:
        return self.present_joints(_ARM_JOINTS)

    def foot_joint_names(self) -> list[str]:
        """Distal foot joints actually present (ankle preferred, foot/toe fallback)."""
        out: list[str] = []
        for side in ("left", "right"):
            ankle = f"{side}_ankle"
            foot = f"{side}_foot"
            if ankle in self.joint_pos:
                out.append(ankle)
            elif foot in self.joint_pos:
                out.append(foot)
        return out

    def all_joint_positions(self) -> NDArray:
        """Stack of all present canonical joint positions: ``(F, J, 3)``."""
        cols = [self.joint_pos[n] for n in CANONICAL_JOINTS if n in self.joint_pos]
        if not cols:
            return self.root_pos[:, None, :]
        return np.stack(cols, axis=1)


def _finite_diff(values: NDArray, dt: float) -> NDArray:
    """First difference along time; first frame copies the second to keep length."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[0] < 2:
        return np.zeros_like(arr)
    out = np.empty_like(arr)
    out[1:] = (arr[1:] - arr[:-1]) / max(dt, 1e-6)
    out[0] = out[1]
    return np.clip(out, -1e4, 1e4)


def _invert_rename(rename: dict[str, str], bone_names: list[str]) -> dict[str, int]:
    """canonical-name -> source bone *index*.

    ``rename`` maps source-name -> canonical-name. When several source bones map
    to the same canonical (e.g. ``spine1`` and ``spine2`` -> ``spine``) the first
    encountered wins, except feet where the distal toe (last) is preferred.
    """
    name_to_idx = {n: i for i, n in enumerate(bone_names)}
    out: dict[str, int] = {}
    for src, canon in rename.items():
        idx = name_to_idx.get(src)
        if idx is None:
            continue
        if canon in out and canon not in ("left_foot", "right_foot"):
            continue
        out[canon] = idx
    return out


def project_motion(motion) -> CanonicalMotionFeatures:
    """Project a human :class:`~hhtools.core.motion.Motion` to canonical features."""
    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    bone_names = list(motion.bone_names)
    rename = auto_source_to_canonical(tuple(bone_names))
    canon_to_idx = _invert_rename(rename, bone_names)

    positions = np.asarray(motion.positions, dtype=np.float32)  # (F, B, 3)
    quats = np.asarray(motion.quaternions, dtype=np.float32)  # (F, B, 4)

    joint_pos: dict[str, NDArray] = {}
    for canon in CANONICAL_JOINTS:
        idx = canon_to_idx.get(canon)
        if idx is not None:
            joint_pos[canon] = positions[:, idx, :].copy()

    # Root: prefer mapped hips, else the hierarchy root (index 0).
    root_idx = canon_to_idx.get("hips")
    if root_idx is None:
        roots = motion.hierarchy.root_indices()
        root_idx = roots[0] if roots else 0
    root_pos = positions[:, root_idx, :].copy()
    root_quat = quats[:, root_idx, :].copy()

    return CanonicalMotionFeatures(
        fps=float(motion.framerate),
        source_kind="human",
        root_pos=root_pos,
        root_quat=root_quat,
        joint_pos=joint_pos,
        dof_q=None,
        up_axis=str(motion.up_axis),
    )


def project_robot_csv(csv) -> CanonicalMotionFeatures:
    """Project a :class:`~hhtools.io.robot_csv.RobotCSV` to canonical features.

    Without a URDF we cannot FK the links, so only the floating-base root and the
    raw actuated DOF angles are exposed.  ``joint_pos`` carries a single ``hips``
    entry (the root) so locomotion / inverted metrics still work; ``dof_q`` drives
    the joint-velocity complexity signal.
    """
    joint_q = np.asarray(csv.joint_q, dtype=np.float32)  # (F, 7 + N)
    if joint_q.shape[1] < 7:
        raise ValueError(f"robot CSV joint_q has too few columns: {joint_q.shape}")
    root_pos = joint_q[:, 0:3].copy()
    root_quat = joint_q[:, 3:7].copy()
    dof_q = joint_q[:, 7:].copy() if joint_q.shape[1] > 7 else None

    return CanonicalMotionFeatures(
        fps=float(csv.sample_rate),
        source_kind="robot",
        root_pos=root_pos,
        root_quat=root_quat,
        joint_pos={"hips": root_pos.copy()},
        dof_q=dof_q,
        up_axis="Z",
    )


def project_to_canonical(obj) -> CanonicalMotionFeatures:
    """Dispatch on the input type (``Motion`` vs ``RobotCSV``)."""
    from hhtools.io.robot_csv import RobotCSV

    if isinstance(obj, RobotCSV):
        return project_robot_csv(obj)
    return project_motion(obj)


__all__ = [
    "CANONICAL_JOINTS",
    "CanonicalMotionFeatures",
    "project_motion",
    "project_robot_csv",
    "project_to_canonical",
]
