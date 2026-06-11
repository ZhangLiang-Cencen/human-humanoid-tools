"""Canonical joint layouts for the SMPL / SMPL-H / SMPL-X body model families.

These tables follow the conventions used by the official ``smplx`` Python package and reflect
the kinematic trees baked into the MPI weights.  Joint indices are the native ordering produced
by ``SMPL.forward().joints`` (up to the declared number of joints -- the tail of that tensor
contains additional landmark / face / hand-tip regressor points that we do not propagate
through our internal :class:`hhtools.core.motion.Motion` representation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# SMPL (24 joints)
# ---------------------------------------------------------------------------
SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

SMPL_PARENTS: tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)

# ---------------------------------------------------------------------------
# SMPL-H (52 joints = 22 body + 15 finger joints per hand)
# ---------------------------------------------------------------------------
_HAND_FINGER_NAMES: tuple[str, ...] = (
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
)  # 15 finger joints per hand

SMPLH_JOINT_NAMES: tuple[str, ...] = (
    *SMPL_JOINT_NAMES[:22],  # 22 body joints (drop left_hand / right_hand)
    *(f"left_{n}" for n in _HAND_FINGER_NAMES),
    *(f"right_{n}" for n in _HAND_FINGER_NAMES),
)

SMPLH_PARENTS: tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,  # body
    # left hand (root joints parented to left_wrist=20)
    20,
    22,
    23,
    20,
    25,
    26,
    20,
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    # right hand (root joints parented to right_wrist=21)
    21,
    37,
    38,
    21,
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
)

# ---------------------------------------------------------------------------
# SMPL-X (55 joints = 22 body + 3 head + 15 fingers per hand)
# ---------------------------------------------------------------------------
SMPLX_JOINT_NAMES: tuple[str, ...] = (
    *SMPL_JOINT_NAMES[:22],  # 22 body joints
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",  # head (3)
    *(f"left_{n}" for n in _HAND_FINGER_NAMES),
    *(f"right_{n}" for n in _HAND_FINGER_NAMES),
)

SMPLX_PARENTS: tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,  # body (22)
    15,
    15,
    15,  # jaw, l_eye, r_eye parent=head(15)
    # left hand (root joints parented to left_wrist=20)
    20,
    25,
    26,
    20,
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    20,
    37,
    38,
    # right hand (root joints parented to right_wrist=21)
    21,
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
    21,
    52,
    53,
)


@dataclass(frozen=True)
class BodyModelLayout:
    """Descriptor for a body model family's kinematic tree."""

    family: Literal["smpl", "smplh", "smplx"]
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    num_body_joints: int  # number of pose joints (excluding root_orient/expression/etc.)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def parents_array(self) -> np.ndarray:
        return np.asarray(self.parents, dtype=np.int32)


SMPL_LAYOUT = BodyModelLayout("smpl", SMPL_JOINT_NAMES, SMPL_PARENTS, num_body_joints=23)
SMPLH_LAYOUT = BodyModelLayout("smplh", SMPLH_JOINT_NAMES, SMPLH_PARENTS, num_body_joints=51)
SMPLX_LAYOUT = BodyModelLayout("smplx", SMPLX_JOINT_NAMES, SMPLX_PARENTS, num_body_joints=54)


def layout_for(family: str) -> BodyModelLayout:
    family = family.lower()
    if family == "smpl":
        return SMPL_LAYOUT
    if family == "smplh":
        return SMPLH_LAYOUT
    if family == "smplx":
        return SMPLX_LAYOUT
    raise ValueError(f"Unknown body model family: {family!r}. Expected 'smpl', 'smplh' or 'smplx'.")


__all__ = [
    "BodyModelLayout",
    "SMPL_JOINT_NAMES",
    "SMPL_LAYOUT",
    "SMPL_PARENTS",
    "SMPLH_JOINT_NAMES",
    "SMPLH_LAYOUT",
    "SMPLH_PARENTS",
    "SMPLX_JOINT_NAMES",
    "SMPLX_LAYOUT",
    "SMPLX_PARENTS",
    "layout_for",
]
